"""Score Crucible on the held-out evaluation set.

Usage:
    python evaluation/score_heldout.py [--grounded] [--model MODEL]

Runs the same semantic screen the application runs, with the same shuffle count
and the same per-model criterion wording, and scores its verdicts against the
answer key. The correlation screen is scored on identical columns so the two
are comparable.

The correlation baseline is reported twice: once at the fixed threshold the
application ships, and once at the threshold that maximises its F1 on the
answers. The second is not a fair comparison in the baseline's disfavour — it
lets the baseline see the answer key and pick its best setting, which no real
user could do. It is reported because a baseline should be beaten at its best,
not at a setting chosen for it.
"""

import argparse
import asyncio
import json
import os
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
from crucible import models, screen, stats            # noqa: E402

SHUFFLES = 3
from crucible.providers import resolve                # noqa: E402


def counts(flagged: set, truth: set, universe: set) -> dict:
    true_positive = len(flagged & truth)
    false_positive = len(flagged - truth)
    false_negative = len(truth - flagged)
    precision = true_positive / (true_positive + false_positive) if flagged else 0.0
    recall = true_positive / len(truth) if truth else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if truth and (precision + recall) else 0.0)
    return {
        "tp": true_positive, "fp": false_positive, "fn": false_negative,
        "precision": precision, "recall": recall, "f1": f1,
        "flagged": len(flagged), "columns": len(universe),
    }


def best_correlation_threshold(correlations: dict, truth: set) -> tuple[float, dict]:
    """The threshold that maximises this baseline's F1 on the answers."""
    universe = set(correlations)
    candidates = sorted({round(abs(v), 4) for v in correlations.values() if v is not None})
    best = (0.0, None)
    for threshold in candidates:
        flagged = {c for c, v in correlations.items() if v is not None and abs(v) >= threshold}
        scored = counts(flagged, truth, universe)
        if scored["f1"] > best[0]:
            best = (scored["f1"], (threshold, scored))
    return best[1] or (0.5, counts(set(), truth, universe))


async def run(model: str, grounded: bool) -> dict:
    answer_key = json.load(open(os.path.join(HERE, "ANSWER_KEY.json")))
    criterion = models.criterion_for(model)
    provider = resolve(models.provider_for(model))
    results = []

    print(f"model {model} | criterion {criterion} | shuffles {SHUFFLES} "
          f"| dictionary {'attached' if grounded else 'withheld'}\n")

    for entry in answer_key["datasets"]:
        name, target = entry["name"], entry["target"]
        table = pd.read_csv(os.path.join(HERE, "heldout", f"{name}.csv"))
        columns = [c for c in table.columns if c != target]
        truth = set(entry.get("positives", {}))

        descriptions = None
        if grounded:
            dictionary = pd.read_csv(os.path.join(HERE, "heldout", f"{name}_dictionary.csv"))
            descriptions = dict(zip(dictionary["column"], dictionary["description"]))

        verdicts, _ = await screen.run_semantic_screen(
            provider, model, columns, target, entry["prediction_point"],
            shuffle_count=SHUFFLES, descriptions=descriptions, criterion=criterion,
        )
        model_flags = {c for c, v in verdicts.items() if v["verdict"] == "LEAK"}

        statistical = stats.statistical_screen(table, target)
        correlations = {c: (statistical.get(c) or {}).get("correlation") for c in columns}
        fixed_flags = {c for c, v in correlations.items()
                       if v is not None and abs(v) >= stats.CORRELATION_THRESHOLD}
        oracle_threshold, oracle_scored = best_correlation_threshold(correlations, truth)

        record = {
            "dataset": name, "columns": len(columns), "positives": len(truth),
            "model": counts(model_flags, truth, set(columns)),
            "correlation_fixed": counts(fixed_flags, truth, set(columns)),
            "correlation_oracle": oracle_scored, "oracle_threshold": oracle_threshold,
            "missed": sorted(truth - model_flags),
            "false_positives": sorted(model_flags - truth),
        }
        results.append(record)
        _print_dataset(record)

    await provider.aclose()
    return {"model": model, "criterion": criterion, "grounded": grounded,
            "shuffles": SHUFFLES, "datasets": results, "pooled": _pool(results)}


def _pool(results: list) -> dict:
    pooled = {}
    for arm in ("model", "correlation_fixed", "correlation_oracle"):
        tp = sum(r[arm]["tp"] for r in results)
        fp = sum(r[arm]["fp"] for r in results)
        fn = sum(r[arm]["fn"] for r in results)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        pooled[arm] = {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 4),
                       "recall": round(recall, 4), "f1": round(f1, 4)}
    return pooled


def _print_dataset(record: dict):
    print(f"--- {record['dataset']}  {record['columns']} columns, "
          f"{record['positives']} documented positives")
    for label, key in (("crucible", "model"), ("correlation @0.5", "correlation_fixed"),
                       ("correlation @best", "correlation_oracle")):
        s = record[key]
        recall = "  n/a" if s["recall"] != s["recall"] else f"{s['recall']:.3f}"
        print(f"      {label:<18} P {s['precision']:.3f}  R {recall}  F1 {s['f1']:.3f}"
              f"   tp {s['tp']:>2} fp {s['fp']:>2} fn {s['fn']:>2}")
    if record["missed"]:
        print(f"      missed          : {', '.join(record['missed'])}")
    if record["false_positives"]:
        print(f"      false positives : {', '.join(record['false_positives'])}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("CRUCIBLE_RECALL_MODEL", models.DEFAULT_MODEL))
    parser.add_argument("--grounded", action="store_true",
                        help="attach each dataset's data dictionary")
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()

    report = asyncio.run(run(arguments.model, arguments.grounded))
    pooled = report["pooled"]
    print("=" * 72)
    print(f"POOLED over {len(report['datasets'])} datasets")
    for label, key in (("Crucible", "model"), ("correlation @0.5", "correlation_fixed"),
                       ("correlation @best (sees answers)", "correlation_oracle")):
        s = pooled[key]
        print(f"  {label:<34} P {s['precision']:.3f}  R {s['recall']:.3f}  F1 {s['f1']:.3f}"
              f"   tp {s['tp']} fp {s['fp']} fn {s['fn']}")

    path = arguments.out or os.path.join(
        HERE, f"results_{'grounded' if arguments.grounded else 'names_only'}.json")
    json.dump(report, open(path, "w"), indent=2)
    print(f"\nwritten to {path}")
