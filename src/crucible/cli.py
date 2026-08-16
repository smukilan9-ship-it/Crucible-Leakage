"""Command line for Crucible.

Three commands, matching the three things the tool does:

    crucible audit    data.csv --target y --at "at approval, before repayment"
    crucible impact   data.csv --target y --drop a --drop b
    crucible models

`audit` prints a report and, unless asked, changes nothing. Writing a cleaned
file takes an explicit flag, and that flag will not run without either
`--yes` or a terminal to ask at, because a tool that silently deletes columns
is the thing this one exists to argue against.
"""

import argparse
import asyncio
import difflib
import json
import os
import sys
import time

import pandas as pd

from . import __version__, models
from .audit import AuditRequest, flagged_columns, run_audit
from .impact import ImpactError, quantify
from .intake import IntakeError, load_dictionary, load_table
from .providers import KEY_VARIABLES, ProviderError, QuotaExhausted, resolve

VERDICT_MARK = {"LEAK": "LEAK", "ABSTAIN": "?", "OK": "ok"}
BUCKET_TEXT = {
    "A": "both screens", "B": "model only", "C": "statistics only", "D": "neither",
}


def _require_column(table: pd.DataFrame, name: str, what: str) -> None:
    """Fail on a column that is not there, saying what was probably meant.

    A pandas KeyError reaches the top level as the bare string `'servived'`,
    which tells a reader nothing about which argument was wrong or what the
    alternatives were. One typo in `--target` is the most likely mistake anyone
    makes with this tool and it deserves better than that.
    """
    if name in table.columns:
        return
    close = difflib.get_close_matches(name, list(table.columns), n=3, cutoff=0.6)
    hint = f" Did you mean {', '.join(repr(c) for c in close)}?" if close else ""
    raise IntakeError(
        f"{what} {name!r} is not a column of this table.{hint} "
        f"The table has {len(table.columns)} columns; run with --json or open the "
        f"file to see them.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="crucible",
        description="Find columns that encode the outcome they are used to predict.",
    )
    parser.add_argument("--version", action="version", version=f"crucible {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="audit a table for target leakage")
    audit.add_argument("table", help="path to a CSV or Parquet file")
    audit.add_argument("--target", required=True, help="the column being predicted")
    audit.add_argument(
        "--at", "--prediction-point", dest="prediction_point", required=True,
        help="when the prediction happens in real life; a phrase, not a date. "
             "This is the one input nothing can infer for you.")
    audit.add_argument("--model", default=models.DEFAULT_MODEL)
    audit.add_argument("--dictionary", help="CSV of column,description — grounds "
                                            "every verdict in your own documentation")
    audit.add_argument("--shuffles", type=int, default=None,
                       help="how many column orders to vote across. Left unset, "
                            "each model gets the count its measured order "
                            "sensitivity justifies; `crucible models` prints it.")
    audit.add_argument("--json", action="store_true", help="emit the full report as JSON")
    audit.add_argument("--write-clean", metavar="PATH",
                       help="write a copy with the flagged columns removed")
    audit.add_argument("--yes", action="store_true",
                       help="do not ask before writing a cleaned file")
    audit.add_argument("--measure", action="store_true",
                       help="also fit every arm and report what the leaks were worth")
    audit.add_argument("--include-contested", action="store_true",
                       help="treat contested columns as leaks; off by default, because "
                            "a contested column is one the tool declines to decide")
    audit.add_argument("--quiet", "-q", action="store_true",
                       help="suppress progress on stderr")

    impact = sub.add_parser("impact", help="measure what named columns are worth")
    impact.add_argument("table")
    impact.add_argument("--target", required=True)
    impact.add_argument("--drop", action="append", default=[], metavar="COLUMN",
                        help="repeatable")
    impact.add_argument("--against-correlation", action="store_true",
                        help="add a third arm using what a correlation threshold would "
                             "have removed instead, so the comparison answers whether "
                             "the cheap check would have done the job")
    impact.add_argument("--json", action="store_true")
    impact.add_argument("--quiet", "-q", action="store_true",
                        help="suppress the fit counter on stderr")

    sub.add_parser("models", help="list the models on offer")

    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "audit":
            return _audit(arguments)
        if arguments.command == "impact":
            return _impact(arguments)
        return _models()
    except (IntakeError, ImpactError, KeyError) as error:
        print(f"crucible: {error}", file=sys.stderr)
        return 2
    except QuotaExhausted as error:
        print(f"crucible: {error}", file=sys.stderr)
        return 4
    except ProviderError as error:
        print(f"crucible: the model could not be reached: {error}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\ncrucible: interrupted", file=sys.stderr)
        return 130


def _audit(arguments) -> int:
    table = load_table(arguments.table)
    _require_column(table, arguments.target, "the target column")
    descriptions = None
    if arguments.dictionary:
        features = [c for c in table.columns if c != arguments.target]
        report = load_dictionary(arguments.dictionary, features, arguments.target)
        descriptions = report["descriptions"]
        print(f"dictionary: {len(descriptions)} of {len(features)} columns documented",
              file=sys.stderr)

    async def on_stage(_stage, detail):
        if not arguments.quiet:
            print(f"  · {detail}", file=sys.stderr)

    print(f"auditing {arguments.table}: {len(table):,} rows, "
          f"{len(table.columns)} columns, target {arguments.target!r}", file=sys.stderr)
    if not arguments.quiet:
        print(f"  {models.shuffle_rationale(arguments.model)}", file=sys.stderr)
    result = asyncio.run(run_audit(
        AuditRequest(
            table=table, target=arguments.target,
            prediction_point=arguments.prediction_point,
            model=arguments.model, descriptions=descriptions,
            shuffles=arguments.shuffles or models.shuffles_for(arguments.model),
        ),
        on_stage=on_stage,
    ))

    if result.get("semantic_skipped"):
        print("\nThe column names carry no readable meaning, so the semantic screen was\n"
              "skipped and only the statistical screen ran. What follows is a baseline,\n"
              "not an audit.", file=sys.stderr)

    if arguments.json:
        json.dump(result, sys.stdout, indent=2, default=str)
        print()
    else:
        _print_report(result)

    leaks = flagged_columns(result, include_contested=arguments.include_contested)
    if arguments.measure and leaks:
        baseline = sorted(c for c, entry in result["statistical"].items()
                          if entry.get("flagged"))
        _print_impact(quantify(table, arguments.target, leaks,
                               baseline_drop_list=baseline or None,
                               on_event=_fit_progress(arguments.quiet)))

    if arguments.write_clean:
        if not leaks:
            print("nothing flagged, so nothing to remove", file=sys.stderr)
            return 0
        if not _confirm(leaks, arguments.write_clean, arguments.yes):
            print("not written", file=sys.stderr)
            return 1
        table.drop(columns=leaks).to_csv(arguments.write_clean, index=False)
        print(f"wrote {arguments.write_clean} without {', '.join(leaks)}", file=sys.stderr)
    return 0


def _confirm(leaks: list[str], path: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        print("crucible: refusing to write a cleaned file unattended; pass --yes "
              "if you have read the report and accept the drops", file=sys.stderr)
        return False
    print(f"\nremove {len(leaks)} column(s) — {', '.join(leaks)} — and write {path}?")
    return input("[y/N] ").strip().lower() in {"y", "yes"}


def _print_report(result: dict) -> None:
    semantic, statistical = result["semantic"], result["statistical"]
    width = max((len(c) for c in result["columns"]), default=10)
    print(f"\nmodel {result['model']} · {result['criterion']} criterion · "
          f"{result['shuffles']} orders"
          f"{' · grounded in your dictionary' if result['grounded'] else ''}\n")
    print(f"{'column'.ljust(width)}  {'verdict':<12} {'votes':<6} {'|r|':<7} which screen")
    print("-" * (width + 45))

    for column in sorted(result["columns"],
                         key=lambda c: (semantic[c]["verdict"] != "LEAK", c)):
        answer = semantic[column]
        correlation = (statistical.get(column) or {}).get("correlation")
        verdict = ("CONTESTED" if answer.get("contested")
                   else answer["mechanism"] or VERDICT_MARK.get(answer["verdict"], "?")
                   if answer["verdict"] == "LEAK"
                   else VERDICT_MARK.get(answer["verdict"], "?"))
        votes = f"{answer['leak_votes']}/{answer['shuffles_counted']}"
        r = "  —  " if correlation is None else f"{abs(correlation):.3f}"
        print(f"{column.ljust(width)}  {verdict:<12} {votes:<6} {r:<7} "
              f"{BUCKET_TEXT[result['buckets'][column]]}")

    for column in sorted(result["columns"]):
        answer = semantic[column]
        if answer["verdict"] != "LEAK":
            continue
        print(f"\n{column}")
        for reason in answer.get("reasons", [])[:3]:
            print(f"    · {reason}")
        if answer.get("contested"):
            print(f"    ! contested: the documentation places this value at or before "
                  f"the prediction point, so the call is yours")

    leaks = flagged_columns(result)
    contested = [c for c in result["columns"] if semantic[c].get("contested")]
    print(f"\n{len(leaks)} flagged of {len(result['columns'])} columns"
          f" ({len(leaks) / max(len(result['columns']), 1):.0%} to review)"
          + (f", {len(contested)} contested and left to you" if contested else ""))


ARM_WORD = {"with_leaks": "with-leaks", "honest": "cleaned", "baseline": "correlation"}


def _print_impact(result: dict) -> None:
    gaps = [arms["inflation"]["macro_f1"] for arms in result["learners"].values()
            if arms["inflation"]["macro_f1"] is not None]
    # Said before the numbers, because a zero difference between two arms that
    # hold the same columns is not a finding about leakage.
    for pair in result.get("identical_arms", []):
        left, right = pair["arms"]
        print(f"\n  ! the {ARM_WORD[left]} and {ARM_WORD[right]} arms hold the same "
              f"{pair['n_features']} encoded columns, so the rows below are one fit "
              f"reported twice rather than a comparison.")
    if result.get("drops_ignored"):
        print(f"  ! not features of this table, so they removed nothing: "
              f"{', '.join(result['drops_ignored'])}")
    if gaps:
        print(f"\n  Your model scored {max(gaps):.3f} macro F1 higher than it deserved to.")
    print(f"\n  {result['n_rows_used']:,} rows, {result['n_classes']} classes "
          f"({', '.join(result['class_labels'])})"
          + (f", grouped on {result['group_column']}" if result.get("group_column") else ""))

    for learner, arms in result["learners"].items():
        print(f"\n  {learner.replace('_', ' ')}")
        rows = [("with leaks", arms["with_leaks"]), ("cleaned", arms["honest"])]
        if arms.get("baseline"):
            rows.append(("correlation", arms["baseline"]))
        print(f"    {'arm':<13} {'macro F1':>9} {'weighted':>9} {'mistakes':>9}")
        for label, arm in rows:
            print(f"    {label:<13} {arm['macro']['f1']:>9.4f} "
                  f"{arm['weighted']['f1']:>9.4f} {_errors(arm):>9,}")

        gap = arms["inflation"]
        print(f"    leaks overstated macro F1 by {gap['macro_f1']:+.4f}, "
              f"weighted by {gap['weighted_f1']:+.4f}")
        if arms.get("baseline_residual"):
            residual = arms["baseline_residual"]["macro_f1"]
            print(f"    a correlation threshold would have left {residual:+.4f} of that "
                  f"in place" if residual > 0.02 else
                  f"    a correlation threshold would have done about as well here "
                  f"({residual:+.4f})")


def _errors(arm: dict) -> int:
    return sum(cell for i, row in enumerate(arm["confusion"])
               for j, cell in enumerate(row) if i != j)


def _fit_progress(quiet: bool):
    """A line on stderr that says where the fit has got to and when it ends.

    This stage is minutes of silence on a wide table, which reads as a hang.
    The count is exact; the time left is the rate this machine is managing so
    far, which is the only honest estimate available and is wrong until a few
    folds have landed. It writes to stderr and rewrites one line, so piping
    stdout to a file still gets clean output.
    """
    if quiet:
        return None
    state = {"done": 0, "total": 0, "began": time.perf_counter()}
    stream = sys.stderr
    live = stream.isatty()

    def report(name: str, payload: dict) -> None:
        if name == "plan":
            configs = sum(payload["configs"].values())
            state["total"] = payload.get("distinct_arms", len(payload["arms"])) \
                * payload["folds"] * configs
            print(f"\nfitting {state['total']} models on identical folds: "
                  f"{payload['rows']:,} rows, {len(payload['arms'])} arms, "
                  f"{len(payload['learners'])} learner families",
                  file=stream, flush=True)
        elif name == "fold":
            state["done"] += 1
            if not state["total"]:
                return
            # A rewriting line is right for a terminal and wrong for a log
            # file, where it becomes ninety lines of noise. Redirected output
            # gets one line per tenth instead.
            step = max(1, state["total"] // 10)
            if not live and state["done"] % step and state["done"] != state["total"]:
                return
            elapsed = time.perf_counter() - state["began"]
            left = (elapsed / state["done"]) * (state["total"] - state["done"])
            line = (f"  {state['done']:>4} of {state['total']} · "
                    f"{_clock(left)} left    ")
            print(line + ("\r" if live else "\n"), end="", file=stream, flush=True)
        elif name == "arm_reused":
            print(f"\n  {payload['arm']}: same columns as an arm already fitted, "
                  f"so not refitted", file=stream, flush=True)
        elif name == "done":
            if live:
                print("\r" + " " * 44 + "\r", end="", file=stream, flush=True)

    return report


def _clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds}s" if seconds < 60 else f"{seconds // 60}m {seconds % 60:02d}s"


def _impact(arguments) -> int:
    table = load_table(arguments.table)
    _require_column(table, arguments.target, "the target column")
    if not arguments.drop:
        print("crucible: name at least one column with --drop", file=sys.stderr)
        return 2

    # Refused here rather than after the fit. Dropping nothing produces two
    # identical arms, and the difference between a fit and itself is zero for
    # reasons that have nothing to do with leakage. `quantify` reports the same
    # thing if it gets that far, but it costs minutes to say it.
    missing = [name for name in arguments.drop if name not in table.columns]
    if len(missing) == len(arguments.drop):
        _require_column(table, missing[0], "the column to drop")
    if missing:
        print(f"crucible: not columns of this table, so they will remove nothing: "
              f"{', '.join(missing)}", file=sys.stderr)

    baseline = None
    if arguments.against_correlation:
        from .stats import statistical_screen
        screened = statistical_screen(table, arguments.target)
        baseline = sorted(c for c, e in screened.items() if e.get("flagged"))
        print(f"correlation would drop: {', '.join(baseline) or 'nothing'}", file=sys.stderr)
    result = quantify(table, arguments.target, arguments.drop,
                      baseline_drop_list=baseline or None,
                      on_event=_fit_progress(arguments.quiet))
    if arguments.json:
        json.dump(result, sys.stdout, indent=2, default=str)
        print()
    else:
        _print_impact(result)
    return 0


def _models() -> int:
    for entry in models.CATALOGUE:
        mark = "*" if entry["recommended"] else " "
        variable = KEY_VARIABLES.get(entry["provider"], "")
        # Whether the key is present, never any part of the key itself. A model
        # listed as available that then fails for want of a credential is a
        # worse experience than being told up front.
        ready = "key found" if os.environ.get(variable) else f"set {variable}"
        print(f"{mark} {entry['id']}")
        print(f"    {entry['name']} · {entry['size']} · via {entry['provider']} "
              f"· {entry['criterion']} criterion · "
              f"{models.shuffles_for(entry['id'])} order"
              f"{'' if models.shuffles_for(entry['id']) == 1 else 's'} · {ready}")
        print(f"    {entry['note']}")
    print("\n* default. The criterion column is the wording of the derivation "
          "clause,\n  chosen per model from measured results; neither wording is "
          "uniformly better.")
    print("  Keys are read from the environment and never from an argument, so "
          "they\n  stay out of your shell history.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
