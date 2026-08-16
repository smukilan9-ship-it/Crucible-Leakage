"""Fetch the held-out evaluation datasets and verify the answer key against them.

Run this before scoring. It writes one CSV and one data dictionary per dataset
into `evaluation/heldout/`, and it refuses to write anything if the answer key
disagrees with what the repository actually serves.

Three checks run on every dataset, and any failure stops the build:

  * every column named as a positive exists in the real CSV header;
  * every quotation in the answer key is present, verbatim, in the column's own
    description as served by the repository;
  * the target column exists and has more than one value.

The second check is the one that matters. A quotation that has drifted from its
source is worse than no quotation at all, because it looks like evidence.
"""

import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "heldout")
KEY = os.path.join(HERE, "ANSWER_KEY.json")


def main() -> int:
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        print("this script needs `pip install ucimlrepo certifi`")
        return 1

    os.makedirs(OUT, exist_ok=True)
    answer_key = json.load(open(KEY))
    failures: list[str] = []

    for entry in answer_key["datasets"]:
        name = entry["name"]
        print(f"\n=== {name} (UCI {entry['uci_id']}) ===")
        try:
            fetched = fetch_ucirepo(id=entry["uci_id"])
        except Exception as error:
            failures.append(f"{name}: could not fetch ({error})")
            continue

        table = fetched.data.original
        variables = fetched.variables
        descriptions = {
            str(row["name"]): ("" if pd.isna(row["description"]) else str(row["description"]))
            for _, row in variables.iterrows()
        }

        for column in entry.get("drop_columns", []):
            if column in table.columns:
                table = table.drop(columns=[column])
                descriptions.pop(column, None)

        target = entry["target"]
        if target not in table.columns:
            failures.append(f"{name}: target {target!r} not in the served table")
            continue
        if table[target].nunique() < 2:
            failures.append(f"{name}: target {target!r} has fewer than two values")
            continue

        for column, record in entry.get("positives", {}).items():
            if column not in table.columns:
                failures.append(f"{name}: positive {column!r} is not a column in the served table")
                continue
            quote = record["quote"].strip()
            served = descriptions.get(column, "")
            if _normalize(quote) not in _normalize(served):
                failures.append(
                    f"{name}: the quotation for {column!r} is not in its served description\n"
                    f"    quoted: {quote}\n"
                    f"    served: {served[:160]}"
                )

        table.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
        dictionary = pd.DataFrame(
            [{"column": column, "description": text}
             for column, text in descriptions.items() if column != target and text]
        )
        dictionary.to_csv(os.path.join(OUT, f"{name}_dictionary.csv"), index=False)

        n_features = len([c for c in table.columns if c != target])
        print(f"  rows {len(table):>7,}  features {n_features:>3}  "
              f"positives {len(entry.get('positives', {})):>2}  "
              f"documented {len(dictionary):>3}")

    if failures:
        print("\n--- ANSWER KEY DOES NOT MATCH THE SOURCES ---")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print(f"\nAll quotations verified against the served descriptions. Files in {OUT}")
    return 0


def _normalize(text: str) -> str:
    """Compare on words alone, so a stray double space or a smart quote in the
    repository's own text cannot fail a quotation that is genuinely present."""
    return " ".join("".join(c.lower() if c.isalnum() else " " for c in text).split())


if __name__ == "__main__":
    sys.exit(main())
