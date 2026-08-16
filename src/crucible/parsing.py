"""Turning a model reply into per-column verdicts.

Kept separate from every provider because the shape of a reply is a property of
the prompt, not of who served it, and because this is the one piece of the
model path that can be tested without a network.
"""

import json
import re


def parse_verdicts(text: str, expected_columns: list[str]) -> tuple[dict, str]:
    """Parse a model reply into per-column verdicts.

    Returns ({column: {verdict, mechanism, reason}}, parser_used) where
    parser_used is "json" when the reply parsed cleanly and "salvage" when
    individual objects had to be pulled out of truncated or prose-wrapped
    output. Only rows naming a real column are kept; any column name the
    model invented is dropped. Checking that enough columns were answered
    (the coverage floor) is the caller's job.
    """
    expected = set(expected_columns)

    def keep_known_columns(rows) -> dict:
        verdicts = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            column = row.get("column")
            if column in expected:
                verdicts[column] = {
                    "verdict": str(row.get("verdict", "ABSTAIN")).upper(),
                    "mechanism": row.get("mechanism"),
                    "reason": str(row.get("reason", "")),
                }
        return verdicts

    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        rows = json.loads(stripped)
        if isinstance(rows, list):
            return keep_known_columns(rows), "json"
    except json.JSONDecodeError:
        pass

    # Salvage path: pull individual {...} objects out of output that was
    # truncated mid-array or wrapped in explanatory prose.
    salvaged = []
    for match in re.finditer(r"\{[^{}]*\}", stripped, flags=re.DOTALL):
        try:
            salvaged.append(json.loads(match.group(0)))
        except json.JSONDecodeError:
            continue
    return keep_known_columns(salvaged), "salvage"


