"""Intake: load the table, validate the target, and refuse jobs that cannot
be judged honestly."""

import os
import re

import pandas as pd

ANONYMOUS_NAME_PATTERN = re.compile(r"^(attr|var|col|feature|[avxf])_?\d+$", re.IGNORECASE)


class IntakeError(Exception):
    pass


# Delimiters worth trying, in the order they are worth trying. Published
# scientific tables are not reliably comma separated, and a table read with the
# wrong delimiter collapses into a single column of joined text that every
# stage downstream will then process without complaint.
DELIMITERS = (",", "\t", ";", "|")


def _comment_preamble(path: str, marker: str = "#") -> int:
    """How many leading lines are a comment block rather than data.

    Written for real archive exports. The NASA Exoplanet Archive prefixes its
    tables with a block of `#` lines that names every column; pandas reads the
    first of those as a one-field header and then raises on the first real row,
    so the whole file is unusable for the sake of six lines.

    Only a *leading* run counts. A `#` further down the file is data.
    """
    count = 0
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.strip() and not line.lstrip().startswith(marker):
                    break
                count += 1
    except OSError as error:
        raise IntakeError(f"could not read {path}: {error}") from error
    return count


def load_table(path: str) -> pd.DataFrame:
    """Read a table, working out the delimiter and skipping any comment header.

    Every candidate parse is attempted and the widest result wins. Width is the
    right criterion because both failure modes here, an unrecognized delimiter
    and an unskipped preamble, show up as too few columns rather than too many.
    """
    if path.endswith(".parquet"):
        try:
            return pd.read_parquet(path)
        except Exception as error:
            raise IntakeError(f"could not read the Parquet file: {error}") from error

    skip = _comment_preamble(path)
    attempts = sorted({skip, 0})
    best, last_error = None, None
    for delimiter in DELIMITERS:
        for skiprows in attempts:
            try:
                frame = pd.read_csv(path, sep=delimiter, skiprows=skiprows,
                                    engine="python")
            except Exception as error:          # noqa: BLE001 - surfaced below
                last_error = error
                continue
            if frame.empty or not len(frame.columns):
                continue
            if best is None or frame.shape[1] > best.shape[1]:
                best = frame

    name = os.path.basename(path)
    if best is None:
        raise IntakeError(
            f"could not parse {name} as a table"
            + (f": {last_error}" if last_error else ""))

    if best.shape[1] < 2:
        raise IntakeError(
            f"{name} parsed as a single column named "
            f"{str(best.columns[0])[:60]!r}. That almost always means the "
            f"delimiter is not one of "
            + ", ".join(repr(d) for d in DELIMITERS)
            + ", or the file opens with a header block this reader did not "
              "recognize. An audit needs a target and at least one feature.")
    return best


def check_target(table: pd.DataFrame, target: str) -> None:
    if target not in table.columns:
        raise IntakeError(f"target column {target!r} not in table")


def names_are_anonymized(columns: list[str], threshold: float = 0.6) -> bool:
    """Return True when most column names look like placeholders (A1, var_3).

    The audit works by reading what column names say about where each value
    comes from. A name like "A17" says nothing, so a table full of them
    cannot be judged. Refusing the job outright is more honest than
    returning guesses dressed up as findings.
    """
    anonymous_count = sum(
        1 for column in columns
        if ANONYMOUS_NAME_PATTERN.match(column.strip()) or len(column.strip()) <= 3
    )
    return anonymous_count / max(len(columns), 1) > threshold


DICTIONARY_NAME_KEYS = ("column", "field", "name", "variable", "attribute")
DICTIONARY_TEXT_KEYS = ("description", "definition", "meaning", "notes", "detail", "comment")


def load_dictionary(path: str, columns: list[str], target: str | None = None) -> dict:
    """Read an optional data dictionary: one row per column, with that
    column's documented meaning.

    This is the single highest-value thing a user can provide. Without it the
    audit reasons from column names alone, which is what the research measured
    and what works well enough. With it, every verdict is grounded in the
    dataset's own documentation, and each dropped column comes with a quotable
    sentence explaining why, which is what makes the decision citable in a
    paper rather than merely plausible.

    The column holding the name and the column holding the description are
    found by looking for any of several common headings, because no two
    published dictionaries agree on what to call them.
    """
    frame = load_table(path)
    if frame.empty:
        raise IntakeError("the data dictionary is empty")

    name_column = _match_heading(frame.columns, DICTIONARY_NAME_KEYS) or frame.columns[0]
    text_column = _match_heading(frame.columns, DICTIONARY_TEXT_KEYS)
    if text_column is None:
        remaining = [column for column in frame.columns if column != name_column]
        if not remaining:
            raise IntakeError(
                "the data dictionary needs two columns: the column name and its description"
            )
        text_column = remaining[0]

    known = {str(column).strip().lower(): column for column in columns}
    descriptions, unmatched = {}, []
    for _, row in frame.iterrows():
        name = str(row[name_column]).strip()
        text = str(row[text_column]).strip()
        if not name or not text or text.lower() == "nan":
            continue
        actual = known.get(name.lower())
        if actual is not None:
            descriptions[actual] = text
        elif target is None or name.lower() != target.strip().lower():
            # A dictionary almost always documents the target column too. That
            # is not a mismatch, so it is not reported as one.
            unmatched.append(name)

    return {
        "descriptions": descriptions,
        "matched": sorted(descriptions),
        "missing": sorted(column for column in columns if column not in descriptions),
        "unmatched_rows": unmatched[:20],
        "name_column": str(name_column),
        "description_column": str(text_column),
    }


def _match_heading(headings, keys) -> str | None:
    for heading in headings:
        cleaned = str(heading).strip().lower()
        if any(key in cleaned for key in keys):
            return heading
    return None


def describe_table(table: pd.DataFrame, target: str) -> dict:
    feature_columns = [column for column in table.columns if column != target]
    return {
        "n_rows": int(len(table)),
        "n_cols": int(table.shape[1]),
        "target": target,
        "feature_columns": feature_columns,
        "dtypes": {column: str(dtype) for column, dtype in table.dtypes.items()},
        # Sample rows are collected for the human-facing report only. They are
        # never sent to a model; see the note at the top of prompts.py.
        # Missing values become the empty string rather than staying as NaN,
        # which is not valid JSON and which most real tables contain.
        "sample_rows": table.head(5).astype(str).fillna("").to_dict(orient="records"),
    }
