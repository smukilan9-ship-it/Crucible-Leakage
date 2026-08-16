"""Triage: cross-tabulate the two screens into four evidence buckets.

The two screens are blind in opposite directions, so where they disagree is
informative, and buckets A and B are the review queue.

That queue is short. In the research corpus the semantic screen flagged between
10% and 23% of all columns depending on which model read them, and for this
tool's default model those flags were 14% of the columns and contained 94% of
the documented leaks.
"""

BUCKETS = {
    "A": "near-certain (both screens flag)",
    "B": "semantic-only: the leaks statistics cannot see",
    "C": "statistical-only: usually just a strong legitimate predictor",
    "D": "neither screen flagged it",
}


def fuse(semantic: dict, statistical: dict, columns: list[str]) -> dict:
    """Return {column: bucket_letter}.

    An abstention from the semantic screen routes to bucket B for human
    review. Treating "the model declined to judge" as "clean" would hide
    exactly the columns most worth a person's attention.
    """
    buckets = {}
    for column in columns:
        semantic_verdict = semantic.get(column, {}).get("verdict", "OK")
        statistically_flagged = statistical.get(column, {}).get("flagged", False)
        if semantic_verdict in ("LEAK", "ABSTAIN"):
            buckets[column] = "A" if (semantic_verdict == "LEAK" and statistically_flagged) else "B"
        else:
            buckets[column] = "C" if statistically_flagged else "D"
    return buckets
