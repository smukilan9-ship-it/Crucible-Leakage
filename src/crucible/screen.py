"""The semantic screen: ask a language model about every column, several
times in different column orders, and take a majority vote.

**What the shuffling is for, stated at the strength the evidence supports.**
Order sensitivity is a property of the model, not of the task. Changing nothing
but the sequence in which the columns are listed moved one model's score by
0.380 between two runs of a byte-identical prompt, while another model returned
the same answer under three shuffles, spread 0.000.

So voting across orders is a hedge, not a remedy. The research this tool comes
from proposed it as a general improvement and then withdrew that claim: where a
model is stable there is nothing to average, and the vote costs three calls to
reproduce one answer. Where a model is unstable it is the difference between an
answer and a coin flip.

Three orders is the default because a caller usually does not know which kind of
model they have. `--shuffles 1` is the correct setting for a model measured
stable on tables like theirs, and it makes the audit three times cheaper.
"""

import asyncio
import math
import random
import re

from . import prompts
from .parsing import parse_verdicts
from .providers import ProviderError, output_budget

COVERAGE_FLOOR = 0.9


async def run_semantic_screen(
    provider,
    model: str,
    columns: list[str],
    target: str,
    prediction_point: str | None,
    shuffle_count: int = 3,
    emit=None,
    max_reissues: int = 2,
    descriptions: dict | None = None,
    criterion: str = "temporal",
) -> dict:
    """Run the full screen and return one combined verdict per column:
    {column: {verdict, mechanism, reasons, leak_votes, shuffles_counted}}.

    A reply that answers fewer than 90% of the columns is treated as a failed
    attempt and reissued, never scored as a partial answer. A call that fails
    outright is retried inside the client and never recorded as an answer.

    Returns (combined_verdicts, verdicts_by_shuffle). The per-shuffle detail
    is kept so the interface can show the vote landing, order by order.
    """
    async def one_order(shuffle_index: int):
        """One column order, reissued until it answers enough of the table.

        Returns the verdicts, or None if this order could not be made to work.
        A single order failing is not the run failing: the vote wants a
        majority, not unanimity, and a provider that is briefly overloaded
        should cost a pass rather than the audit.
        """
        shuffled_columns = columns[:]
        random.Random(shuffle_index * 7919 + 17).shuffle(shuffled_columns)
        prompt = prompts.build_screen_prompt(
            shuffled_columns, target, prediction_point, descriptions, criterion)

        for attempt in range(max_reissues + 1):
            try:
                text = await provider.chat(model, prompt,
                                           max_tokens=output_budget(len(columns)))
            except ProviderError as error:
                if emit:
                    await emit("shuffle_failed", {
                        "shuffle": shuffle_index + 1, "of": shuffle_count,
                        "attempt": attempt + 1, "reason": str(error)})
                continue
            verdicts, parser_used = parse_verdicts(text, columns)
            coverage = len(verdicts) / len(columns)
            if emit:
                await emit("shuffle", {
                    "model": model, "shuffle": shuffle_index + 1, "of": shuffle_count,
                    "coverage": round(coverage, 3), "parser": parser_used,
                    "attempt": attempt + 1,
                    "leaks_found": sum(1 for v in verdicts.values() if v["verdict"] == "LEAK"),
                    "abstained": sum(1 for v in verdicts.values() if v["verdict"] == "ABSTAIN"),
                    # the interface replays this order and these verdicts, so a
                    # viewer sees the same thing the model was asked
                    "order": shuffled_columns,
                    "verdicts": {column: answer["verdict"] for column, answer in verdicts.items()},
                })
            # The join gate. If nothing the model returned names a real column,
            # the cell is refused rather than scored. A reply whose verdict
            # keys miss the schema entirely scores as "every column is clean",
            # which is indistinguishable from a model that looked carefully and
            # found nothing. One model once returned a single column literally
            # named `Pstatus,paid,etc...`; scored naively that is a table full
            # of false negatives.
            if verdicts and coverage >= COVERAGE_FLOOR:
                return verdicts
        return None

    # The orders are independent, so they are asked concurrently. Run in
    # sequence this was three round trips deep, and on a wide table each round
    # trip is the model writing a verdict for every column. The seeds and the
    # vote are untouched: `gather` preserves order, so the same orders produce
    # the same answer as before, roughly three times sooner.
    answers = await asyncio.gather(*(one_order(index) for index in range(shuffle_count)))
    verdicts_by_shuffle = [answer for answer in answers if answer]

    if not verdicts_by_shuffle:
        raise ProviderError(
            f"{model}: no column order produced a usable answer. Either the "
            f"provider refused every call, or it answered under "
            f"{COVERAGE_FLOOR:.0%} of the table each time.")
    if emit and len(verdicts_by_shuffle) < shuffle_count:
        await emit("orders_counted", {
            "counted": len(verdicts_by_shuffle), "asked": shuffle_count})

    return majority_vote(verdicts_by_shuffle, columns), verdicts_by_shuffle


def majority_vote(verdicts_by_shuffle: list[dict], columns: list[str]) -> dict:
    """Combine per-shuffle verdicts: a column is flagged as a leak when a
    majority of the shuffles flagged it, and likewise for abstentions."""
    votes_needed = math.ceil(len(verdicts_by_shuffle) / 2)
    combined = {}
    for column in columns:
        answers = [shuffle.get(column) for shuffle in verdicts_by_shuffle if shuffle.get(column)]
        leak_votes = sum(1 for answer in answers if answer["verdict"] == "LEAK")
        abstain_votes = sum(1 for answer in answers if answer["verdict"] == "ABSTAIN")
        if leak_votes >= votes_needed:
            verdict = "LEAK"
        elif abstain_votes >= votes_needed:
            verdict = "ABSTAIN"
        else:
            verdict = "OK"
        mechanisms = [
            answer.get("mechanism") for answer in answers
            if answer["verdict"] == "LEAK" and answer.get("mechanism")
        ]
        # A column the passes disagreed about. Reporting only the winning side
        # would throw away the more useful half: the model argued both ways and
        # a reader is entitled to see both arguments rather than the score of a
        # vote they did not watch. Measured on DROPOUT, the same prompt sent six
        # times moved on two of thirty-six columns, and both were real leaks, so
        # this is not a rare shape.
        split = len(answers) > 1 and 0 < leak_votes < len(answers)
        combined[column] = {
            "verdict": verdict,
            "mechanism": max(set(mechanisms), key=mechanisms.count) if mechanisms else None,
            # Every pass answers for every column, so a column usually comes
            # back with three reasons that say the same thing in different
            # words. Near-duplicates are collapsed, because a reader wants the
            # distinct grounds for the verdict, not the model's phrasing three
            # times over.
            "reasons": _distinct_reasons(answer.get("reason") for answer in answers),
            "leak_votes": leak_votes,
            "shuffles_counted": len(answers),
        }
        if split:
            combined[column].update({
                "split": True,
                "reasons_for": _distinct_reasons(
                    answer.get("reason") for answer in answers
                    if answer["verdict"] == "LEAK"),
                "reasons_against": _distinct_reasons(
                    answer.get("reason") for answer in answers
                    if answer["verdict"] != "LEAK"),
            })
    return combined


def _distinct_reasons(reasons) -> list[str]:
    """Keep the reasons that say something new, in the order they arrived."""
    kept: list[str] = []
    for reason in reasons:
        reason = (reason or "").strip()
        if not reason:
            continue
        fingerprint = _fingerprint(reason)
        if any(_overlaps(fingerprint, _fingerprint(existing)) for existing in kept):
            continue
        kept.append(reason)
    return kept


def _fingerprint(text: str) -> set:
    return {word for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) > 3}


def _overlaps(left: set, right: set, threshold: float = 0.7) -> bool:
    if not left or not right:
        return False
    shared = len(left & right)
    return shared / min(len(left), len(right)) >= threshold
