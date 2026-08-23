"""The audit, end to end, independent of any interface.

This is the function the command line calls and the function the web service
calls, so the two cannot drift apart. It takes a table and returns a report; it
does not print, does not serve, and does not decide anything a human should
decide.

The `on_stage` callback exists only so a caller can show progress. It is never
required, and nothing about the result depends on whether anyone is watching.
"""

import dataclasses

import pandas as pd

from . import contested as contested_gate
from . import fusion, intake, models, screen, stats
from .providers import Provider, output_budget, resolve


@dataclasses.dataclass
class AuditRequest:
    table: pd.DataFrame
    target: str
    prediction_point: str
    model: str = models.DEFAULT_MODEL
    descriptions: dict | None = None
    shuffles: int = 3


async def run_audit(request: AuditRequest, provider: Provider | None = None,
                    on_stage=None) -> dict:
    """Run both screens, fuse them, and put the flagged columns through the
    contested gate. Returns a report; drops nothing."""
    async def announce(stage: str, detail: str):
        if on_stage:
            await on_stage(stage, detail)

    intake.check_target(request.table, request.target)
    columns = [c for c in request.table.columns if c != request.target]

    # N-04. A table of V1…V57 has nothing for a semantic screen to read, and a
    # model asked anyway will produce confident noise. The screen is skipped
    # rather than run, but the statistical screen needs no names at all, so the
    # job continues and the report says exactly what was and was not done.
    # Refusing the whole table would throw away the half that still works.
    if intake.names_are_anonymized(columns):
        await announce("statistical", "column names carry no meaning; statistical screen only")
        statistical = stats.statistical_screen(request.table, request.target)
        return {
            "target": request.target,
            "prediction_point": request.prediction_point,
            "model": request.model,
            "criterion": None,
            "shuffles": 0,
            "grounded": False,
            "semantic_skipped": "INSUFFICIENT_SEMANTICS",
            "columns": columns,
            "semantic": {c: {"verdict": "ABSTAIN", "mechanism": None,
                             "reasons": ["column names carry no readable meaning, "
                                         "so the semantic screen was not run"],
                             "leak_votes": 0, "shuffles_counted": 0} for c in columns},
            "statistical": statistical,
            "buckets": {c: ("C" if statistical[c]["flagged"] else "D") for c in columns},
            "contested": {},
            "per_shuffle": [],
        }

    criterion = models.criterion_for(request.model)
    owns_provider = provider is None
    if owns_provider:
        provider = resolve(models.provider_for(request.model))

    try:
        await announce("semantic", f"reading {len(columns)} columns, {request.shuffles} orders")
        semantic, per_shuffle = await screen.run_semantic_screen(
            provider, request.model, columns, request.target, request.prediction_point,
            shuffle_count=request.shuffles, descriptions=request.descriptions,
            criterion=criterion,
        )

        await announce("statistical", "measuring correlation with the target")
        statistical = stats.statistical_screen(request.table, request.target)

        await announce("triage", "sorting by which screens flagged each column")
        buckets = fusion.fuse(semantic, statistical, columns)

        await announce("contested", "checking flagged columns against the documentation")
        flagged = [c for c, answer in semantic.items() if answer["verdict"] == "LEAK"]
        contested = await contested_gate.contested_gate(
            provider, request.model, flagged, request.target,
            request.prediction_point, request.descriptions,
        )
        for column, reason in contested.items():
            semantic[column]["contested"] = True
            semantic[column]["contested_reason"] = reason
    finally:
        if owns_provider:
            await provider.aclose()

    return {
        "target": request.target,
        "prediction_point": request.prediction_point,
        "model": request.model,
        "criterion": criterion,
        "shuffles": request.shuffles,
        "grounded": bool(request.descriptions),
        "semantic_skipped": None,
        "columns": columns,
        "semantic": semantic,
        "statistical": statistical,
        "buckets": buckets,
        "contested": contested,
        "per_shuffle": [
            {c: v["verdict"] for c, v in verdicts.items()} for verdicts in per_shuffle
        ],
    }


def flagged_columns(report: dict, include_contested: bool = False) -> list[str]:
    """The columns the audit flagged.

    Two kinds are excluded by default, for the same reason. A **contested**
    column is flagged by the screen and documented as fixed at the prediction
    point: two different questions, two different answers. A **split** column
    is one the passes could not agree on among themselves. Admitting either
    silently would be the tool making exactly the call it exists to hand to a
    person, and a majority of three is not a decision, it is a tally.
    """
    return sorted(
        column for column, answer in report["semantic"].items()
        if answer["verdict"] == "LEAK"
        and (include_contested
             or not (answer.get("contested") or answer.get("split")))
    )


__all__ = ["AuditRequest", "run_audit", "flagged_columns", "output_budget"]
