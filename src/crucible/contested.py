"""Stage 5 — the contested-column gate.

This replaces a stage that was withdrawn, and the reason it was withdrawn is
the reason this one is shaped the way it is.

The earlier design had a fifth mechanism, SURROGATE: a column that is a prior
estimate of the same target, such as a physician's survival estimate or a
first-period grade. It was dropped because no source could be found that
licenses it. The documentation for those columns says the values are recorded
at the prediction point, and the strongest statement anyone had made about them
was that predicting without them is "more difficult but much more useful" — a
claim about difficulty, not about admissibility.

The underlying worry is real: a model fed a physician's survival estimate
predicts the physician rather than the patient. But that is a claim about what
a model is *for*, not about whether a value could honestly have been obtained
at the moment of prediction. Those are different questions, and only the second
one is leakage.

So this stage does not decide. It asks one question of the documentation — does
the source say this value is fixed at or before the prediction point? — and
where the answer is yes it marks the column CONTESTED and hands it to a person.
A contested column is never promoted to a leak automatically, and never quietly
dismissed either.
"""

from . import prompts
from .parsing import parse_verdicts
from .providers import output_budget

CONTESTED = "CONTESTED"


async def contested_gate(provider, model: str, candidates: list[str],
                         target: str, prediction_point: str,
                         descriptions: dict | None = None) -> dict:
    """Ask, for each flagged candidate, whether its value is already fixed at
    the prediction point.

    Returns {column: reason} for the columns that are contested. An empty
    result means every candidate is an ordinary flag.
    """
    if not candidates:
        return {}
    prompt = prompts.build_contested_probe(candidates, target, prediction_point, descriptions)
    text = await provider.chat(model, prompt, max_tokens=output_budget(len(candidates)))
    answers, _ = parse_verdicts(text, candidates)
    return {
        column: answer.get("reason", "")
        for column, answer in answers.items()
        if answer["verdict"] == "FIXED"
    }
