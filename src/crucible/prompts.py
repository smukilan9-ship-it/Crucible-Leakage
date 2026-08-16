"""Prompt assembly.

The two criterion clauses below are taken verbatim from the research benchmark
this tool is built on. They are the product of measured ablations, so their
wording should not be edited casually.

Sample rows are collected for the report and deliberately never placed in a
prompt. The ablation that tested whether they help was under-powered — two
models slightly worse with rows, one substantially better, all single-shuffle
against a larger shuffle-order spread — so it establishes nothing in either
direction and no claim is made here. Rows are withheld because they cost tokens
and invite the model to reason from correlation, which is exactly the inference
this tool exists to avoid.
"""

DERIVATION_CRITERION = """\
There are two distinct reasons a column can be UNAVAILABLE, and both count:
(a) TIMING - the value does not exist, or is not yet final, at the prediction
point. (b) DERIVATION - the value records WHY the target's outcome was
assigned, or was itself an input used to determine the target. This holds
EVEN IF the value was recorded BEFORE the prediction point.
A column can satisfy (b) while being chronologically earlier than the target.
Judging only by (a) will mark such a column AVAILABLE, which is wrong.
Being merely predictive is not sufficient for either."""

# The same criterion stated without any temporal vocabulary inside it, and with
# a reconstruction test in place of a timing test. Verbatim from the research.
#
# The two wordings are not ranked. They fail in mirror image: the temporal
# wording lets a literal reader excuse a column recorded at the same moment as
# the target, while the information wording has no brake and will flag every
# column of a table whose target is a rule over its own sensors. Which one wins
# is a property of the model, not of the task, so the choice is made per model
# from measured results rather than argued for in general.
DERIVATION_CRITERION_INFORMATION = """\
There are two distinct reasons a column can be UNAVAILABLE, and both count:

  (a) TIMING - the value does not exist, or is not yet final, at the prediction
      point.
  (b) DERIVATION - the target's value is a function of this column's value: the
      column was an input to the process that assigned the target, or the
      target was computed, defined or decided from it.

Criterion (b) is about INFORMATION, not about time. Do not test (b) by asking
when the value was measured. Test it by asking: could the target be
reconstructed, wholly or in part, from this column? If yes, the column is
UNAVAILABLE whatever its timing - earlier, later, or simultaneous.

Being merely predictive is not sufficient for either: a column can correlate
strongly with the target and still be AVAILABLE."""

CRITERION_WORDINGS = {
    "temporal": DERIVATION_CRITERION,
    "information": DERIVATION_CRITERION_INFORMATION,
}

# The surrogate criterion that used to sit here has been withdrawn. It asserted
# that a prior estimate of the target is inadmissible even though its value
# exists at the prediction point. No source could be found that licenses the
# claim, and the documentation for the columns it was written from says those
# values are recorded at the prediction point. The concern it encoded is real
# but it is a question about what a model is for, not about whether a value
# could honestly be obtained, so it is now surfaced to a human by the contested
# gate in `contested.py` rather than asserted in a prompt.

RESPONSE_CONTRACT = """\
Answer for EVERY column listed. Respond with ONLY a JSON array, one object per
column, no prose before or after:
[{"column": "<name>", "verdict": "LEAK" | "OK" | "ABSTAIN",
  "mechanism": "REASON" | "CONSEQUENCE" | "TIMING" | null,
  "reason": "<one line>"}]
ABSTAIN is permitted when the task is not decidable from the information given
(say why). A partial answer is a failed answer."""


CITATION_CONTRACT = """\
Each column below is followed by its description from the dataset's own
documentation. Base your verdict on that description, and in "reason" quote the
part of it that decides the matter, so the decision can be checked against the
source. If a description contradicts what the column name suggests, the
description wins."""


def build_screen_prompt(columns: list[str], target: str, prediction_point: str | None,
                        descriptions: dict | None = None, criterion: str = "temporal") -> str:
    """Build the main audit prompt. All columns are judged together in one
    prompt, against the named target, because whether a column leaks is a
    relational question: a residual class like "Other_Faults" is only
    recognizable when its sibling columns are visible in the same list.

    The prediction point is only ever sent together with the derivation
    criterion. Sent on its own, it makes things worse: the model checks
    whether each value exists by that moment, correctly concludes that a
    prior estimate of the target does, and waves it through.

    When a data dictionary is supplied, each column carries its documented
    description and the model is told to quote the deciding phrase. Note that
    this is documentation, not data: the rule against sending sample rows still
    holds, because what a column means is written down, while what it contains
    is not evidence of where it came from.
    """
    prediction_point_block = (
        f"The prediction point (when, in deployment, the prediction is made):\n{prediction_point}\n\n"
        if prediction_point
        else ""
    )
    descriptions = descriptions or {}
    column_list = "\n".join(
        f"- {column}: {descriptions[column]}" if descriptions.get(column) else f"- {column}"
        for column in columns
    )
    citation_block = f"\n{CITATION_CONTRACT}\n" if descriptions else ""
    derivation = CRITERION_WORDINGS.get(criterion, DERIVATION_CRITERION)
    return f"""You are auditing a tabular ML dataset for feature-level target leakage:
columns whose value encodes the outcome they would be used to predict.

The target column is: {target}
{prediction_point_block}All columns to judge (against that target, alongside each other):
{column_list}
{citation_block}
{derivation}

{RESPONSE_CONTRACT}"""


def build_contested_probe(columns: list[str], target: str, prediction_point: str,
                          descriptions: dict | None = None) -> str:
    """Build the contested-column probe: a narrow second pass over the columns
    the main screen flagged.

    One question only, and it is a question about the documentation rather
    than about the world: does the source say this value is already fixed at
    or before the prediction point? A column where the answer is yes is not
    thereby innocent, and it is not thereby guilty either. It is contested,
    and it goes to a person.

    This is the stage that stops the tool from asserting a category it cannot
    license. A prior estimate of the target is the standing example: it is
    genuinely available at the prediction point, using it is genuinely a bad
    idea, and those two facts do not settle each other.
    """
    descriptions = descriptions or {}
    column_list = "\n".join(
        f"- {column}: {descriptions[column]}" if descriptions.get(column) else f"- {column}"
        for column in columns
    )
    return f"""Each column below has been flagged as a possible leak in a tabular dataset
whose target is "{target}".

The prediction point is:
{prediction_point}

One question about each, and only this question: does the column's own
description state, or clearly entail, that its value is already fixed and
recorded at or before that prediction point?

Answer FIXED only when the documentation supports it. Answer NOT_FIXED when
the value is produced later, or when the documentation does not say. Do not
judge whether using the column is a good idea, and do not consider how
predictive it is; that is not what is being asked here.

Columns:
{column_list}

Respond with ONLY a JSON array:
[{{"column": "<name>", "verdict": "FIXED" | "NOT_FIXED", "reason": "<one line>"}}]
Answer for every column."""
