"""The models Crucible offers, and which wording of the criterion each gets.

Two things are recorded here and both come from measurement rather than taste.

**Which models are offered.** Reasoning quality is not cosmetic for this task.
Across ten models from eight laboratories, baseline recall runs 97% on columns
recorded after the prediction point, 89% on columns that exist because the
outcome happened, and 62% on columns that record *why* a label was assigned.
The whole spread between a good model and a poor one sits in that third
category, so the note on each entry says what is actually being traded away.

**Which criterion wording each model gets.** There are two statements of the
same derivation criterion and neither is uniformly better; they fail in mirror
image. Stated in temporal vocabulary, a literal reader excuses a column
recorded at the same moment as the target — one model un-flagged all six of a
dataset's sibling fault columns with the stated reason "measured concurrently".
Stated as a reconstruction test it has no brake — on a synthetic dataset whose
target is a threshold rule over its own sensors, a frontier model flagged all
ten columns. The measured F1 on matched cells decides each row below, and where
a model has not been measured it gets the temporal wording, which is the more
conservative of the two.
"""

from . import evidence

TEMPORAL = "temporal"
INFORMATION = "information"
DEFAULT_CRITERION = TEMPORAL

# Gemini is the default provider. Two reasons, neither of them a preference.
#
# Speed. A reasoning model spends output tokens thinking before it writes
# anything, and a provider that bills such a model at four concurrency units
# against a four-unit plan serializes every call. Together those made a
# thirteen-column audit take a quarter of an hour. A flash model has neither
# property.
#
# Measured quality. On the benchmark's twelve datasets at five shuffles each,
# which are the most-measured cells in the table, Gemini 3.7 Flash reaches
# precision 0.867, recall 0.938, F1 0.901: third of the ten models tested, and
# within 0.017 of the best figure any of them reached. Fast is not the trade.
#
# Every `measured` block reports the model under the criterion wording this
# tool actually sends it, on matched cells, and not its best cell in any
# condition. A figure from a wording the tool does not use is a number nobody
# could reproduce by running the tool.
#
# Model identifiers are what the provider serves and can move; override with
# CRUCIBLE_RECALL_MODEL if one is retired.
CATALOGUE = [
    {
        "id": "gemini-3.7-flash",
        "provider": "gemini",
        "name": "Gemini 3.7 Flash",
        "size": "Google",
        "criterion": TEMPORAL,          # measured C1 0.834 -> C6 0.901
        "note": "The default, and the only entry that needs no key of your own. "
                "F1 0.901 on the benchmark, third of ten models tested, and the "
                "steadiest under reordering at a spread of 0.019, so it is asked "
                "once rather than three times. Fast enough to audit while you wait.",
        "measured": evidence.MODEL_CELLS["gemini-3.7-flash"].as_dict(),
        "recommended": True,
    },
    {
        "id": "gemini-3.5-flash",
        "provider": "gemini",
        "name": "Gemini 3.5 Flash",
        "size": "Google",
        "criterion": TEMPORAL,          # measured C1 0.833 -> C6 0.868
        "note": "A little weaker and a little cheaper: F1 0.868, sixth of ten. "
                "Also the substitute if Gemini stops serving 3.7 Flash, so an "
                "audit finishes rather than failing on a retired identifier.",
        "measured": evidence.MODEL_CELLS["gemini-3.5-flash"].as_dict(),
        "recommended": False,
    },
    {
        "id": "claude-opus-5",
        "provider": "anthropic",
        "name": "Claude Opus 5",
        "size": "Anthropic",
        "criterion": INFORMATION,       # measured C6 0.909 -> C9 0.929
        "note": "The strongest result in the benchmark: F1 0.929, and recall "
                "1.000 at every condition on the held-out set. It is also the "
                "clearest case for reading the derivation clause skeptically, "
                "since the clause moves it by 0.004 with two columns going each "
                "way. Bring an Anthropic key.",
        "measured": evidence.MODEL_CELLS["claude-opus-5"].as_dict(),
        "needs_key": "anthropic",
        "recommended": False,
    },
    {
        "id": "gpt-5.6-sol",
        "provider": "openai",
        "name": "GPT-5.6 Sol",
        "size": "OpenAI",
        "criterion": TEMPORAL,          # measured C1 0.864 -> C6 0.918
        "note": "The best figure anywhere on the benchmark's condition ladder, "
                "F1 0.918. It is famously steady under reordering, but that was "
                "measured without the derivation clause; with the clause this "
                "tool sends, its spread is 0.159, so it still gets three orders. "
                "Bring an OpenAI key.",
        "measured": evidence.MODEL_CELLS["gpt-5.6-sol"].as_dict(),
        "needs_key": "openai",
        "recommended": False,
    },
    {
        "id": "moonshotai/Kimi-K3",
        "provider": "featherless",
        "name": "Kimi K3",
        "size": "Moonshot AI, via Featherless",
        "criterion": INFORMATION,       # measured C6 0.882 -> C9 0.896
        "note": "A reasoning model, and the only one tested that needed no help "
                "with derivation: 93% recall on label-deciding columns before any "
                "clause was added, joint-highest of the ten. It thinks before it "
                "answers, so expect minutes per audit rather than seconds. Bring "
                "a Featherless key.",
        "measured": evidence.MODEL_CELLS["moonshotai/Kimi-K3"].as_dict(),
        "needs_key": "featherless",
        "recommended": False,
    },
]

BY_ID = {entry["id"]: entry for entry in CATALOGUE}
DEFAULT_MODEL = next(entry["id"] for entry in CATALOGUE if entry["recommended"])


def entry(model_id: str) -> dict:
    """Look up a model, or fail with a message naming the alternatives."""
    try:
        return BY_ID[model_id]
    except KeyError:
        raise KeyError(
            f"unknown model {model_id!r}; available: {', '.join(sorted(BY_ID))}")


def criterion_for(model_id: str) -> str:
    return BY_ID.get(model_id, {}).get("criterion", DEFAULT_CRITERION)


def provider_for(model_id: str) -> str:
    return BY_ID.get(model_id, {}).get("provider", "gemini")


def shuffles_for(model_id: str) -> int:
    """How many column orders this model is worth asking.

    Order sensitivity is a property of the model, so the count is too. A model
    measured steady under reordering gains nothing from a vote and pays three
    times over for it; a model that swings needs every order it can get.

    Only `gemini-3.7-flash` currently clears the bar, at a measured spread of
    0.019. `gemini-3.5-flash` does not, despite being the same family and the
    same vendor: it produced the widest spread in the study, 0.380 on the
    held-out set. Treating them as one family would hand the safe number to the
    unsafe model, which is why this is keyed on the identifier and not the
    provider.
    """
    measured = evidence.ORDER_SPREAD.get(model_id)
    if measured and measured["worst"] < evidence.STABLE_ENOUGH_FOR_ONE_ORDER:
        return 1
    return evidence.DEFAULT_SHUFFLES


def shuffle_rationale(model_id: str) -> str:
    """Why this model gets the number of orders it gets, in one line."""
    measured = evidence.ORDER_SPREAD.get(model_id)
    if not measured:
        return (f"{evidence.DEFAULT_SHUFFLES} orders: this model's order sensitivity "
                f"has not been measured, so it gets the cautious default.")
    if measured["worst"] < evidence.STABLE_ENOUGH_FOR_ONE_ORDER:
        return (f"1 order: measured spread {measured['worst']:.3f} across "
                f"{measured['seeds']} seeds, steady enough that a vote would cost "
                f"three calls to reproduce one answer.")
    return (f"{evidence.DEFAULT_SHUFFLES} orders: measured spread "
            f"{measured['worst']:.3f} across {measured['seeds']} seeds, wide enough "
            f"that a single pass is not a stable answer.")


def public_catalogue() -> list[dict]:
    """The catalogue as an interface may show it. Contains no credentials, and
    no field that varies with who is asking."""
    return [dict(item) for item in CATALOGUE]
