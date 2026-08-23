"""Every measured figure this tool quotes, in one place, with its source.

Nothing in Crucible was measured on Crucible. Each number below comes from the
benchmark the design is drawn from, and each carries the section of that paper
it can be checked against. Modules import from here rather than embedding
figures in prose, so that when the benchmark is re-run the tool changes in one
file instead of eight, and so that a reader can grep for a number and find out
where it came from.

Two rules for anything added here.

**A figure is quoted under the condition the tool actually uses.** A model's
best cell in some other condition is not this tool's number, because nobody
could reproduce it by running this tool.

**A figure that has been withdrawn is deleted, not softened.** The benchmark
withdrew a fifth leakage mechanism and withdrew order-averaging as a general
remedy. Neither appears here.
"""

import dataclasses

PAPER = ("Detecting Feature-Level Target Leakage with Language Models: "
         "A Source-Grounded Benchmark")


@dataclasses.dataclass(frozen=True)
class Cell:
    """One measured (model, condition) cell."""
    precision: float
    recall: float
    f1: float
    datasets: int
    shuffles: int
    condition: str
    source: str

    def as_dict(self) -> dict:
        return {"precision": self.precision, "recall": self.recall, "f1": self.f1,
                "datasets": self.datasets, "shuffles": self.shuffles,
                "condition": self.condition}


# ── the corpus the figures below are measured on ─────────────────────────────

CORPUS = {"datasets": 15, "columns": 604, "positives": 68, "source": "PAPER abstract"}
MAIN_CORPUS = {"datasets": 12, "columns": 306, "positives": 40, "source": "N §1"}

# The documentation sweep. Three rates, because they answer different questions;
# the corrected rate is the one this tool quotes.
SWEEP = {
    "records": 7109,
    "frozen": 6, "frozen_rate": 0.00084,
    "corrected": 7, "corrected_rate": 0.00098,
    "combined": 8, "combined_rate": 0.00113,
    "source": "PAPER §4.3, N §4",
}


# ── baselines, every threshold swept on the answer key ───────────────────────
#
# Each is therefore an upper bound no deployment could reach, which is the
# point: the semantic screen is compared against the best a statistic could
# possibly do, not against a statistic handicapped for the comparison.

BASELINES = {
    "correlation": Cell(0.697, 0.575, 0.630, 12, 0, "B3", "PAPER §5.4"),
    "univariate_auc": Cell(0.733, 0.275, 0.400, 12, 0, "B2", "PAPER §5.4"),
    "missingness": Cell(0.667, 0.150, 0.245, 12, 0, "B4", "PAPER §5.4"),
    "name_regex": Cell(0.667, 0.100, 0.174, 12, 0, "B1", "PAPER §5.4"),
}

# The cutoff the benchmarked correlation baseline used, swept on the answers.
# Not the shipped default; see stats.py for why.
BASELINE_THRESHOLD = 0.3202

BEST_ON_LADDER = 0.918          # gpt-5.6-sol at C6, the best figure anywhere
BEST_NAMES_ONLY = 0.905         # claude-opus-5 at C1, names and a target alone
BASELINE_F1 = BASELINES["correlation"].f1


# ── the models this tool offers, in the condition it sends them ──────────────

MODEL_CELLS = {
    "gemini-3.7-flash": Cell(0.867, 0.938, 0.901, 12, 5, "C6", "PAPER §6.1"),
    "gemini-3.5-flash": Cell(0.805, 0.943, 0.868, 12, 5, "C6", "PAPER §6.1"),
    "moonshotai/Kimi-K3": Cell(0.833, 0.968, 0.896, 11, 1, "C9", "PAPER §7.3, N §6"),
    "gpt-5.6-sol": Cell(0.867, 0.975, 0.918, 12, 1, "C6", "PAPER §6.1"),
    "claude-opus-5": Cell(0.886, 0.975, 0.929, 12, 1, "C9", "PAPER §7.3"),
}

# Where each model sits among the ten tested, ranked by its best measured cell.
MODEL_RANK = {
    "claude-opus-5": 1, "gpt-5.6-sol": 2, "gemini-3.7-flash": 3,
    "moonshotai/Kimi-K3": 4, "gemini-3.5-flash": 6,
}
MODELS_TESTED = 10
LABORATORIES = 8


# ── why the derivation clause is in the prompt, and what it is worth ─────────
#
# Mean recall by subtype before any intervention, then after one clause naming
# derivation as a second reason. The gap is concentrated in the third kind.

SUBTYPE_RECALL = {
    "timing": (0.97, 0.97), "consequence": (0.89, 0.92), "reason": (0.62, 0.81),
    "source": "PAPER §6.2",
}

# What that clause is worth per model, resampling datasets rather than columns.
# Read this before claiming the clause helps: on the strongest models it does
# not measurably, and on one model it hurts.
CLAUSE_EFFECT = {
    "gemini-3.7-flash": {"delta": +0.067, "ci": (-0.004, +0.234),
                         "fixed": 19, "broken": 1, "p": 0.001},
    "gemini-3.5-flash": {"delta": +0.035, "ci": (-0.027, +0.181),
                         "fixed": 18, "broken": 8, "p": 0.076},
    "claude-opus-5": {"delta": +0.004, "ci": (-0.018, +0.050),
                      "fixed": 2, "broken": 2, "p": 1.000},
    "gpt-5.6-sol": {"delta": +0.053, "ci": (0.000, +0.206),
                    "fixed": 4, "broken": 0, "p": 0.125},
    "moonshotai/Kimi-K3": {"delta": 0.000, "ci": (0.000, 0.000),
                           "fixed": 0, "broken": 0, "p": 1.000},
    "source": "PAPER §6.5, N §19",
}

CLAUSE_SUMMARY = (
    "The derivation criterion repairs weak detectors and does nothing "
    "measurable for the strongest ones: every confidence interval that "
    "excludes zero belongs to a model scoring under 0.66 without it."
)


# ── shuffle order ────────────────────────────────────────────────────────────
#
# A property of the model, not of the task. One model moved this far between
# two runs of a byte-identical prompt; another returned the same answer under
# every order, which is why averaging was withdrawn as a general remedy.

ORDER_SPREAD_MAX = 0.380        # gemini-3.5, held-out stratum, PAPER §7.5
ORDER_SPREAD_MIN = 0.000        # gpt-5.6-sol, three shuffles, one answer

# The worst spread measured for each model *at the condition this tool sends
# it*, across both strata. This is what decides how many orders are worth
# paying for, and it is the reason "Gemini" is not one answer: 3.7 Flash is the
# steadiest model in the study at 0.019, while 3.5 Flash produced the widest
# spread anywhere in the paper, 0.380 on the held-out set. Reading them as one
# family would take the safe number from the unsafe model.
ORDER_SPREAD = {
    "gemini-3.7-flash": {"worst": 0.019, "seeds": 5, "source": "N §13 Stratum A, C6"},
    "gemini-3.5-flash": {"worst": 0.380, "seeds": 3, "source": "N §13 Stratum B, C6"},
    "gpt-5.6-sol": {"worst": 0.159, "seeds": 3, "source": "N §13 Stratum B, C6"},
    "claude-opus-5": {"worst": 0.149, "seeds": 3, "source": "N §13 Stratum B, C9"},
}

# Below this, a second and third order buy nothing worth three times the cost.
# It sits well under the smallest difference this tool reports on.
STABLE_ENOUGH_FOR_ONE_ORDER = 0.05
DEFAULT_SHUFFLES = 3

# ── the variance the order study does not cover ──────────────────────────────
#
# ORDER_SPREAD answers "how far does the answer move when the columns move?".
# It does not answer "does the same prompt give the same answer twice?", and
# those come apart. Measured 2026-08-23 on DROPOUT, 36 columns, the identical
# prompt sent six times with one column order and temperature 0.0:
#
#     LEAK count per call:  11, 12, 10, 11, 10, 11
#     Curricular units 1st sem (credited):  LEAK 4 / OK 2
#     Curricular units 1st sem (enrolled):  OK 5 / LEAK 1
#
# Two of thirty-six columns disagreed with themselves, and both are documented
# positives. The model is genuinely the steadiest in the study under reordering
# at 0.019, and that is a separate fact from this one.
#
# The single-order exemption rested on the claim that "a vote would cost three
# calls to reproduce one answer". This measurement is what falsifies the
# premise: at k = 1 there is no answer to reproduce, only a draw. So the
# exemption now requires a model to have been measured on both axes, and no
# model has yet cleared the second one.
REPEAT_STABILITY = {
    "gemini-3.7-flash": {"unstable_columns": 2, "of": 36, "calls": 6,
                         "dataset": "DROPOUT",
                         "source": "measured 2026-08-23, Vertex, C6"},
}

# A model may skip the vote only if repeated identical calls agreed. None do.
STABLE_ENOUGH_FOR_ONE_CALL = 0


# ── what a reviewer is actually asked to do ──────────────────────────────────

TRIAGE = {
    "default_model": {"burden": 0.140, "recall": 0.936},   # gemini-3.7 at C6
    "best_model": {"burden": 0.157, "recall": 1.000},      # claude-opus-5 at C6
    "burden_range": (0.103, 0.229),
    "source": "N §14",
}


# ── what the leakage was worth downstream ────────────────────────────────────

DOWNSTREAM = {
    "inflation_f1_mean": 0.147, "inflation_f1_max": 0.306,   # random forest
    "inflation_auc_mean": 0.129,
    "residual_model_arms": 0.024,      # how close model cleaning lands to the ceiling
    "residual_correlation": 0.048,     # the baseline lands twice as far away
    "source": "PAPER §7.4, N §8",
}

# The cost of over-flagging, and the reason this tool refuses to delete anything
# on its own. The smallest table in the corpus, where every model drops one
# column too many.
OVERFLAG = {
    "dataset": "ECHO", "rows": 131,
    "documented_cleaning_f1": 0.677, "model_cleaning_f1": 0.407,
    "source": "PAPER §7.4",
}

# The two examples the interface uses. Both are real rows of the benchmark.
TITANIC = {
    "leak_correlation": 0.014,       # `body`, a consequence of the outcome
    "legitimate_correlation": 0.529,  # `sex`, entirely admissible
    "baseline_arm_f1": 0.658,         # what a correlation threshold scores here
    "documented_cleaning_f1": 0.722,  # and the honest figure it falls below
    "source": "PAPER §7.4, N §18",
}


def sources() -> dict:
    """Every figure group and the section it is checked against. Used by the
    report manifest so a stored audit says where its constants came from."""
    return {
        "paper": PAPER,
        "corpus": CORPUS["source"],
        "sweep": SWEEP["source"],
        "baselines": BASELINES["correlation"].source,
        "models": MODEL_CELLS["gemini-3.7-flash"].source,
        "subtype_recall": SUBTYPE_RECALL["source"],
        "clause_effect": CLAUSE_EFFECT["source"],
        "triage": TRIAGE["source"],
        "downstream": DOWNSTREAM["source"],
    }
