# Crucible — Impact Forge 2026 submission notes

**Track.** Computational Research.

**Compute integration.** The semantic screen runs on Google Gemini 3.7 Flash by
default, on a pooled free tier so the tool works with no account at all. Four
providers are wired: Gemini, OpenAI, Anthropic and Featherless.ai. A user pastes
one key into one field and the vendor is read from the key's own prefix, which
also means a key is never forwarded to a company that did not issue it. Adding a
provider is one dictionary entry, and three of the four share a single
OpenAI-shaped transport.
Gemini is the default on measured grounds: on the benchmark's twelve datasets
at five shuffles each it reaches F1 0.901, third of ten models tested and within
0.017 of the best of them, and it is not a reasoning model, so an audit finishes
in seconds rather than the quarter of an hour a reasoning model serialized at
four concurrency units took.

The inference pipeline is the substance here, not the endpoint: three shuffled
passes with majority voting, a coverage floor, a join gate, a second narrow
pass for contested columns, per-model selection between two measured wordings
of the derivation criterion, and adaptive output budgets with a doubling retry.

---

## What this is

A tool that audits a tabular dataset for **feature-level target leakage** —
columns whose value encodes the outcome they would be used to predict — and
then measures what those columns were worth to the model.

The failure is not exotic. It is the single most common reason a model scores
brilliantly offline and collapses in deployment, and nothing in a standard
workflow catches it, because cross-validation cannot: the leaked column sits on
both sides of every split, so every fold independently confirms the model is
excellent.

## Provenance, stated plainly

The **research** that establishes this problem and validates the design — a
604-column benchmark across 15 datasets, ten models from eight laboratories,
and the repository sweep — **predates the hackathon.** It is the author's own
prior work.

**Every line of code in this repository was written during the hackathon
window.** The research is used only to decide what the tool should do, and each
such decision is marked in the source with the measurement that motivated it.

Judges should read this as: the science is prior, the software is new, and the
software is what is being submitted.

## What the tool does that a person cannot do quickly

Three things, in order of how much they matter.

**It finds leaks that no statistic can reach.** On Titanic, `boat` correlates
0.013 with survival and decides it outright; `body` has no computable
correlation at all and exists only for passengers who died; `sex` correlates
0.529 and is entirely legitimate. A correlation threshold keeps both leaks and
deletes the real predictor. There is no setting at which it does otherwise.

**It explains every verdict and refuses to act on its own.** Each column comes
back with the model's reasoning, the mechanism, the vote across three column
orders, and — when a data dictionary is supplied — the documented sentence the
model quoted in reaching its decision. Nothing is deleted without a human
confirming it.

**It measures the damage.** The same learners are fit twice, with and without
the confirmed columns, on identical folds with an identical hyperparameter
search on both arms, so the cleaned fit is the best version of itself rather
than a strawman.

## Live result, run during the hackathon

Titanic, three shuffled passes, data dictionary attached:

| column | verdict | mechanism | votes | \|r\| | which screen |
|---|---|---|---|---|---|
| `boat` | LEAK | CONSEQUENCE | 3/3 | 0.013 | model only |
| `body` | LEAK | CONSEQUENCE | 3/3 | — | model only |
| `sex` | OK | — | 0/3 | 0.529 | statistics only |

Both real leaks land in **model only** because correlation cannot reach them.
The only column correlation flags is a false positive. With the dictionary
attached the model's reasoning came back as direct quotations from the
documentation, which is what makes a dropped column defensible in writing:

> `boat` — *"Recorded only for passengers who were picked up alive, and
> therefore populated only for survivors."*

Downstream: macro F1 0.972 → 0.797, **35 mistakes with the leaks against 252 without**.

## Held-out evaluation

Because the tool's design was measured against a 15-dataset corpus, a separate
held-out set was built during the hackathon from five UCI datasets that appear
in none of it, under the same admission rule: a positive requires a verbatim
quotation from the dataset's own documentation. `evaluation/build_heldout.py`
refuses to run unless every quotation still matches what the repository serves.

Three of the five have **zero** positives, to measure false positives on
ordinary tables. One of the five has been scored so far; the partial state and
the remaining four are recorded in `evaluation/RESULTS.md` rather than averaged
away. CIRRHOSIS, 18 columns, 1 documented positive:

| | precision | recall | F1 |
|---|---|---|---|
| **Crucible** | **1.000** | **1.000** | **1.000** |
| correlation at the shipped threshold | 0.000 | 0.000 | 0.000 |
| correlation at its best threshold, *given the answers* | 0.167 | 1.000 | 0.286 |

The evaluation also earned its keep by finding a real defect: on a 36-column
table the audit failed outright, because no output-token budget was requested
and a reasoning model spent the provider's default allowance on internal
reasoning before writing anything. Fixed, with an adaptive retry and a
regression test.

## Running it

```bash
pip install "crucible-leakage[web]"
export CRUCIBLE_GEMINI_KEYS=key1,key2   # or leave unset to use the shared pool
uvicorn web.app:app --reload            # then open http://localhost:8000
```

The running service is the landing page and the tool at once: `/` is the
interface and `/api/docs` is the generated API reference. The written
explanation lives in the repository's README rather than being served.

Or without a browser at all:

```bash
pip install crucible-leakage
crucible audit titanic.csv --target survived \
  --at "at boarding, before any rescue or recovery" --measure
```

The demonstration button needs no key and no upload: it runs the real Titanic
table with a real downstream comparison computed server-side, so the column
editor and the re-measurement work in it too.

```bash
pip install -e ".[dev]" && pytest     # 59 tests, none of which touch a network
```

## What it does not do

It reads names and documentation, so it cannot judge a table whose columns are
anonymised — it refuses such tables rather than guessing. It covers
feature-level target leakage only, not group leakage, train/test contamination
or procedural leakage; those are different failures with different fixes, and a
tool that conflates them misleads. Interactions between column *pairs* are out
of scope, as is a plausibly-named column silently backfilled from the outcome.

One honest limit on the measurement itself: the downstream comparison cannot
detect a leak that both the tool and the reviewer missed. If a leak is absent
from both, both arms are inflated identically and the difference is zero. It
shows the tool agrees with you, not that you are right.
