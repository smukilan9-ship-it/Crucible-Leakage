# Crucible: a semantic screen for feature-level target leakage in tabular datasets

*Software metapaper, drafted to the Journal of Open Research Software structure.*

---

## (1) Overview

### Introduction

A feature exhibits **target leakage** when its value encodes the outcome it is
used to predict. The failure is old, well described, and still routine, because
nothing in an ordinary workflow catches it. Cross-validation cannot: the leaked
column sits on both sides of every split, so every fold independently confirms
that the model is excellent. The cost is deferred to deployment, where the
column does not yet exist, and to the literature, where the result cannot be
reproduced.

Existing tooling addresses the other failures that share the name — duplicated
units across a split, train/test contamination, identifier artefacts. These are
mechanically checkable. Feature-level target leakage is not, because whether a
column leaks depends on what the column *means* relative to a target and to a
moment of prediction. That triple is not recoverable from the values.

Crucible applies the condition set out by Kaufman, Rosset and Perlich (2011) to
a table the user has just been handed. A language model reads the column names,
and optionally the dataset's own data dictionary, against a stated target and a
stated prediction point. A correlation screen runs beside it as a deliberately
weaker second witness. Where the two disagree is where the tool earns its
place: a leaked column may have almost no linear correlation with the target,
and the most correlated column in a table is frequently legitimate.

The software does not delete anything. It triages, explains each verdict, and
then measures what the columns a human confirmed were actually worth, by
fitting the same learners twice on identical folds.

### Implementation and architecture

Three layers, deliberately separable.

**Library** (`src/crucible/`) is pure Python with no web dependency, installable
from PyPI metadata, and is the part a researcher imports. It ships a console
entry point, `crucible`, so the same audit is available from a shell script as
from a notebook.

| module | responsibility |
|---|---|
| `evidence` | every measured figure the software quotes, each carrying the section of the benchmark paper it is checked against |
| `intake` | reads the table, validates the target, refuses tables whose column names are placeholders, parses an optional data dictionary |
| `prompts` | assembles the audit prompt; holds both wordings of the derivation criterion verbatim |
| `providers` | Gemini, OpenAI, Anthropic and Featherless behind one `chat` method, with backoff, adaptive token budgets, a quota-aware key pool, and key-issuer detection so a credential is never sent to a vendor that did not issue it |
| `screen` | runs the audit several times in different column orders and combines the verdicts by majority |
| `stats` | the correlation screen |
| `fusion` | sorts columns into four buckets by which screens flagged them |
| `cli` | the `crucible` command: audit, impact, models |
| `contested` | a narrow second pass asking whether the documentation fixes a flagged column's value at the prediction point |
| `metrics` | classification diagnostics for two-class and multi-class targets, in one shape |
| `impact` | the downstream comparison: every column, the confirmed drops, and what a correlation threshold would have removed instead |

**Providers** (`src/crucible/providers/`) put every provider-specific concern —
authentication, retry policy, token budgets, key rotation — behind a single
`chat` method, which is why no other module knows which provider is in use and
why the tests cover the whole pipeline without a network. A `KeyPool` supports
public demonstrations on free-tier keys: quota is spent least-used-first so one
visitor cannot drain the pool, a request is counted when started rather than
when it succeeds, and no key or key fragment is returned by any status call or
written to any log. That last property is asserted by a test.

**Service** (`web/app.py`) is FastAPI and contains no pipeline logic of its own;
it calls the same `run_audit` the command line calls, so the two interfaces
cannot drift apart. A long audit is a background task
that streams progress over Server-Sent Events and *pauses* in a
`awaiting_review` state until a human posts decisions. The pause is a state in
the job model, not a user-interface convention.

**Interface** (`web/static/`) is dependency-free HTML, CSS and JavaScript served
as static files. The uploaded file is parsed in the browser, so the table is
displayed without being sent anywhere; only column names, and descriptions if a
dictionary is supplied, are transmitted.

Four design decisions are worth stating because each is a constraint rather
than a preference, and each was measured before it was adopted.

**Sample values are never sent to the model.** Whether a column leaks is a fact
about where its value came from, and that is not visible in the values. This is
enforced by a test.

**Columns are judged together, not one at a time.** Leakage is relational: a
residual class such as `Other_Faults` is only recognizable when its siblings
are in the same list.

**The audit runs several times in different column orders.** Column order alone
moved one model's score by 0.380 between two runs of a byte-identical prompt, so
a single pass is not a stable answer, and the verdicts are combined by majority.
This is stated as a hedge rather than a remedy: order sensitivity is a property
of the model, and a model measured stable returns the same answer under every
order, at which point the vote costs three calls to reproduce one result. The
shuffle count is a parameter for that reason.

**The wording of the derivation criterion is chosen per model from measured
results.** Two wordings exist — one stating the criterion in temporal terms,
one as a reconstruction test — and neither is uniformly better. They fail in
mirror image: the temporal wording lets some models excuse a column recorded at
the same moment as the target, while the reconstruction wording causes others
to flag every column of a table whose target is a rule over its own sensors.
The mapping from model to wording is a lookup table with the measured F1 for
each cell recorded beside it.

### Quality control

**Where the design's numbers come from.** Every measured claim in this software
is taken from the author's benchmark of 15 datasets, 604 columns and 68
documented positives, and none of it was measured on this software. The
distinction matters for a metapaper: what is validated here is the tool's
faithfulness to a design, not the design itself. Three consequences are carried
into the code rather than left in the paper.

The correlation baseline reaches F1 0.630 with its threshold swept on the answer
key, an upper bound no deployment could reach, against 0.918 for the best model
tested; the module implementing that baseline says so in its own docstring and
exposes the benchmarked threshold as a named constant beside the deployable one.
Order-averaging was proposed as a general remedy for shuffle sensitivity and
then **withdrawn**, because a model that is stable under reordering has nothing
to average; the shuffle count is therefore a parameter and the docstring states
the withdrawal rather than the original claim. And the account that one clause
lifts a single leakage subtype held on the main corpus and **did not survive the
held-out stratum**, so the prompt module records both the result and its
limit.

A tool that quotes only the half of its evidence that flatters it is not
reusable, because the first person to check will find the other half.

**Unit and integration tests.** `pytest` covers every module that decides
something: reply parsing including invented column names and prose-wrapped
JSON, majority combination, reason de-duplication, prompt assembly for both
criterion wordings and with and without a dictionary, correlation edge cases,
bucket fusion, metrics for two-class and multi-class targets, identifier
detection, class-label ordering, encoding stability across arms, and an
end-to-end downstream comparison on a table with a planted leak of known size.

Several tests exist because the defect they describe occurred. One asserts that
no reported number is non-finite, because a single not-a-number in the metrics
structure returned HTTP 500 to the browser. Another asserts that identifier
names match by word and not by substring, because `id` occurs inside `paid` and
inside `cons.price.idx`, and treating either as a unit identifier would
silently change how every fold is built.

No test calls a language model. Everything model-dependent is behind one
function, and what is tested is the code surrounding it.

**Held-out evaluation.** The design of this tool was measured against a corpus
of 15 datasets. To report a figure not fitted to that corpus, `evaluation/`
contains a held-out set of five UCI datasets, none of which appears in it, with
an answer key built under the same admission rule: a column is a positive only
if a citable source statement, quoted verbatim, places it after the stated
prediction point or shows it records the outcome.

`build_heldout.py` fetches the datasets and refuses to write anything unless
every quotation in the answer key is present, verbatim, in the description the
repository actually serves. A quotation that has drifted from its source is
worse than no quotation, because it looks like evidence.

Three of the five datasets have **zero** positives. They are not padding: a
detector is only useful if it stays quiet on an ordinary table, and a
recall-only evaluation cannot see that. One of the three is a hard control
whose columns are strongly predictive of the target by design, so a detector
that confuses predictiveness with leakage fails on it.

Results, including which datasets have been scored and which have not, are in
`evaluation/RESULTS.md`. One of the five is scored at the time of writing;
stating that is preferable to reporting a single dataset as an evaluation.

**Reproducing.**

```bash
pip install -e ".[dev]"
pytest                                        # 59 tests, no network
python evaluation/build_heldout.py            # fetch and verify the answer key
export FEATHERLESS_API_KEY=...
python evaluation/score_heldout.py            # re-score the held-out set
```

The evaluation calls a hosted model, so a re-run reproduces the protocol
exactly and the numbers approximately. The verdicts that produced the reported
figures are stored alongside them.

---

## (2) Availability

**Operating system.** Any system with Python 3.11 or later. Developed on macOS,
deployed on Linux.

**Programming language.** Python 3.11+ for the library, command line and
service;
dependency-free JavaScript, HTML and CSS for the interface.

**Additional system requirements.** None beyond a browser. Memory scales with
the uploaded table; the downstream comparison samples to 5,000 rows.

**Dependencies.** The library needs httpx, pandas, scikit-learn and numpy. The
optional `web` extra adds FastAPI, Uvicorn and python-multipart; the `dev`
extra adds pytest and, for rebuilding the held-out evaluation, `ucimlrepo` and
`certifi`. All bounded in `pyproject.toml`.

An API key for a hosted model provider is required for the audit. The
downstream comparison, the test suite and the demonstration run without one.

**List of contributors.** Mukilan.

**Software location.**

*Archive.* To be deposited at release. The intended archive is Zenodo, linked to
the source repository so that each tagged release mints its own DOI.

*Persistent identifier.* Pending deposit. No DOI is claimed here, because a
metapaper that cites an identifier which does not resolve is worse than one that
says the deposit has not happened yet.

*License.* MIT, in `LICENSE`.

*Publisher.* The author.

*Version.* 0.1.0. *Date published.* Pending deposit.

**Language.** English.

---

## (3) Reuse potential

The library installs and imports independently of the web service, and ships a
command-line entry point. `run_audit` takes a dataframe and returns a
dictionary; `quantify` takes a dataframe and a column list and returns another.
Nothing in either requires a server, a configuration file or a working
directory.

Three reuse cases the design anticipates.

**As a screening step in a study.** The audit produces a per-column report
pairing each verdict with its reasoning and, when a data dictionary is
supplied, with the documented sentence the model quoted in reaching it. That
pairing is what makes a dropped column defensible in writing rather than merely
plausible.

**As a measurement instrument.** The arm comparison is usable on its own to
answer "what is this column worth?" for any column set, not only one the screen
proposed. The interface exposes this directly: any column can be added or
removed and the comparison re-run.

**As a substrate for prompt research.** Both criterion wordings are held
verbatim and selected through a lookup table in `models.py`, so adding a third
is one dictionary entry. Adding a provider is one entry in another. The
held-out evaluation harness scores any catalogued model through a single
command-line flag, which makes the model-by-wording grid cheap to extend.

**Limits, stated plainly.** The tool reads names and documentation, so it
cannot judge a table whose columns are anonymised — it refuses such tables
rather than guessing. It addresses feature-level target leakage only, not group
leakage, contamination or procedural leakage. It handles tabular data with a
categorical target. Interactions between column *pairs* are out of scope, as is
a plausibly-named column silently backfilled from the outcome.

**Support.** Issues and pull requests via the repository.

---

## References

Kaufman, S., Rosset, S., & Perlich, C. (2011). Leakage in Data Mining:
Formulation, Detection, and Avoidance. *Proceedings of the 17th ACM SIGKDD
International Conference on Knowledge Discovery and Data Mining*, 556–563.
Extended as Kaufman, Rosset, Perlich & Stitelman, *ACM Transactions on
Knowledge Discovery from Data*, 6(4), 2012. Supplies the definition this
software implements: a feature is legitimate if it is available at the
prediction point.

Kapoor, S., & Narayanan, A. (2023). Leakage and the Reproducibility Crisis in
Machine-Learning-Based Science. *Patterns*, 4(9), 100804. Establishes the scale
of the problem this software addresses, across 17 fields and 294 papers.

Bordt, S., Nori, H., Rodrigues, V., Nushi, B., & Caruana, R. (2024). Elephants
Never Forget: Memorization and Learning of Tabular Data in Large Language
Models. *Conference on Language Modeling*. The standing objection to reading
well-known public tables with a language model. The benchmark behind this
software runs their released checker and a renaming control; the software itself
inherits the objection and does not resolve it.

The benchmark that fixes this software's design decisions is the author's own
and is under submission. This metapaper does not restate its results as though
they had been established here, and every figure quoted in the source is
attributed to it.
