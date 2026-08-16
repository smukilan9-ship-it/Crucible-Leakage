# Held-out evaluation

The design of this tool was measured against a 15-dataset research corpus. Any
figure taken from that corpus is fitted to it. This directory exists to produce
one number that is not.

## The set

Five UCI datasets, none of which appears in the research corpus, coded under the
same admission rule that corpus used: a column is a positive only if a citable
source statement, quoted verbatim, places it after the stated prediction point
or shows that it records the outcome.

| dataset | UCI id | documented positives |
|---|---|---|
| CIRRHOSIS | 878 | 1 |
| DROPOUT | 697 | 12 |
| ADULT | 2 | 0 |
| MATERNAL | 863 | 0 |
| CDC_DIABETES | 891 | 0 |

Three of the five have **zero** positives on purpose. A detector is only useful
if it stays quiet on an ordinary table, and a recall-only evaluation cannot see
that. ADULT is the hard control: several of its columns are strongly predictive
of income by design, so a detector that confuses predictiveness with leakage
fails on it.

`build_heldout.py` refuses to write anything unless every quotation in
`ANSWER_KEY.json` is still present, verbatim, in the description the repository
actually serves. A quotation that has drifted from its source is worse than no
quotation at all, because it looks like evidence.

## Status

**One of the five datasets has been scored.** The other four are fetched and
verified against the answer key but have not been run against a model, because
scoring them requires provider quota this repository does not carry. The command
is below and the harness is complete; what is missing is the run, not the code.

Reporting one dataset as though it were the evaluation would be the exact
failure this tool exists to argue against, so the partial state is stated here
rather than summarized away.

## CIRRHOSIS, 18 columns, 1 documented positive

Semantic screen at the shipped defaults: three shuffled orders, majority vote,
column names only, no data dictionary supplied.

| | precision | recall | F1 |
|---|---|---|---|
| **Crucible** | **1.000** | **1.000** | **1.000** |
| correlation at the shipped threshold (0.5) | 0.000 | 0.000 | 0.000 |
| correlation at its best threshold, given the answers | 0.167 | 1.000 | 0.286 |

The third row is not a fair comparison in the baseline's disfavour. It lets the
baseline see the answer key and choose its own best setting, which no real user
can do, and it still costs five false positives to reach the one true one. A
baseline should be beaten at its best rather than at a setting picked for it.

## Not yet scored

DROPOUT (12 positives), ADULT, MATERNAL, CDC_DIABETES. DROPOUT is the one that
matters most of the four: it carries more documented positives than the rest of
the held-out set combined, and it is the only remaining dataset that can move
recall.

## Reproducing

```bash
pip install -e ".[dev]"
python evaluation/build_heldout.py     # fetch, and verify every quotation
export CRUCIBLE_GEMINI_KEYS=...        # or FEATHERLESS_API_KEY for Kimi K3
python evaluation/score_heldout.py     # add --grounded to supply the dictionaries
```

The scorer calls a hosted model, so a re-run reproduces the protocol exactly and
the numbers approximately. Shuffle order is seeded, but a provider is free to
answer differently on a repeated call.

## What this evaluation found besides a score

It caught a real defect before any user did. On a 36-column table the audit
failed outright and returned nothing, because no output-token budget was
requested and a reasoning model spent the provider's default allowance on
internal reasoning before writing a character. An 18-column table succeeded on
the same code path, which is why the bug survived until a wider table hit it.
Fixed with a budget scaled to the column count and a doubling retry, and covered
by a regression test.
