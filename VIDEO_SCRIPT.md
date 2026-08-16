# Three-minute demo script

The rubric asks for two things from this video, and they are worth ten points
together: **live execution** and **architecture**. Both have to be on screen,
not described. Everything below is timed to fit 3:00 with about eight seconds
of slack.

**Before you record**

- Open the app and click *Replay the Titanic example* once, all the way to the
  results tab. Warms the page and means nothing is loading on camera.
- Have a second tab on the repository, scrolled to the README architecture
  diagram.
- Have a terminal open in the repo with a large font.
- Turn off notifications.
- Record at 1440 wide or better. The confusion matrices carry small numbers.

---

## 0:00 – 0:22 · The problem, in one concrete case

**On screen** — the landing page, showing the `boat` / `body` / `sex` figure.

> Every practitioner has met the model that scores 0.99 and turns out to be
> worthless. Usually one column is giving away the answer.
>
> Here is why that is hard to catch. On the Titanic table, this column —
> lifeboat number — correlates 0.013 with survival, and it decides survival
> outright, because it is only filled in for people who were rescued. This one,
> passenger sex, correlates 0.529 and is completely legitimate.
>
> Any correlation threshold you pick keeps the leak and deletes the real
> feature. And cross-validation cannot help, because the leaked column sits on
> both sides of every split.

*Do not rush this. It is the whole argument, and it is visual.*

---

## 0:22 – 0:38 · What the tool is

**On screen** — scroll slowly through the *Why this matters* panel.

> Crucible reads what your columns *mean*, against a target and against the
> moment the prediction actually happens. That third thing is the one nothing
> can compute for you, so you type it in.

---

## 0:38 – 1:25 · Live execution

**On screen** — upload a CSV, set the target, click a prediction-point example,
run the audit. Use a **small table** so this finishes on camera.

> I upload a table, name the column I am predicting, and say when the prediction
> would happen: at boarding, before any rescue.

*(audit runs — narrate over it)*

> Every column gets read three times in three different column orders, because
> order alone moves the score by up to 0.380, and the majority settles it.
>
> A correlation screen runs beside it, deliberately — it is the baseline, and
> where the two disagree is the interesting part.

**On screen** — the review tab. Click `boat` to expand it.

> Two columns flagged, both in *model only* — meaning correlation could not
> reach them. Every verdict comes with the model's reasoning, and if I attach a
> data dictionary, it quotes the documentation back at me, which is what makes
> a dropped column defensible in a paper.
>
> And nothing is deleted. I confirm each one.

---

## 1:25 – 2:10 · The measurement, and the number that matters

**On screen** — click through to results. Let the verdict banner land.

> Now it measures what those columns were worth. Same models, same folds, same
> hyperparameter search on both arms, so the cleaned model is the best version
> of itself and not a strawman.

*(pause on the banner)*

> The model scored 0.17 macro F1 higher than it deserved to.

**On screen** — scroll to the correlation-baseline box.

> This is the part I would point a judge at. The third arm is what a
> correlation threshold would have removed instead. It scores **0.974** —
> which is *worse than doing nothing*. It deleted the useful column and kept
> both leaks. The obvious approach does not just underperform here; it actively
> harms you.

**On screen** — the two confusion matrices.

> Thirty-five mistakes with the leaks. Two hundred and fifty-two without. That
> is what the leakage was hiding.

---

## 2:10 – 2:35 · Architecture

**On screen** — the README architecture diagram, then a quick pass over
`src/crucible/`.

> The architecture: an importable Python package with a command line, and a
> FastAPI service that contains no pipeline logic of its own — both call the
> same audit function, so they cannot drift apart. Providers sit behind one
> method, so Gemini or Featherless is one dictionary entry.

**On screen** — terminal, run `pytest`.

> Fifty-nine tests, none of which touch a network. Several exist because the
> bug they describe actually happened.

---

## 2:35 – 2:52 · The command line

**On screen** — run this, having pre-warmed it so output is instant:

```bash
crucible impact titanic.csv --target survived \
  --drop boat --drop body --against-correlation
```

> The same measurement without a browser, for putting in a CI pipeline.

---

## 2:52 – 3:00 · Close

**On screen** — back to the results page.

> Crucible does not delete your columns. It tells you which ones are giving
> away the answer, why, and what believing them was costing you.

---

## Things to say only if asked

- **Provenance.** The research behind the design predates the hackathon and is
  the author's own. Every line of code in the repository was written during the
  window. This is in the README and in `SUBMISSION.md`.
- **Held-out evidence.** On a UCI dataset the tool has never seen, it found the
  documented leak with precision 1.000 and recall 1.000, where the correlation
  baseline scored 0.000 at its shipped threshold and 0.286 even when allowed to
  see the answer key.
- **Honest limit.** The comparison cannot detect a leak that both the tool and
  the reviewer missed; it shows the tool agrees with you, not that you are
  right.

## Recording notes

- The demo replay is **labeled as a replay** in the interface. Do not claim it
  is calling a model live. If a judge asks, the verdicts are recorded from a
  real run and the downstream comparison is computed live — which is exactly
  what the label says.
- If the audit is slow on the day, record the upload and the result separately
  and cut. Say that you did.
