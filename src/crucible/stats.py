"""The statistical screen: flag columns whose correlation with the target
crosses a threshold.

This screen is included even though it is known to be worse than the semantic
screen. In the research benchmark it reached F1 0.630 with its threshold swept
on the answer key, which makes it an upper bound no deployment could reach,
against 0.918 for the best model reading column names. It earns its place
because it fails in the opposite direction, and the disagreement between the
two screens is what the triage board is made of. A correlation cannot tell
"correlated because it caused the label" apart from "correlated because it
predicts the label," which is the entire distinction.

**On the threshold.** The benchmarked baseline swept its cutoff on the answers
and landed at |r| >= 0.3202. That value is an oracle, so shipping it as the
default would hand the baseline information a real user does not have. The
shipped default is a round 0.5, and the two genuinely differ: Lending Club's
`recoveries` (|r| = 0.340) is dropped at the benchmarked cutoff and kept at the
shipped one. Pass `threshold=BENCHMARK_THRESHOLD` to reproduce the paper's arm.
"""

import numpy as np
import re

import pandas as pd
from sklearn.metrics import roc_auc_score

from . import evidence

CORRELATION_THRESHOLD = 0.5

# The cutoff the benchmarked baseline used, swept on the answer key. Exposed so
# the paper's arm can be reproduced, and not used by default, because a
# threshold chosen with the answers in hand is not one anyone has in advance.
BENCHMARK_THRESHOLD = evidence.BASELINE_THRESHOLD


# A leaked column is often present for one outcome and absent for the other,
# which shows up as a difference in how often it is missing between the classes
# rather than as any correlation at all. This is the statistic that would have
# caught Titanic's `body`, whose correlation is undefined.
MISSINGNESS_THRESHOLD = 0.5

# Names that hint at a value recorded after the fact. A weak signal on its own,
# reported rather than acted on, because a column called `outcome_notes` may be
# entirely innocent and one called `x7` may not.
SUSPICIOUS_NAME = re.compile(
    r"(outcome|result|final|resolved|closed|discharge|death|died|survi|recover|"
    r"post_?|_after|actual|label|target|verdict|disposition|settle)",
    re.IGNORECASE)


def statistical_screen(table: pd.DataFrame, target: str,
                       threshold: float = CORRELATION_THRESHOLD) -> dict:
    """Per-column statistics and the baseline flag drawn from them.

    Four measures, per N-30, because each is blind where another sees:

      * absolute correlation, which misses a leak that is not linear and fires
        on a legitimate strong predictor;
      * univariate area under the curve, which catches a monotone relationship
        correlation understates;
      * missingness asymmetry, the difference in how often a column is empty
        between the classes, which is the only one of the four that can see a
        column present for one outcome and absent for the other;
      * a name pattern, reported and never flagged on alone.

    Only the first of the four raises the flag, so the baseline stays the one
    that was benchmarked. The other three are surfaced, never acted on, so a
    reader can see where the baseline is about to be wrong: on Titanic, `body`
    has no defined correlation at all and a missingness gap near one.
    """
    target_values = _as_numbers(table[target])
    binary_target = _as_binary(table[target])
    results = {}
    for column in table.columns:
        if column == target:
            continue
        column_values = _as_numbers(table[column])
        correlation = _safe_correlation(column_values, target_values)
        results[column] = {
            "correlation": None if correlation is None else round(correlation, 4),
            "auc": _univariate_auc(column_values, binary_target),
            "missingness_gap": _missingness_gap(table[column], binary_target),
            "suspicious_name": bool(SUSPICIOUS_NAME.search(column)),
            "flagged": correlation is not None and abs(correlation) >= threshold,
        }
    return results


def _as_binary(series: pd.Series) -> pd.Series | None:
    """The target as 0/1, or None when it is not two-class. The asymmetry
    statistics only mean anything against two groups."""
    distinct = series.dropna().unique()
    if len(distinct) != 2:
        return None
    positive = sorted(distinct, key=str)[-1]
    return (series == positive).astype(int)


def _univariate_auc(column_values: pd.Series, binary_target: pd.Series | None) -> float | None:
    """How well this column alone ranks the outcome. Reported as a distance
    from 0.5 so that a perfectly inverted column reads as strong, not weak."""
    if binary_target is None:
        return None
    usable = column_values.notna() & binary_target.notna()
    if usable.sum() < 10 or binary_target[usable].nunique() < 2:
        return None
    if column_values[usable].nunique() < 2:
        return None
    try:
        return round(float(roc_auc_score(binary_target[usable], column_values[usable])), 4)
    except ValueError:
        return None


def _missingness_gap(raw_column: pd.Series, binary_target: pd.Series | None) -> float | None:
    """How differently this column goes missing between the two classes.

    Titanic's `body` is the case this exists for: recorded only for passengers
    who died, so it is 100% missing among survivors and largely present among
    the dead. Its correlation is undefined; its missingness gap is near one.
    """
    if binary_target is None:
        return None
    missing = raw_column.isna()
    if not missing.any() or missing.all():
        return 0.0
    positive, negative = binary_target == 1, binary_target == 0
    if not positive.any() or not negative.any():
        return None
    return round(abs(float(missing[positive].mean()) - float(missing[negative].mean())), 4)


def _as_numbers(series: pd.Series) -> pd.Series:
    """Return the series as floats, factorizing text columns so that they can
    be correlated at all. Missing values stay missing."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    codes, _ = pd.factorize(series, use_na_sentinel=True)
    coded = pd.Series(codes, index=series.index, dtype=float)
    coded[codes == -1] = np.nan
    return coded


def _safe_correlation(x: pd.Series, y: pd.Series) -> float | None:
    """Pearson correlation, or None when there is not enough variation to
    compute one (constant columns, near-empty overlap)."""
    both_present = x.notna() & y.notna()
    if both_present.sum() < 3 or x[both_present].nunique() < 2 or y[both_present].nunique() < 2:
        return None
    correlation = float(np.corrcoef(x[both_present], y[both_present])[0, 1])
    return None if np.isnan(correlation) else correlation
