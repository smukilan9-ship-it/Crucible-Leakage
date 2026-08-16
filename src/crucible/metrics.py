"""Classification diagnostics computed from pooled out-of-fold predictions.

Both arms are scored by exactly this function, so any difference between them
comes from which columns were present and nothing else.

Targets with two classes and targets with more than two are both handled. The
shape of the result is deliberately the same either way, so the interface has
one thing to draw rather than two:

  * the confusion matrix is always a square list of lists, actual down the
    rows and predicted across the columns;
  * per-class precision, recall and F1 are always a list, one entry per class;
  * macro and weighted averages are always present;
  * the receiver operating characteristic and precision-recall curves are
    always a list of named curves, which for a two-class target is a list of
    one and for a multi-class target is one curve per class, computed one
    class against the rest.

The one genuine difference is where the decision boundary comes from. With two
classes we search every threshold and keep the one with the best F1, because a
default of 0.5 can flatter one arm over the other for reasons that have nothing
to do with leakage. With more than two classes the prediction is the class with
the highest probability, which is the standard rule and needs no search.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def evaluate(actual: np.ndarray, probabilities: np.ndarray, class_labels: list,
             threshold: float | None = None) -> dict:
    """Full diagnostic set for one arm.

    `actual` holds class indices, `probabilities` has one column per class in
    the same index order, and `class_labels` gives the human-readable name for
    each index.
    """
    actual = np.asarray(actual)
    probabilities = np.asarray(probabilities)
    n_classes = len(class_labels)
    class_indices = list(range(n_classes))

    if n_classes == 2:
        # The threshold is supplied by the caller, chosen inside each fold on
        # the training part alone. Searching it here, over the very predictions
        # about to be scored, is selection on the evaluation set: it reports the
        # luckiest cut rather than an achievable one, and the overstatement is
        # larger on the weaker arm, so it distorts a comparison as well as a
        # score. Only when no threshold is given — a caller scoring a single
        # arm out of context — is one searched, and the result says so.
        if threshold is None:
            f1_best, threshold = _best_f1(actual, probabilities[:, 1])
            threshold_source = "searched on these predictions"
        else:
            f1_best, threshold_source = None, "chosen on training folds only"
        predicted = (probabilities[:, 1] >= threshold).astype(int)
    else:
        f1_best, threshold, threshold_source = None, None, "highest probability"
        predicted = probabilities.argmax(axis=1)

    matrix = confusion_matrix(actual, predicted, labels=class_indices)
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=class_indices, zero_division=0
    )

    # With two classes the reported F1 is the one from the threshold search, so
    # that the headline number and the confusion matrix describe the same
    # decision rule rather than two different ones.
    macro_f1 = float(np.mean(f1))
    weighted_f1 = float(np.average(f1, weights=support)) if support.sum() else 0.0

    result = {
        "n_classes": n_classes,
        "labels": [str(label) for label in class_labels],
        "confusion": [[int(cell) for cell in row] for row in matrix],
        "support": [int(value) for value in support],
        "per_class": [
            {
                "label": str(class_labels[index]),
                "precision": round(float(precision[index]), 4),
                "recall": round(float(recall[index]), 4),
                "f1": round(float(f1[index]), 4),
                "support": int(support[index]),
            }
            for index in class_indices
        ],
        "macro": {
            "precision": round(float(np.mean(precision)), 4),
            "recall": round(float(np.mean(recall)), 4),
            "f1": round(macro_f1, 4),
        },
        "weighted": {
            "precision": round(float(np.average(precision, weights=support)), 4) if support.sum() else 0.0,
            "recall": round(float(np.average(recall, weights=support)), 4) if support.sum() else 0.0,
            "f1": round(weighted_f1, 4),
        },
        "accuracy": round(float((predicted == actual).mean()), 4),
        "balanced_accuracy": round(float(np.mean(recall)), 4),
        "matthews": round(float(matthews_corrcoef(actual, predicted)), 4),
        "auc": _auc(actual, probabilities, n_classes),
        "roc_curves": _roc_curves(actual, probabilities, class_labels),
        "pr_curves": _pr_curves(actual, probabilities, class_labels),
        "class_share": [round(float((actual == index).mean()), 4) for index in class_indices],
    }

    result["threshold_source"] = threshold_source
    if n_classes == 2:
        result["threshold"] = round(float(threshold), 4)
        if f1_best is not None:
            result["best_f1"] = round(float(f1_best), 4)
        result["average_precision"] = round(
            float(average_precision_score(actual, probabilities[:, 1])), 4)
        result["sweep"] = _threshold_sweep(actual, probabilities[:, 1])
    return result


def headline(arm: dict) -> dict:
    """The four numbers the results page compares, pulled out so the interface
    does not have to know where each one lives."""
    return {
        "macro_f1": arm["macro"]["f1"],
        "weighted_f1": arm["weighted"]["f1"],
        "accuracy": arm["accuracy"],
        "auc": arm["auc"],
    }


def _auc(actual, probabilities, n_classes) -> float | None:
    """Area under the receiver operating characteristic curve.

    Returns None rather than a misleading number when a class is missing from
    the pooled predictions, which can happen on a very small or very skewed
    table.
    """
    if len(np.unique(actual)) < n_classes:
        return None
    try:
        if n_classes == 2:
            return round(float(roc_auc_score(actual, probabilities[:, 1])), 4)
        return round(float(roc_auc_score(
            actual, probabilities, multi_class="ovr", average="macro")), 4)
    except ValueError:
        return None


def _roc_curves(actual, probabilities, class_labels) -> list:
    """One curve for a two-class target, one curve per class otherwise, each
    computed for that class against all the others."""
    curves = []
    for index in _curve_indices(class_labels):
        one_against_rest = (actual == index).astype(int)
        if len(np.unique(one_against_rest)) < 2:
            continue
        false_positive_rate, true_positive_rate, _ = roc_curve(
            one_against_rest, probabilities[:, index])
        curves.append({
            "label": str(class_labels[index]),
            # The area for this class alone. The headline `auc` is averaged
            # over the classes, so it cannot label a single curve, and a curve
            # drawn without its own number is a shape rather than a result.
            "auc": round(float(roc_auc_score(one_against_rest, probabilities[:, index])), 4),
            "points": _thin(list(zip(false_positive_rate, true_positive_rate))),
        })
    return curves


def _pr_curves(actual, probabilities, class_labels) -> list:
    curves = []
    for index in _curve_indices(class_labels):
        one_against_rest = (actual == index).astype(int)
        if len(np.unique(one_against_rest)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(
            one_against_rest, probabilities[:, index])
        curves.append({
            "label": str(class_labels[index]),
            "baseline": round(float(one_against_rest.mean()), 4),
            "ap": round(float(average_precision_score(
                one_against_rest, probabilities[:, index])), 4),
            "points": _thin(list(zip(recall, precision))),
        })
    return curves


def _curve_indices(class_labels) -> list:
    """For two classes only the positive class is worth drawing, because the
    negative curve is its mirror image and adds nothing."""
    return [1] if len(class_labels) == 2 else list(range(len(class_labels)))


def _best_f1(actual: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Best F1 over every candidate threshold. Both arms get this same search,
    so neither is advantaged by where its threshold happens to sit."""
    order = np.argsort(-scores)
    sorted_actual = actual[order]
    true_positives = np.cumsum(sorted_actual)
    predicted_positives = np.arange(1, len(actual) + 1)
    actual_positives = sorted_actual.sum()
    if actual_positives == 0:
        return 0.0, 0.5
    precision = true_positives / predicted_positives
    recall = true_positives / actual_positives
    denominator = precision + recall
    f1_scores = np.zeros_like(denominator)
    np.divide(2 * precision * recall, denominator, out=f1_scores, where=denominator > 0)
    best = int(np.argmax(f1_scores))
    return float(f1_scores[best]), float(scores[order][best])


def _threshold_sweep(actual: np.ndarray, scores: np.ndarray, points: int = 60) -> list:
    """Precision, recall and F1 across the threshold range, so the operating
    point is a choice the reader can see rather than a number to trust."""
    thresholds = np.linspace(scores.min(), scores.max(), points)
    positives = actual.sum()
    sweep = []
    for threshold in thresholds:
        labels = scores >= threshold
        true_positive = float((labels & (actual == 1)).sum())
        predicted_positive = float(labels.sum())
        precision = true_positive / predicted_positive if predicted_positive else 1.0
        recall = true_positive / positives if positives else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        sweep.append([round(float(threshold), 4), round(precision, 4),
                      round(recall, 4), round(f1, 4)])
    return sweep


def _thin(points, max_points: int = 90) -> list:
    points = list(points)
    if len(points) <= max_points:
        indices = range(len(points))
    else:
        indices = np.unique(np.linspace(0, len(points) - 1, max_points).astype(int))
    return [[round(float(points[i][0]), 4), round(float(points[i][1]), 4)] for i in indices]
