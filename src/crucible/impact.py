"""Impact measurement: what were the confirmed leaks actually buying?

The comparison is two arms that differ only in which columns are present:
one fit with every column, one fit with the confirmed drops removed. To make
the comparison unimpeachable, every arm gets the exact same treatment:

  * the same fixed, tuning-free encoding;
  * the same cross-validation folds, computed once and reused everywhere;
  * the same small hyperparameter search, so the honest arm is the best
    version of itself and nobody can claim it was handicapped;
  * two learner families, because leakage is a property of the data, and an
    effect that appears under only one learner is an artifact of the learner.

When the table has repeated units (the same patient across rows), folds are
built on the unit, not the row, so nothing leaks across the split inside
either arm.

Everything reported comes from pooled out-of-fold predictions: each row is
predicted by a model that never saw it. From those predictions we report AUC,
the full ROC curve, per-fold AUC spread, and the best F1 over all thresholds,
computed by the identical procedure for every arm.
"""

import os
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold

from . import metrics

IDENTIFIER_NAME_TOKENS = {"id", "ids", "nbr", "key", "uuid", "guid"}
MIN_DISTINCT_UNITS = 20
# A shared-CPU host fits far more slowly than a laptop, and the comparison is
# ninety fits. Lowering this trades a little precision for a demonstration that
# finishes: the sample is stratified by the fold builder either way, and the
# report says how many rows were used.
MAX_ROWS = int(os.environ.get("CRUCIBLE_MAX_ROWS", "5000"))

SEARCH_GRIDS = {
    "random_forest": [
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 3},
        {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 1},
    ],
    "gradient_boosting": [
        {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3},
        {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5},
    ],
}


class ImpactError(Exception):
    pass


def _looks_like_identifier(column: str) -> bool:
    """Match identifier names by word, not by substring.

    Substring matching is a trap here: "id" appears inside "paid" and inside
    "cons.price.idx", and treating either as a unit identifier would silently
    change how every fold is built.
    """
    name = column.strip()
    tokens = [token for token in re.split(r"[^a-z0-9]+", name.lower()) if token]
    if any(token in IDENTIFIER_NAME_TOKENS for token in tokens):
        return True
    # camelCase identifiers ("serviceID", "objID") that tokenizing cannot split
    return bool(re.search(r"[a-z]ID$", name))


def detect_group_column(table: pd.DataFrame, target: str) -> str | None:
    """Find a column that looks like a unit identifier with repeats.

    Three conditions together, because any one alone misfires: the name must
    read like an identifier, values must repeat (otherwise grouping changes
    nothing), and there must be enough distinct units to build folds from. The
    last condition is what keeps a small-count column such as "number of
    comorbidities" from being mistaken for a patient identifier.
    """
    for column in table.columns:
        if column == target or not _looks_like_identifier(column):
            continue
        distinct = table[column].nunique()
        repeats = distinct < len(table) * 0.9
        enough_units = distinct >= MIN_DISTINCT_UNITS
        if repeats and enough_units:
            return column
    return None


def quantify(table: pd.DataFrame, target: str, drop_list: list[str], n_splits: int = 5,
             baseline_drop_list: list[str] | None = None, on_event=None) -> dict:
    """Fit every arm and report what the dropped columns were worth.

    `on_event`, when given, is called with (name, payload) as the work proceeds.
    It is purely observational: nothing it does can change a number, and a
    caller that omits it gets byte-identical results. It exists so an interface
    can show the fitting happening rather than a spinner, including the trees
    the model actually built.
    """
    n_rows_total = len(table)
    if n_rows_total > MAX_ROWS:
        table = table.sample(n=MAX_ROWS, random_state=0).reset_index(drop=True)

    target_values, class_labels = _target_as_classes(table[target])
    group_column = detect_group_column(table, target)
    excluded = {target} | ({group_column} if group_column else set())

    all_columns = [column for column in table.columns if column not in excluded]
    honest_columns = [column for column in all_columns if column not in set(drop_list)]
    if not honest_columns:
        raise ImpactError("drop list removes every feature column")

    def encoded_subset(keep: list[str]) -> pd.DataFrame:
        kept = set(keep)
        return features_all[[
            name for name in features_all.columns
            if name.rsplit("=", 1)[0] in kept or name in kept
        ]]

    features_all = _encode(table[all_columns])
    features_honest = encoded_subset(honest_columns)

    # N-81. A third arm, when the caller can say what a correlation threshold
    # would have removed. Without it the comparison is only "these columns
    # versus none of them", which says nothing about whether the semantic
    # screen beat the cheap alternative on this particular table.
    features_baseline = None
    if baseline_drop_list is not None:
        baseline_columns = [c for c in all_columns if c not in set(baseline_drop_list)]
        if baseline_columns:
            features_baseline = encoded_subset(baseline_columns)
    groups = table[group_column] if group_column else None
    folds = _build_folds(features_all, target_values, groups, n_splits)

    result = {
        "group_column": group_column,
        "n_rows_total": int(n_rows_total),
        "n_rows_used": int(len(table)),
        "n_classes": len(class_labels),
        "class_labels": [str(label) for label in class_labels],
        "configs_searched": {name: len(grid) for name, grid in SEARCH_GRIDS.items()},
        "learners": {},
    }

    # Which of the requested drops actually removed a feature. A name that is
    # not a feature column — the target itself, the grouping column, a typo, a
    # column the file does not have — changes nothing, and silently ignoring it
    # is how an arm ends up identical to the arm it is supposed to be compared
    # against.
    requested = list(dict.fromkeys(drop_list))
    feature_names = set(all_columns)
    result["drops_applied"] = [name for name in requested if name in feature_names]
    result["drops_ignored"] = [name for name in requested if name not in feature_names]
    def emit(name: str, payload: dict) -> None:
        if on_event:
            on_event(name, payload)

    arm_plan = [("with_leaks", features_all), ("honest", features_honest)]
    if features_baseline is not None:
        arm_plan.append(("baseline", features_baseline))

    # Two arms are a comparison only when they differ. Where they do not, the
    # cache below returns one fit for both and the difference between them is
    # exactly zero — which reads as "there was no leakage" when what happened
    # is "nothing was removed". Recorded here, before any fitting, so every
    # interface can say which of the two it is looking at.
    result["identical_arms"] = _identical_arms(arm_plan)

    emit("plan", {
        "learners": list(SEARCH_GRIDS),
        "arms": [name for name, _ in arm_plan],
        # Arms that share a column set are fitted once, so a progress bar drawn
        # against the number of arms would stall short of full and an estimate
        # drawn from it would overshoot. This is the number actually fitted.
        "distinct_arms": len({tuple(frame.columns) for _, frame in arm_plan}),
        "folds": len(folds),
        "configs": {name: len(grid) for name, grid in SEARCH_GRIDS.items()},
        "rows": int(len(table)),
        "features": {name: int(frame.shape[1]) for name, frame in arm_plan},
    })

    arm_cache: dict = {}
    for learner_name in SEARCH_GRIDS:
        with_leaks = _best_arm(learner_name, features_all, target_values, folds,
                               class_labels, arm_cache, emit, "with_leaks")
        honest = _best_arm(learner_name, features_honest, target_values, folds,
                           class_labels, arm_cache, emit, "honest")
        entry = {
            "with_leaks": with_leaks,
            "honest": honest,
            "inflation": _inflation(with_leaks, honest),
        }
        if features_baseline is not None:
            baseline = _best_arm(learner_name, features_baseline, target_values, folds,
                                 class_labels, arm_cache, emit, "baseline")
            entry["baseline"] = baseline
            # Residual against the arm the user actually confirmed. Near zero
            # means the correlation threshold would have done the same job here;
            # positive means it left something in.
            entry["baseline_residual"] = _inflation(baseline, honest)
        result["learners"][learner_name] = entry

    result["arms_refitted"] = len(arm_cache)
    emit("done", {"arms_refitted": len(arm_cache)})
    return result


def _identical_arms(arm_plan: list) -> list[dict]:
    """Which pairs of arms are the same column set, and therefore the same fit.

    Compared on encoded columns rather than on the drop list, because that is
    what the learner sees: dropping a column that encodes to nothing, or a name
    the table does not have, leaves the two frames identical no matter how
    different the two requests looked.
    """
    columns = [(name, tuple(frame.columns)) for name, frame in arm_plan]
    pairs = []
    for index, (left, left_columns) in enumerate(columns):
        for right, right_columns in columns[index + 1:]:
            if left_columns == right_columns:
                pairs.append({"arms": [left, right], "n_features": len(left_columns)})
    return pairs


def _inflation(with_leaks: dict, honest: dict) -> dict:
    """How much each headline number was overstated by keeping the leaks.

    A plain subtraction, so the sign is unambiguous: positive means the leaked
    fit looked better than the honest one.
    """
    def gap(reader):
        left, right = reader(with_leaks), reader(honest)
        if left is None or right is None:
            return None
        return round(left - right, 4)

    return {
        "macro_f1": gap(lambda arm: arm["macro"]["f1"]),
        "weighted_f1": gap(lambda arm: arm["weighted"]["f1"]),
        "macro_precision": gap(lambda arm: arm["macro"]["precision"]),
        "macro_recall": gap(lambda arm: arm["macro"]["recall"]),
        "accuracy": gap(lambda arm: arm["accuracy"]),
        "balanced_accuracy": gap(lambda arm: arm["balanced_accuracy"]),
        "matthews": gap(lambda arm: arm["matthews"]),
        "auc": gap(lambda arm: arm["auc"]),
        "extra_errors_when_honest": int(_error_count(honest) - _error_count(with_leaks)),
    }


def _error_count(arm: dict) -> int:
    """Every off-diagonal cell of the confusion matrix: how many rows the fit
    got wrong."""
    return sum(
        cell
        for row_index, row in enumerate(arm["confusion"])
        for column_index, cell in enumerate(row)
        if row_index != column_index
    )


def _best_arm(learner_name: str, features: pd.DataFrame, target_values: pd.Series,
              folds: list, class_labels: list, cache: dict | None = None,
              emit=None, arm_label: str = "") -> dict:
    """Run the full search grid for one arm and report the best configuration
    with its full diagnostic set.

    The configuration is chosen by pooled out-of-fold macro F1, which weights
    every class equally and so cannot be won by a fit that ignores a small
    class. Both arms are selected the same way, so the honest arm really is
    the best version of itself.
    """
    # N-82. Arms collapse: two detectors that flag the same columns produce the
    # same column set, and refitting it is pure waste. Keyed on the retained
    # columns and the learner, because those are what determine the fit.
    key = (learner_name, tuple(features.columns))
    if cache is not None and key in cache:
        # Two detectors that flagged the same columns give the same arm. Say so
        # rather than showing a fit that is not happening.
        if emit:
            emit("arm_reused", {"learner": learner_name, "arm": arm_label})
        return {**cache[key], "refit": False}

    if emit:
        emit("arm_start", {"learner": learner_name, "arm": arm_label,
                           "features": int(features.shape[1]),
                           "configs": len(SEARCH_GRIDS[learner_name])})
    best = None
    actual = target_values.to_numpy()
    for config_index, config in enumerate(SEARCH_GRIDS[learner_name]):
        probabilities, threshold = _out_of_fold_probabilities(
            learner_name, config, features, target_values, folds, len(class_labels),
            emit=emit, context={"learner": learner_name, "arm": arm_label,
                                "config": config_index, "config_detail": config})
        score = metrics.evaluate(actual, probabilities, class_labels,
                                 threshold=threshold)["macro"]["f1"]
        if best is None or score > best["score"]:
            best = {"score": score, "config": config,
                    "probabilities": probabilities, "threshold": threshold}

    probabilities, threshold = best["probabilities"], best["threshold"]
    arm = {
        **metrics.evaluate(actual, probabilities, class_labels, threshold=threshold),
        "per_fold_macro_f1": _per_fold_scores(
            actual, probabilities, folds, class_labels, threshold),
        "config": best["config"],
        "n_features": int(features.shape[1]),
    }
    if cache is not None:
        cache[key] = arm
    if emit:
        emit("arm_done", {"learner": learner_name, "arm": arm_label,
                          "macro_f1": arm["macro"]["f1"], "config": best["config"]})
    return {**arm, "refit": True}


def _per_fold_scores(actual, probabilities, folds, class_labels, threshold=None) -> list:
    """Macro F1 within each held-out fold, so the reader can see whether the
    gap between the arms is consistent or driven by one lucky split."""
    scores = []
    for _, test_index in folds:
        if len(np.unique(actual[test_index])) < 2:
            continue
        fold = metrics.evaluate(actual[test_index], probabilities[test_index],
                                class_labels, threshold=threshold)
        scores.append(fold["macro"]["f1"])
    return scores


def _out_of_fold_probabilities(learner_name, config, features, target_values,
                               folds, n_classes, emit=None,
                               context: dict | None = None) -> tuple[np.ndarray, float | None]:
    """Out-of-fold probabilities, plus a decision threshold that never saw a
    held-out row.

    Each outer fold fits the learner on its training part, and — for a
    two-class target — runs a small inner cross-validation *inside that
    training part* to pick the threshold that maximises F1 there. The reported
    threshold is the mean of the per-fold choices, so it is a number a
    deployment could actually have arrived at before seeing any of this data.

    Choosing it on the pooled predictions instead is the standard way to
    overstate a classifier. It is worse than merely optimistic here: measured
    on Titanic the overstatement was +0.0015 on the arm with leaks and +0.0190
    on the cleaned arm, so it shrinks the very gap this tool exists to report.
    """
    probabilities = np.full((len(target_values), n_classes), 1.0 / n_classes)
    fold_thresholds = []

    for fold_number, (train_index, test_index) in enumerate(folds):
        learner = _make_learner(learner_name, config)
        learner.fit(features.iloc[train_index], target_values.iloc[train_index])

        # One real tree off the fitted model, on the first fold of each config.
        # The root split is the interesting part: an arm that still holds a
        # leaked column reaches for it immediately, and that is visible here
        # before any score is computed.
        if emit and fold_number == 0:
            estimator = _first_tree(learner)
            if estimator is not None:
                names = list(features.columns)
                emit("tree", {**(context or {}), "fold": fold_number,
                              "trees_in_fit": len(getattr(learner, "estimators_", [])),
                              "root_census": root_split_census(learner, names),
                              **tree_sketch(estimator, names)})

        fold_probabilities = learner.predict_proba(features.iloc[test_index])
        # A fold can be missing a rare class entirely, so place each column by
        # the class the learner actually saw rather than by position.
        for position, class_index in enumerate(learner.classes_):
            probabilities[test_index, int(class_index)] = fold_probabilities[:, position]

        if n_classes == 2:
            chosen = _threshold_from_training(
                learner_name, config, features, target_values, train_index)
            if chosen is not None:
                fold_thresholds.append(chosen)
        if emit:
            emit("fold", {**(context or {}), "fold": fold_number,
                          "of": len(folds), "train_rows": int(len(train_index))})

    threshold = float(np.mean(fold_thresholds)) if fold_thresholds else None
    return probabilities, threshold


def _threshold_from_training(learner_name, config, features, target_values,
                             train_index, inner_splits: int = 3) -> float | None:
    """Pick a decision threshold using only the training part of one outer fold.

    An inner cross-validation produces predictions for training rows from
    models that did not fit them, and the threshold that maximises F1 on those
    is returned. Nothing here touches the outer fold's held-out rows.
    """
    inner_features = features.iloc[train_index]
    inner_target = target_values.iloc[train_index]
    if inner_target.nunique() < 2 or len(inner_target) < inner_splits * 4:
        return None

    scores = np.full(len(inner_target), 0.5)
    splitter = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=0)
    for fit_index, score_index in splitter.split(inner_features, inner_target):
        learner = _make_learner(learner_name, config)
        learner.fit(inner_features.iloc[fit_index], inner_target.iloc[fit_index])
        predicted = learner.predict_proba(inner_features.iloc[score_index])
        positive = list(learner.classes_).index(1) if 1 in learner.classes_ else -1
        scores[score_index] = predicted[:, positive]

    _, threshold = metrics._best_f1(inner_target.to_numpy(), scores)
    return threshold


def tree_sketch(estimator, feature_names: list, max_nodes: int = 31) -> dict:
    """A compact description of one tree that a fitted model actually built.

    Not a drawing and not a summary: the split feature, threshold and sample
    count at each node, read straight off `estimator.tree_`. An interface can
    draw it, and what it draws is what the model did.

    Breadth first and capped, because a forest tree can run to thousands of
    nodes and the first few levels are where the interesting choice is: which
    column the model reached for first.
    """
    tree = estimator.tree_
    order, nodes, index_of = [0], [], {}
    while order and len(nodes) < max_nodes:
        node = order.pop(0)
        index_of[node] = len(nodes)
        left, right = int(tree.children_left[node]), int(tree.children_right[node])
        feature = int(tree.feature[node])
        nodes.append({
            "feature": feature_names[feature] if feature >= 0 else None,
            "threshold": round(float(tree.threshold[node]), 4) if feature >= 0 else None,
            "samples": int(tree.n_node_samples[node]),
            "impurity": round(float(tree.impurity[node]), 4),
            "_left": left, "_right": right,
        })
        if left >= 0:
            order.append(left)
        if right >= 0:
            order.append(right)

    for node in nodes:
        node["left"] = index_of.get(node.pop("_left"), None)
        node["right"] = index_of.get(node.pop("_right"), None)
    return {
        "nodes": nodes,
        "depth": int(estimator.get_depth()),
        "total_nodes": int(tree.node_count),
        "root_feature": nodes[0]["feature"] if nodes else None,
    }


def root_split_census(learner, feature_names: list, sample: int = 40) -> list:
    """Which column each of the first trees split on first, counted.

    One tree is not evidence: a random forest samples features at every node, so
    the root of tree zero is partly luck. Across the first few dozen trees it
    stops being luck, and the answer is the whole argument of this tool made
    visible. An arm that still holds a leaked column reaches for it again and
    again; the same forest without it spreads across ordinary features.
    """
    trees = getattr(learner, "estimators_", None)
    if trees is None or not len(trees):
        return []
    counts: dict = {}
    for entry in list(trees)[:sample]:
        estimator = entry[0] if hasattr(entry, "__len__") else entry
        feature = int(estimator.tree_.feature[0])
        if feature < 0:
            continue
        name = feature_names[feature]
        counts[name] = counts.get(name, 0) + 1
    total = sum(counts.values()) or 1
    ranked = sorted(counts.items(), key=lambda item: -item[1])
    return [{"feature": name, "count": count, "share": round(count / total, 3)}
            for name, count in ranked[:6]]


def _first_tree(learner):
    """The first tree of a fitted ensemble, whichever ensemble it is."""
    trees = getattr(learner, "estimators_", None)
    if trees is None or not len(trees):
        return None
    first = trees[0]
    # Gradient boosting stores a 2-D array of trees, one row per boosting stage.
    return first[0] if hasattr(first, "__len__") else first


def _make_learner(learner_name: str, config: dict):
    if learner_name == "random_forest":
        return RandomForestClassifier(**config, n_jobs=-1, random_state=0)
    return GradientBoostingClassifier(**config, random_state=0)


def _build_folds(features, target_values, groups, n_splits) -> list:
    if groups is not None:
        splitter = GroupKFold(n_splits=min(n_splits, groups.nunique()))
        fold_iterator = splitter.split(features, target_values, groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        fold_iterator = splitter.split(features, target_values)
    return [(train_index, test_index) for train_index, test_index in fold_iterator]


MAX_CLASSES = 12


def _target_as_classes(target_values: pd.Series) -> tuple[pd.Series, list]:
    """Turn the target column into class indices plus the label for each index.

    Labels are sorted as text so the ordering is stable between the two arms
    and between runs. Anything with more classes than MAX_CLASSES is refused,
    because at that point the column is almost certainly a continuous
    measurement or an identifier rather than a class to predict.
    """
    distinct_values = sorted(target_values.dropna().unique(), key=str)
    if len(distinct_values) < 2:
        raise ImpactError(
            f"{target_values.name!r} has only one value, so there is nothing to predict"
        )
    if len(distinct_values) > MAX_CLASSES:
        raise ImpactError(
            f"{target_values.name!r} has {len(distinct_values)} distinct values, which is too "
            f"many to treat as classes; this tool measures classification, not regression"
        )
    index_of = {label: index for index, label in enumerate(distinct_values)}
    return target_values.map(index_of).astype(int), distinct_values


def _encode(features: pd.DataFrame) -> pd.DataFrame:
    """One fixed, tuning-free encoding used identically by both arms:
    numbers pass through with median fill, small text columns become one-hot
    indicator columns, and large text columns become integer codes."""
    encoded_parts = []
    for column in features.columns:
        values = features[column]
        if pd.api.types.is_numeric_dtype(values):
            # A column that is empty from top to bottom has a median of NaN, so
            # filling with it changes nothing and the NaN reaches the learner.
            # Random forests tolerate that; gradient boosting refuses outright,
            # and the whole measurement dies on a column carrying no
            # information at all. Real archive exports are full of these.
            fill = values.median()
            if pd.isna(fill):
                fill = 0.0
            encoded_parts.append(values.fillna(fill).to_frame(column))
        elif values.nunique() <= 20:
            indicators = pd.get_dummies(values.astype(str), prefix=column, prefix_sep="=")
            encoded_parts.append(indicators)
        else:
            codes, _ = pd.factorize(values, use_na_sentinel=True)
            encoded_parts.append(pd.Series(codes, index=values.index, name=column).to_frame())
    return pd.concat(encoded_parts, axis=1)
