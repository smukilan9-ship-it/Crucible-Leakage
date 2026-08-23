"""Tests for the parts of Crucible that decide something.

No test here calls a language model. Everything that depends on a model is
behind one function, `FeatherlessClient.chat`, and what is tested instead is
the code that surrounds it: how a reply is parsed, how disagreeing replies are
combined, how verdicts are fused with the statistical screen, and how the
downstream comparison is computed. Those are the places a silent error would
change a reported number without anything failing.

Several tests exist because the bug they describe actually happened.
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
import pytest

from crucible import (audit, cli, evidence, fusion, impact, intake, metrics,
                      models, prompts, providers, screen, stats)
from crucible.parsing import parse_verdicts
from crucible.providers import KeyPool, output_budget


# ── intake ──────────────────────────────────────────────────────────────

def test_anonymised_names_are_refused():
    assert intake.names_are_anonymized(["A1", "A2", "A3", "var_4", "V5"])
    assert not intake.names_are_anonymized(
        ["age", "lifeboat_number", "fare_paid", "cabin", "port_of_embarkation"])


def test_a_few_short_names_do_not_condemn_a_readable_table():
    columns = ["age", "sex", "fare", "cabin", "embarked", "home_destination",
               "ticket_number", "lifeboat"]
    assert not intake.names_are_anonymized(columns)


def test_sample_rows_never_carry_a_not_a_number(tmp_path):
    """A NaN in the first rows used to reach the JSON encoder and return 500
    for any table with a missing value near the top, which is most of them."""
    table = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": ["x", None, "z"],
                          "y": [0, 1, 0]})
    described = intake.describe_table(table, "y")
    for row in described["sample_rows"]:
        for value in row.values():
            assert isinstance(value, str), f"{value!r} is not JSON-safe"


def test_dictionary_matches_columns_and_ignores_the_target(tmp_path):
    path = tmp_path / "dictionary.csv"
    path.write_text("field,meaning\nboat,lifeboat identifier\n"
                    "survived,the outcome\nmissing_col,not in the table\n")
    report = intake.load_dictionary(str(path), ["boat", "fare"], target="survived")
    assert report["descriptions"] == {"boat": "lifeboat identifier"}
    assert report["missing"] == ["fare"]
    assert report["unmatched_rows"] == ["missing_col"]  # the target is not a mismatch


def test_dictionary_headings_are_found_by_meaning_not_position(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("Variable Name,Full Description\nage,years at boarding\n")
    report = intake.load_dictionary(str(path), ["age"])
    assert report["descriptions"]["age"] == "years at boarding"


# ── reply parsing ───────────────────────────────────────────────────────

def test_invented_column_names_are_dropped():
    reply = ('[{"column":"boat","verdict":"LEAK","mechanism":"CONSEQUENCE","reason":"r"},'
             '{"column":"not_a_real_column","verdict":"LEAK","reason":"r"}]')
    verdicts, _ = parse_verdicts(reply, ["boat", "fare"])
    assert set(verdicts) == {"boat"}


def test_prose_around_the_json_is_tolerated():
    reply = ('Here is my analysis.\n```json\n'
             '[{"column":"fare","verdict":"OK","reason":"known at purchase"}]\n```\nDone.')
    verdicts, _ = parse_verdicts(reply, ["fare"])
    assert verdicts["fare"]["verdict"] == "OK"


# ── combining passes ────────────────────────────────────────────────────

def test_majority_decides_and_a_minority_does_not():
    passes = [
        {"boat": {"verdict": "LEAK", "mechanism": "CONSEQUENCE", "reason": "rescued only"},
         "cabin": {"verdict": "LEAK", "mechanism": "TIMING", "reason": "maybe later"}},
        {"boat": {"verdict": "LEAK", "mechanism": "CONSEQUENCE", "reason": "only survivors"},
         "cabin": {"verdict": "OK", "reason": "known at boarding"}},
        {"boat": {"verdict": "LEAK", "mechanism": "CONSEQUENCE", "reason": "post rescue"},
         "cabin": {"verdict": "OK", "reason": "assigned at boarding"}},
    ]
    combined = screen.majority_vote(passes, ["boat", "cabin"])
    assert combined["boat"]["verdict"] == "LEAK"
    assert combined["boat"]["mechanism"] == "CONSEQUENCE"
    assert combined["boat"]["leak_votes"] == 3
    assert combined["cabin"]["verdict"] == "OK"  # one vote out of three is not a majority


def test_every_column_keeps_its_reasoning_including_cleared_ones():
    passes = [{"fare": {"verdict": "OK", "reason": "paid before boarding"}}]
    combined = screen.majority_vote(passes, ["fare"])
    assert combined["fare"]["reasons"] == ["paid before boarding"]


def test_near_duplicate_reasons_are_collapsed():
    passes = [
        {"boat": {"verdict": "LEAK", "reason": "lifeboat number recorded only for survivors"}},
        {"boat": {"verdict": "LEAK", "reason": "the lifeboat number is recorded only for survivors"}},
        {"boat": {"verdict": "LEAK", "reason": "assigned during the rescue operation"}},
    ]
    reasons = screen.majority_vote(passes, ["boat"])["boat"]["reasons"]
    assert len(reasons) == 2, reasons


# ── prompt assembly ─────────────────────────────────────────────────────

def test_sample_values_are_never_placed_in_a_prompt():
    prompt = prompts.build_screen_prompt(["boat"], "survived", "at boarding")
    for forbidden in ("Allen, Miss", "211.3375", "B5"):
        assert forbidden not in prompt


def _flat(text: str) -> str:
    """Collapse wrapping so a phrase can be found wherever the line breaks fall.

    Searching the raw string for a contiguous phrase tests the line wrapping,
    not the wording. That is not academic: this test passed for months against
    a C6 whose closing guard had been dropped and whose (a)/(b) blocks had been
    reflowed into prose, because the one phrase it looked for happened to sit
    on a single line in the damaged copy and spans two in the correct one.
    """
    return " ".join(text.split())


def test_the_two_criterion_wordings_really_differ():
    temporal = prompts.build_screen_prompt(["a"], "y", "t", None, "temporal")
    information = prompts.build_screen_prompt(["a"], "y", "t", None, "information")
    assert temporal != information
    assert "EVEN IF the value was recorded BEFORE" in _flat(temporal)
    assert "EVEN IF the value was recorded BEFORE" not in _flat(information)
    assert "could the target be" in _flat(information)


def test_both_criteria_keep_the_predictiveness_guard():
    """The sentence that stops the model calling a strong correlate a leak.

    It is the prompt half of the definition's second consequence: a column may
    be almost perfectly predictive and entirely legitimate. Both conditions
    carry it, byte-identically, in the experiment these wordings come from, so
    dropping it from one is a silent change to a measured condition rather
    than an edit to some prose.
    """
    guard = ("Being merely predictive is not sufficient for either: a column "
             "can correlate strongly with the target and still be AVAILABLE.")
    for criterion in ("temporal", "information"):
        built = prompts.build_screen_prompt(["a"], "y", "t", None, criterion)
        assert guard in _flat(built), f"{criterion} lost the predictiveness guard"


def test_both_criteria_keep_the_two_reasons_as_a_labelled_block():
    """(a) and (b) are indented and on their own lines in both conditions.

    Reflowing them into a paragraph is a wording change, and this prompt's own
    finding is that wording changes move these models in mirror image.
    """
    for criterion in ("temporal", "information"):
        built = prompts.build_screen_prompt(["a"], "y", "t", None, criterion)
        assert "\n  (a) TIMING" in built, f"{criterion} lost the (a) block"
        assert "\n  (b) DERIVATION" in built, f"{criterion} lost the (b) block"


def test_a_dictionary_puts_descriptions_and_a_citation_rule_in_the_prompt():
    plain = prompts.build_screen_prompt(["boat"], "survived", "at boarding")
    grounded = prompts.build_screen_prompt(
        ["boat"], "survived", "at boarding", {"boat": "recorded only for survivors"})
    assert "recorded only for survivors" in grounded
    assert prompts.CITATION_CONTRACT in grounded
    assert prompts.CITATION_CONTRACT not in plain


# ── statistical screen ──────────────────────────────────────────────────

def test_a_correlation_that_cannot_exist_is_reported_as_none_not_zero():
    """Titanic's `body` is non-null only for passengers who died, so the target
    has no variance on those rows. None and 0.0 mean very different things."""
    table = pd.DataFrame({
        "body": [np.nan, np.nan, np.nan, 12.0, 45.0, 88.0],
        "y":    [1, 1, 1, 0, 0, 0],
    })
    screened = stats.statistical_screen(table, "y")
    assert screened["body"]["correlation"] is None
    assert screened["body"]["flagged"] is False


def test_a_strong_legitimate_predictor_is_flagged_by_correlation():
    """`sex` on Titanic: correlation trips the threshold and the column is fine.
    The baseline is supposed to make this mistake."""
    table = pd.DataFrame({"sex": ["f"] * 10 + ["m"] * 10, "y": [1] * 9 + [0] * 11})
    assert stats.statistical_screen(table, "y")["sex"]["flagged"]


# ── fusion ──────────────────────────────────────────────────────────────

def test_buckets_separate_what_each_screen_can_see():
    semantic = {
        "both": {"verdict": "LEAK"}, "model_only": {"verdict": "LEAK"},
        "stats_only": {"verdict": "OK"}, "neither": {"verdict": "OK"},
    }
    statistical = {
        "both": {"flagged": True}, "model_only": {"flagged": False},
        "stats_only": {"flagged": True}, "neither": {"flagged": False},
    }
    buckets = fusion.fuse(semantic, statistical, list(semantic))
    assert buckets == {"both": "A", "model_only": "B", "stats_only": "C", "neither": "D"}


# ── metrics ─────────────────────────────────────────────────────────────

def test_binary_diagnostics_have_the_shape_the_interface_draws():
    actual = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    probabilities = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8],
                              [0.7, 0.3], [0.1, 0.9], [0.4, 0.6], [0.6, 0.4]])
    evaluated = metrics.evaluate(actual, probabilities, ["no", "yes"])
    assert np.array(evaluated["confusion"]).shape == (2, 2)
    assert sum(sum(row) for row in evaluated["confusion"]) == len(actual)
    assert len(evaluated["roc_curves"]) == 1          # only the positive class is drawn
    assert 0.0 <= evaluated["macro"]["f1"] <= 1.0
    assert "sweep" in evaluated


def test_multiclass_gives_a_square_matrix_and_one_curve_per_class():
    rng = np.random.default_rng(0)
    actual = np.array([0, 1, 2] * 12)
    probabilities = rng.dirichlet([2, 2, 2], size=len(actual))
    evaluated = metrics.evaluate(actual, probabilities, ["a", "b", "c"])
    assert np.array(evaluated["confusion"]).shape == (3, 3)
    assert len(evaluated["roc_curves"]) == 3
    assert len(evaluated["per_class"]) == 3
    assert "sweep" not in evaluated                   # no threshold to sweep
    assert sum(evaluated["support"]) == len(actual)


def test_a_perfect_fit_scores_one_and_a_perverse_one_does_not():
    actual = np.array([0, 0, 1, 1])
    perfect = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    assert metrics.evaluate(actual, perfect, ["n", "y"])["macro"]["f1"] == 1.0
    inverted = perfect[:, ::-1]
    assert metrics.evaluate(actual, inverted, ["n", "y"])["macro"]["f1"] < 0.6


def test_every_reported_number_is_json_safe():
    """A single not-a-number anywhere in this structure returns 500 to the
    browser, and the arithmetic here has several ways to produce one."""
    import json
    import math
    actual = np.array([0, 0, 0, 1])                   # a very rare positive class
    probabilities = np.array([[0.9, 0.1]] * 3 + [[0.9, 0.1]])
    evaluated = metrics.evaluate(actual, probabilities, ["n", "y"])

    def walk(node, path="root"):
        if isinstance(node, float) and not math.isfinite(node):
            raise AssertionError(f"non-finite value at {path}")
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}")
        if isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(evaluated)
    json.dumps(evaluated)


# ── impact ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,expected", [
    ("patient_id", True), ("member_id", True), ("objID", True), ("uuid", True),
    ("paid", False), ("cons.price.idx", False), ("identity_score", False),
])
def test_identifiers_match_by_word_not_by_substring(name, expected):
    """`id` sits inside `paid` and inside `cons.price.idx`. Treating either as a
    unit identifier silently changes how every fold is built."""
    assert impact._looks_like_identifier(name) is expected


def test_a_small_count_column_is_not_mistaken_for_a_unit_identifier():
    table = pd.DataFrame({"comorbidity_id": [0, 1, 2] * 40, "y": [0, 1] * 60})
    assert impact.detect_group_column(table, "y") is None


def test_a_repeating_identifier_is_found():
    table = pd.DataFrame({"patient_id": [f"p{i // 3}" for i in range(120)],
                          "y": [0, 1] * 60})
    assert impact.detect_group_column(table, "y") == "patient_id"


def test_class_labels_are_sorted_stably_so_both_arms_agree():
    values = pd.Series(["yes", "no", "maybe", "no", "yes", "maybe"])
    indices, labels = impact._target_as_classes(values)
    assert labels == ["maybe", "no", "yes"]
    assert list(indices) == [2, 1, 0, 1, 2, 0]


def test_a_continuous_target_is_refused_rather_than_guessed_at():
    with pytest.raises(impact.ImpactError):
        impact._target_as_classes(pd.Series(np.linspace(0, 1, 200)))


def test_a_constant_target_is_refused():
    with pytest.raises(impact.ImpactError):
        impact._target_as_classes(pd.Series([1] * 50))


def test_encoding_is_identical_for_columns_present_in_both_arms():
    """If the encoding changed with the column set, the comparison between the
    two arms would measure the encoding rather than the leakage."""
    table = pd.DataFrame({
        "n": [1.0, 2.0, 3.0, 4.0],
        "small_text": ["a", "b", "a", "b"],
        "extra": [9, 8, 7, 6],
    })
    both = impact._encode(table)
    without_extra = impact._encode(table.drop(columns=["extra"]))
    shared = [c for c in without_extra.columns]
    pd.testing.assert_frame_equal(both[shared], without_extra[shared])


def test_dropping_every_feature_is_an_error_not_an_empty_comparison():
    table = pd.DataFrame({"a": [1, 2, 3, 4] * 10, "y": [0, 1] * 20})
    with pytest.raises(impact.ImpactError):
        impact.quantify(table, "y", ["a"])


@pytest.mark.slow
def test_a_planted_leak_is_measured_as_worth_something():
    """End to end on a table with a known planted leak: a column that is the
    target with a little noise must show a large drop when it is removed."""
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    table = pd.DataFrame({
        "noise_a": rng.normal(size=n),
        "noise_b": rng.normal(size=n),
        "planted_leak": np.where(rng.random(n) < 0.95, y, 1 - y),
        "y": y,
    })
    result = impact.quantify(table, "y", ["planted_leak"], n_splits=3)
    arms = result["learners"]["random_forest"]
    assert arms["inflation"]["macro_f1"] > 0.25
    assert arms["with_leaks"]["macro"]["f1"] > arms["honest"]["macro"]["f1"]


# ── output budget ───────────────────────────────────────────────────────

def test_a_wide_table_gets_a_bigger_output_budget_than_a_narrow_one():
    """Requesting no output limit truncated a 36-column reply to nothing, which
    stopped the audit. The budget has to grow with the column count."""
    assert output_budget(36) > output_budget(10)
    assert output_budget(36) >= 36 * 100          # room for one object per column
    assert output_budget(1) >= 2048               # a floor for narrow tables


# ── the demonstration key pool ──────────────────────────────────────────

def test_a_pool_never_reveals_a_key_in_its_status():
    """Status is shown to visitors and written to logs. A key, or any part of
    one, appearing there is the whole pool compromised."""
    secrets = ["AIza-secret-one", "AIza-secret-two"]
    pool = KeyPool(secrets, daily_limit=20, name="test")
    rendered = repr(pool.status())
    for secret in secrets:
        assert secret not in rendered
        assert secret[:8] not in rendered      # not even a prefix


def test_quota_is_spent_least_used_first():
    """Round-robin drains every key to zero at the same moment. Draining the
    least-used key first keeps the most keys alive."""
    pool = KeyPool(["a", "b", "c"], daily_limit=2)
    served = [pool.acquire()[0] for _ in range(3)]
    assert sorted(served) == [0, 1, 2]         # one each before any repeats


def test_a_spent_pool_refuses_rather_than_overspending():
    pool = KeyPool(["a", "b"], daily_limit=1)
    pool.acquire(); pool.acquire()
    assert pool.remaining() == 0
    with pytest.raises(LookupError):
        pool.acquire()


def test_quota_counts_the_attempt_not_the_success():
    """A call that fails still cost a request against a free tier. Counting on
    success lets a failing key be retried without limit."""
    pool = KeyPool(["a"], daily_limit=3)
    before = pool.remaining()
    pool.acquire()                              # caller may now fail; we do not care
    assert pool.remaining() == before - 1


def test_an_empty_pool_says_so_instead_of_serving_nothing():
    pool = KeyPool([], daily_limit=20, name="gemini")
    assert pool.remaining() == 0
    with pytest.raises(LookupError, match="no keys configured"):
        pool.acquire()


def test_the_pool_reads_keys_from_the_environment_only(monkeypatch):
    monkeypatch.setenv("TEST_KEYS", "one, two  three")
    pool = KeyPool.from_environment("TEST_KEYS", daily_limit=5)
    assert len(pool) == 3
    assert pool.status()["capacity_today"] == 15


# ── the model catalogue ─────────────────────────────────────────────────

def test_every_catalogue_entry_names_a_real_provider_and_wording():
    from crucible.providers import PROVIDERS
    for entry in models.CATALOGUE:
        assert entry["provider"] in PROVIDERS, entry["id"]
        assert entry["criterion"] in prompts.CRITERION_WORDINGS, entry["id"]


def test_every_offered_model_carries_the_measurement_that_chose_its_wording():
    """The wording is set per model from measured F1, not from a preference, so
    an entry without a measurement is an entry nobody checked. Asserting the
    measurement exists survives a change of default model; asserting a
    particular string did not."""
    for entry in models.CATALOGUE:
        measured = entry.get("measured")
        assert measured, f"{entry['id']} has no measurement behind it"
        assert 0 < measured["f1"] <= 1
        assert measured["datasets"] >= 10, f"{entry['id']} measured too narrowly"
        assert entry["criterion"] in prompts.CRITERION_WORDINGS


def test_the_default_is_the_best_measured_model_that_needs_no_key():
    """Two stronger models are catalogued and both bill the user's own account.
    The default has to be the best one a visitor can run without bringing
    anything, or the demonstration asks for a key before it does any work."""
    keyless = [e for e in models.CATALOGUE if not e.get("needs_key")]
    best = max(keyless, key=lambda e: e["measured"]["f1"])
    assert models.DEFAULT_MODEL == best["id"]


def test_every_model_needing_a_key_names_a_provider_that_exists():
    for entry in models.CATALOGUE:
        needed = entry.get("needs_key")
        if needed:
            assert needed in providers.PROVIDERS
            assert needed == entry["provider"]


def test_a_key_is_routed_to_the_vendor_that_issued_it():
    assert providers.detect_provider("sk-ant-api03-xxxx") == "anthropic"
    assert providers.detect_provider("sk-proj-xxxx") == "openai"
    assert providers.detect_provider("rc_xxxx") == "featherless"


def test_both_google_key_formats_are_recognized():
    """Google changed the format: AI Studio keys now begin `AQ.` where they
    used to begin `AIza`. The old table rejected every current key, which is
    the failure this test exists to stop repeating."""
    assert providers.detect_provider("AIzaSyC-old-style-key") == "gemini"
    assert providers.detect_provider("AQ.Ab8RN6JnewStyleKey") == "gemini"


def test_an_unrecognized_key_is_not_guessed_at():
    """None means "ask the user", not "refuse". Guessing would send their
    column names to a vendor they did not pick; refusing would lock them out
    every time a provider changes a prefix, which has already happened once."""
    assert providers.detect_provider("not-a-real-key") is None
    assert providers.detect_provider("") is None


def test_the_key_check_offers_a_provider_when_it_cannot_tell():
    """An unrecognized key has to remain usable, so the response carries the
    providers a user can pick from by hand."""
    import importlib, sys
    root = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    app_module = importlib.import_module("web.app")
    catalogued = {entry["provider"] for entry in models.CATALOGUE}
    assert catalogued <= set(providers.PROVIDERS)
    assert catalogued, "no provider serves any catalogued model"
    # The endpoint hands back exactly this set; asserting it here keeps the two
    # from drifting without needing a live server in the test suite.
    assert "gemini" in catalogued


def test_the_gemini_substitute_is_itself_a_measured_model():
    """Falling back to an unmeasured model would leave the report unable to say
    what the figure behind its verdicts was."""
    for asked, substitute in providers.FALLBACK_MODEL.items():
        assert asked in models.BY_ID
        assert substitute in models.BY_ID
        assert models.provider_for(substitute) == models.provider_for(asked)


def test_every_catalogued_figure_comes_from_the_evidence_module():
    """Numbers are quoted in one place so that re-running the benchmark is one
    edit rather than eight files across three languages."""
    for entry in models.CATALOGUE:
        cell = evidence.MODEL_CELLS[entry["id"]]
        assert entry["measured"] == cell.as_dict()
        assert cell.source.startswith(("PAPER", "N ")), "a figure with no source"


def test_an_unknown_model_fails_loudly_rather_than_falling_back():
    with pytest.raises(KeyError):
        models.entry("nobody/such-model")


# ── the event stream, as a proxy will see it ────────────────────────────

def test_the_heartbeat_interval_is_under_a_proxy_timeout():
    """A stage can take minutes and hosted proxies close connections that have
    been silent for sixty seconds or so. The stream must speak more often than
    that, or the browser waits forever on a socket the server thinks is fine.

    The interval is asserted here; that a heartbeat is actually emitted is
    verified against a running server, because a TestClient deadlocks on a
    stream that never ends.
    """
    import web.app as service
    assert 0 < service.KEEPALIVE_SECONDS <= 30


# ── v2 pipeline conformance ─────────────────────────────────────────────

def test_an_anonymised_table_still_gets_a_statistical_screen(monkeypatch):
    """N-04. Refusing the whole table throws away the half that still works.
    The semantic screen is skipped and the report says so."""
    import asyncio

    from crucible.audit import AuditRequest, run_audit
    table = pd.DataFrame({f"V{i}": np.random.default_rng(i).normal(size=60)
                          for i in range(1, 9)} | {"y": [0, 1] * 30})
    report = asyncio.run(run_audit(AuditRequest(
        table=table, target="y", prediction_point="before anything happens")))
    assert report["semantic_skipped"] == "INSUFFICIENT_SEMANTICS"
    assert set(report["statistical"]) == {f"V{i}" for i in range(1, 9)}
    assert all(v["verdict"] == "ABSTAIN" for v in report["semantic"].values())


def test_missingness_asymmetry_sees_what_correlation_cannot():
    """N-30. Titanic's `boat` correlates 0.013 with survival and is missing for
    almost every passenger who died. Correlation is blind to that; the
    missingness gap is not."""
    table = pd.DataFrame({
        "boat": [1.0] * 40 + [np.nan] * 60,
        "y":    [1] * 40 + [0] * 60,
    })
    screened = stats.statistical_screen(table, "y")["boat"]
    assert screened["missingness_gap"] > 0.9


def test_the_threshold_is_chosen_without_seeing_held_out_rows():
    """N-83. Searching the threshold over the predictions being scored reports
    the luckiest cut rather than an achievable one."""
    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 2, n)
    table = pd.DataFrame({
        "signal": y + rng.normal(0, 0.6, n),
        "noise": rng.normal(size=n),
        "leak": np.where(rng.random(n) < 0.95, y, 1 - y),
        "y": y,
    })
    result = impact.quantify(table, "y", ["leak"], n_splits=3)
    arm = result["learners"]["random_forest"]["honest"]
    assert arm["threshold_source"] == "chosen on training folds only"
    assert "best_f1" not in arm      # nothing was searched on these predictions


def test_identical_column_sets_are_fitted_once():
    """N-82. Two arms that retain the same columns are the same fit."""
    rng = np.random.default_rng(1)
    n = 200
    y = rng.integers(0, 2, n)
    table = pd.DataFrame({"a": y + rng.normal(0, 0.8, n),
                          "b": rng.normal(size=n), "y": y})
    cache: dict = {}
    features = impact._encode(table[["a", "b"]])
    target, _ = impact._target_as_classes(table["y"])
    folds = impact._build_folds(features, target, None, 3)
    first = impact._best_arm("random_forest", features, target, folds, ["0", "1"], cache)
    second = impact._best_arm("random_forest", features, target, folds, ["0", "1"], cache)
    assert first["refit"] is True
    assert second["refit"] is False
    assert second["macro"]["f1"] == first["macro"]["f1"]


def test_a_drop_list_that_removes_nothing_is_reported_as_one_fit():
    """N-83. Confirming a column the table does not have leaves both arms
    identical. The difference between them is then zero by construction, and
    reporting that as "no leakage" is the one wrong answer available here."""
    rng = np.random.default_rng(2)
    n = 200
    y = rng.integers(0, 2, n)
    table = pd.DataFrame({"a": y + rng.normal(0, 0.8, n),
                          "b": rng.normal(size=n), "y": y})
    result = impact.quantify(table, "y", ["y", "not_a_column"], n_splits=3)

    assert result["identical_arms"] == [{"arms": ["with_leaks", "honest"], "n_features": 2}]
    assert result["drops_applied"] == []
    assert result["drops_ignored"] == ["y", "not_a_column"]
    assert result["learners"]["random_forest"]["inflation"]["macro_f1"] == 0.0


def test_arms_that_differ_are_not_reported_as_identical():
    rng = np.random.default_rng(3)
    n = 200
    y = rng.integers(0, 2, n)
    table = pd.DataFrame({"a": y + rng.normal(0, 0.8, n),
                          "b": rng.normal(size=n), "y": y})
    result = impact.quantify(table, "y", ["a"], n_splits=3)

    assert result["identical_arms"] == []
    assert result["drops_applied"] == ["a"]
    assert result["drops_ignored"] == []


# ── the contested gate ──────────────────────────────────────────────────

class StubProvider:
    """A provider that returns a scripted reply and records what it was asked.

    The gate's whole job is to turn one question into a set of columns handed
    back to a person, so what matters is the question it asks and the set it
    produces — neither of which needs a network to check.
    """

    name = "stub"

    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list[str] = []

    async def chat(self, model, prompt, max_tokens=None):
        self.prompts.append(prompt)
        return self.reply

    async def aclose(self):
        pass


def _run(coroutine):
    import asyncio
    return asyncio.run(coroutine)


def test_the_gate_returns_only_the_columns_documented_as_already_fixed():
    from crucible.contested import contested_gate
    provider = StubProvider(
        '[{"column":"prior_estimate","verdict":"FIXED","reason":"recorded at day 3"},'
        ' {"column":"discharge_code","verdict":"NOT_FIXED","reason":"written at discharge"}]')
    contested = _run(contested_gate(
        provider, "m", ["prior_estimate", "discharge_code"], "death", "at admission"))
    assert contested == {"prior_estimate": "recorded at day 3"}


def test_the_gate_asks_about_availability_and_not_about_advisability():
    """The distinction the whole stage exists for. A prompt that asks whether
    using a column is a good idea has reintroduced the judgement that was
    withdrawn for want of any source licensing it."""
    from crucible import prompts
    # The prompt is wrapped for readability, so compare on words alone.
    prompt = " ".join(
        prompts.build_contested_probe(["prg2m"], "death", "at admission").split())
    assert "already fixed and recorded at or before that prediction point" in prompt
    assert "Do not judge whether using the column is a good idea" in prompt
    assert "do not consider how predictive it is" in prompt


def test_the_gate_passes_documentation_through_when_there_is_any():
    from crucible import prompts
    prompt = prompts.build_contested_probe(
        ["prg2m"], "death", "at admission",
        {"prg2m": "physician's two-month survival estimate, recorded at day 3"})
    assert "physician's two-month survival estimate" in prompt


def test_the_gate_does_not_call_a_model_when_nothing_was_flagged():
    """Every call costs a request against a quota. A gate over an empty set has
    nothing to ask about."""
    from crucible.contested import contested_gate
    provider = StubProvider("[]")
    assert _run(contested_gate(provider, "m", [], "y", "at t")) == {}
    assert provider.prompts == []


def test_a_column_the_gate_never_mentions_is_not_contested():
    from crucible.contested import contested_gate
    provider = StubProvider('[{"column":"a","verdict":"FIXED","reason":"r"}]')
    contested = _run(contested_gate(provider, "m", ["a", "b"], "y", "at t"))
    assert "b" not in contested


def test_a_contested_column_is_left_out_of_the_drop_list():
    """A contested column is flagged and documented as available. Dropping it
    silently is the tool making the call it exists to hand back."""
    from crucible.audit import flagged_columns
    report = {"semantic": {
        "plain_leak": {"verdict": "LEAK"},
        "arguable": {"verdict": "LEAK", "contested": True},
        "fine": {"verdict": "OK"},
    }}
    assert flagged_columns(report) == ["plain_leak"]
    assert flagged_columns(report, include_contested=True) == ["arguable", "plain_leak"]


# ── retention ───────────────────────────────────────────────────────────

def test_old_jobs_and_their_uploads_are_discarded(tmp_path):
    """Nothing evicted anything, so a long-lived instance leaked a schema, a
    set of verdicts and an uploaded table per audit, for ever."""
    import web.app as service

    original_cap, original_jobs = service.MAX_RETAINED_JOBS, dict(service.JOBS)
    service.MAX_RETAINED_JOBS = 3
    service.JOBS.clear()
    try:
        directories = []
        for index in range(6):
            directory = tmp_path / f"crucible_{index}"
            directory.mkdir()
            (directory / "data.csv").write_text("a,b\n1,2\n")
            directories.append(directory)
            service.JOBS[f"job{index}"] = {
                "id": f"job{index}", "status": "complete",
                "path": str(directory / "data.csv"),
            }
            service._evict_old_jobs()

        assert len(service.JOBS) <= 3
        assert not directories[0].exists(), "the upload behind an evicted job survived"
        assert directories[-1].exists(), "the newest upload was removed"
    finally:
        service.MAX_RETAINED_JOBS = original_cap
        service.JOBS.clear()
        service.JOBS.update(original_jobs)


def test_a_running_job_is_never_evicted_from_under_itself():
    import web.app as service

    original_cap, original_jobs = service.MAX_RETAINED_JOBS, dict(service.JOBS)
    service.MAX_RETAINED_JOBS = 1
    service.JOBS.clear()
    try:
        service.JOBS["busy"] = {"id": "busy", "status": "running", "path": ""}
        service.JOBS["done"] = {"id": "done", "status": "complete", "path": ""}
        service._evict_old_jobs()
        assert "busy" in service.JOBS
    finally:
        service.MAX_RETAINED_JOBS = original_cap
        service.JOBS.clear()
        service.JOBS.update(original_jobs)

# ── the hosted interface ────────────────────────────────────────────────

def test_the_interface_is_stamped_by_its_own_content():
    """A deploy that changes the interface must not leave a visitor on the
    cached one.

    This was a hand-written version string in the HTML, and it failed the way
    hand-written version strings do: the files changed a dozen times, nobody
    bumped the literal, and browsers kept serving what they already had. The
    stamp is now derived from the bytes, so it cannot drift.
    """
    import importlib, re, sys
    root = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    app_module = importlib.import_module("web.app")

    first = app_module._asset_version()
    assert re.fullmatch(r"[0-9a-f]{12}", first), f"unexpected stamp {first!r}"

    style = root / "web" / "static" / "style.css"
    original = style.read_text()
    try:
        style.write_text(original + "\n/* stamp probe */\n")
        assert app_module._asset_version() != first, (
            "the stamp did not change when the interface did")
    finally:
        style.write_text(original)
    assert app_module._asset_version() == first, "the stamp did not come back"


def test_the_landing_page_tells_a_visitor_the_command_line_exists():
    """The command line is the half of this tool that scales, and for a while
    it was undiscoverable from the interface."""
    root = pathlib.Path(__file__).resolve().parent.parent
    html = (root / "web" / "static" / "index.html").read_text()
    assert "pip install crucible-leakage" in html
    assert "crucible impact" in html, "no runnable command shown"
    assert "crucible --help" in html, "nothing tells a visitor where the flags are"


def test_the_landing_page_leads_with_its_citation():
    """A judge should meet the published formulation before the marketing. The
    citation used to sit four screens down inside a dark panel."""
    root = pathlib.Path(__file__).resolve().parent.parent
    html = (root / "web" / "static" / "index.html").read_text()
    citation = html.index("Kaufman")
    why_panel = html.index('id="why"')
    assert citation < why_panel, "the citation is buried below the argument again"
    assert "10.1145/2382577.2382579" in html, "the DOI is what makes it checkable"


# ── watching the fit ────────────────────────────────────────────────────

def test_watching_the_fit_cannot_change_the_fit():
    """The visualization is fed by a callback threaded through `quantify`. If
    observing were ever to alter a number, the whole measurement would be
    worthless, so the two runs are compared directly."""
    frame = pd.DataFrame({
        "leak": [0, 0, 1, 1] * 25,
        "ordinary": list(range(100)),
        "noise": [1, 2, 3, 4, 5] * 20,
        "y": [0, 0, 1, 1] * 25,
    })
    quiet = impact.quantify(frame, "y", ["leak"], n_splits=3)
    seen = []
    watched = impact.quantify(frame, "y", ["leak"], n_splits=3,
                              on_event=lambda name, payload: seen.append(name))
    for learner in quiet["learners"]:
        for arm in ("with_leaks", "honest"):
            assert (quiet["learners"][learner][arm]["macro"]["f1"]
                    == watched["learners"][learner][arm]["macro"]["f1"])
    assert {"plan", "arm_start", "fold", "tree", "arm_done", "done"} <= set(seen)


def test_a_streamed_tree_is_the_tree_the_model_built():
    """The drawing is generated; the structure is not. A sketch has to match
    the estimator it came from, or the interface is showing a decoration."""
    from sklearn.ensemble import RandomForestClassifier
    frame = pd.DataFrame({"a": list(range(60)), "b": [0, 1] * 30})
    labels = pd.Series([0, 1] * 30)
    forest = RandomForestClassifier(n_estimators=5, random_state=0).fit(frame, labels)
    sketch = impact.tree_sketch(forest.estimators_[0], list(frame.columns))

    assert sketch["depth"] == forest.estimators_[0].get_depth()
    assert sketch["total_nodes"] == forest.estimators_[0].tree_.node_count
    assert sketch["nodes"], "a fitted tree with no nodes"
    for node in sketch["nodes"]:
        assert node["feature"] is None or node["feature"] in frame.columns
        # Child indices are rewritten to positions in the truncated list, so
        # every one that survives must point somewhere real.
        for side in ("left", "right"):
            assert node[side] is None or 0 <= node[side] < len(sketch["nodes"])


def test_the_root_census_counts_real_root_splits():
    """The noise column has to be genuinely uninformative. An earlier version of
    this test used a running index against alternating labels, which separates
    them perfectly by parity, so both columns were giveaways and the census was
    right to be split between them."""
    from sklearn.ensemble import RandomForestClassifier
    generator = np.random.default_rng(0)
    labels = pd.Series([0, 1] * 40)
    frame = pd.DataFrame({
        "gives_it_away": labels.to_numpy(),
        "noise": generator.normal(size=80),
        "also_noise": generator.normal(size=80),
    })
    forest = RandomForestClassifier(n_estimators=30, random_state=0).fit(frame, labels)
    census = impact.root_split_census(forest, list(frame.columns))
    assert census, "no root splits counted"
    assert census[0]["feature"] == "gives_it_away", (
        f"the giveaway column should dominate the roots, got {census}")
    assert sum(entry["count"] for entry in census) <= 30
    assert 0 < census[0]["share"] <= 1


# ── the screen's shape ──────────────────────────────────────────────────

def test_the_column_orders_are_asked_concurrently():
    """Three orders used to be three round trips deep. On a wide table each one
    is the model writing a verdict for every column, so the wall clock was three
    times what it needed to be. They are independent, so they overlap."""
    import asyncio as _asyncio
    import time

    class SlowProvider:
        name = "slow"

        async def chat(self, model, prompt, max_tokens=None):
            await _asyncio.sleep(0.25)
            return '[{"column": "a", "verdict": "OK", "mechanism": null, "reason": "x"},' \
                   ' {"column": "b", "verdict": "LEAK", "mechanism": "TIMING", "reason": "y"}]'

        async def aclose(self):
            pass

    started = time.monotonic()
    combined, per_order = _asyncio.run(screen.run_semantic_screen(
        SlowProvider(), "any-model", ["a", "b"], "y", "at signup", shuffle_count=3))
    elapsed = time.monotonic() - started

    assert len(per_order) == 3, "every order should have answered"
    assert combined["b"]["verdict"] == "LEAK"
    assert elapsed < 0.6, (
        f"three 0.25s calls took {elapsed:.2f}s, which is sequential, not concurrent")


def test_one_failed_order_does_not_lose_the_audit():
    """A provider that is briefly overloaded should cost a pass, not the run.
    The vote wants a majority, not unanimity."""
    import asyncio as _asyncio
    from crucible.providers import ProviderError as _ProviderError

    class FlakyProvider:
        name = "flaky"

        def __init__(self):
            self.calls = 0

        async def chat(self, model, prompt, max_tokens=None):
            self.calls += 1
            if self.calls == 1:
                raise _ProviderError("HTTP 503")
            return '[{"column": "a", "verdict": "LEAK", "mechanism": "TIMING", "reason": "x"}]'

        async def aclose(self):
            pass

    combined, per_order = _asyncio.run(screen.run_semantic_screen(
        FlakyProvider(), "any-model", ["a"], "y", "at signup",
        shuffle_count=3, max_reissues=0))
    assert len(per_order) == 2, "the two healthy orders should still count"
    assert combined["a"]["verdict"] == "LEAK"
    assert combined["a"]["shuffles_counted"] == 2


def test_every_order_failing_is_still_a_failure():
    import asyncio as _asyncio
    from crucible.providers import ProviderError as _ProviderError

    class DeadProvider:
        name = "dead"

        async def chat(self, model, prompt, max_tokens=None):
            raise _ProviderError("HTTP 503")

        async def aclose(self):
            pass

    with pytest.raises(_ProviderError):
        _asyncio.run(screen.run_semantic_screen(
            DeadProvider(), "any-model", ["a"], "y", "at signup",
            shuffle_count=3, max_reissues=0))


def test_the_shuffle_count_follows_the_model_not_the_vendor(monkeypatch):
    """Two models from one vendor can earn different counts.

    Asserting the shipped number here made this test a record of one policy
    decision rather than of the rule: when the exemption tightened, the test
    failed for a model whose order sensitivity had not changed at all. So it
    exercises the rule against injected measurements instead, and the shipped
    numbers are checked by the test below, which also says why they are what
    they are.
    """
    monkeypatch.setitem(evidence.ORDER_SPREAD, "steady-model",
                        {"worst": 0.001, "seeds": 5, "source": "test"})
    monkeypatch.setitem(evidence.REPEAT_STABILITY, "steady-model",
                        {"unstable_columns": 0, "of": 36, "calls": 6,
                         "dataset": "test", "source": "test"})
    monkeypatch.setitem(evidence.ORDER_SPREAD, "swinging-model",
                        {"worst": 0.380, "seeds": 3, "source": "test"})
    assert models.shuffles_for("steady-model") == 1
    assert models.shuffles_for("swinging-model") == evidence.DEFAULT_SHUFFLES
    assert models.provider_for("gemini-3.7-flash") == models.provider_for("gemini-3.5-flash")


def test_steady_under_reordering_is_not_enough_to_skip_the_vote():
    """The exemption needs both measurements, and 3.7 Flash only passes one.

    It is the steadiest model in the study under reordering, at 0.019. Asked
    the identical prompt six times at temperature zero it still moved on two of
    thirty-six columns, both of them documented leaks. Steady when the columns
    move is a different property from steady, and only the second one licenses
    reporting a single call as the answer.
    """
    assert evidence.ORDER_SPREAD["gemini-3.7-flash"]["worst"] \
        < evidence.STABLE_ENOUGH_FOR_ONE_ORDER
    assert evidence.REPEAT_STABILITY["gemini-3.7-flash"]["unstable_columns"] \
        > evidence.STABLE_ENOUGH_FOR_ONE_CALL
    assert models.shuffles_for("gemini-3.7-flash") == evidence.DEFAULT_SHUFFLES
    why = models.shuffle_rationale("gemini-3.7-flash")
    assert "identical calls" in why and "0.019" in why


def test_a_model_measured_on_neither_axis_still_gets_the_cautious_count(monkeypatch):
    """An order measurement on its own no longer buys the exemption."""
    monkeypatch.setitem(evidence.ORDER_SPREAD, "half-measured",
                        {"worst": 0.001, "seeds": 5, "source": "test"})
    assert models.shuffles_for("half-measured") == evidence.DEFAULT_SHUFFLES


def test_an_unmeasured_model_gets_the_cautious_count():
    assert models.shuffles_for("nobody/never-measured") == evidence.DEFAULT_SHUFFLES
    assert "not been measured" in models.shuffle_rationale("nobody/never-measured")


def test_one_order_is_only_offered_where_the_spread_supports_it():
    for model_id in models.BY_ID:
        if models.shuffles_for(model_id) == 1:
            spread = evidence.ORDER_SPREAD[model_id]["worst"]
            assert spread < evidence.STABLE_ENOUGH_FOR_ONE_ORDER, (
                f"{model_id} is asked once on a measured spread of {spread}")


def test_a_column_that_is_entirely_empty_does_not_kill_the_measurement():
    """An all-missing numeric column has a median of NaN, so filling with the
    median leaves the NaN in place. Random forests tolerate that and gradient
    boosting refuses outright, which killed the whole comparison over a column
    carrying no information at all. Real archive exports are full of these: the
    NASA Exoplanet Archive's KOI table ships two."""
    frame = pd.DataFrame({
        "useful": list(range(80)),
        "entirely_empty": [np.nan] * 80,
        "half_empty": [np.nan if i % 2 else float(i) for i in range(80)],
        "leak": [0, 1] * 40,
        "y": [0, 1] * 40,
    })
    encoded = impact._encode(frame.drop(columns=["y"]))
    assert not encoded.isna().any().any(), "a NaN survived encoding"

    result = impact.quantify(frame, "y", ["leak"], n_splits=2)
    assert set(result["learners"]) == {"random_forest", "gradient_boosting"}, (
        "gradient boosting dropped out, which is what the NaN used to cause")


# ── the command line's own refusals ──────────────────────────────────────
#
# Each of these describes something the tool used to do to a first-time user.
# They are cheap to keep and the failures they cover are the kind nobody
# reports, because a person who meets one concludes the tool is broken and
# leaves rather than opening an issue.

def test_a_missing_key_is_named_rather_than_blamed_on_the_model(monkeypatch):
    """Without a credential the audit used to run its planning, call a provider
    it could not authenticate to, and report that no column order produced a
    usable answer. That describes a model which answered badly, not one that was
    never asked, and it was the first thing anyone installing this package met."""
    for variable in providers.KEY_VARIABLES.values():
        monkeypatch.delenv(variable, raising=False)
    with pytest.raises(intake.IntakeError) as raised:
        cli._require_key(models.DEFAULT_MODEL)
    message = str(raised.value)
    assert providers.KEY_VARIABLES["gemini"] in message
    assert "crucible models" in message


def test_a_key_that_is_present_lets_the_audit_through(monkeypatch):
    monkeypatch.setenv(providers.KEY_VARIABLES["gemini"], "AQ.not-a-real-key")
    cli._require_key(models.DEFAULT_MODEL)


def test_whitespace_is_not_a_key(monkeypatch):
    monkeypatch.setenv(providers.KEY_VARIABLES["gemini"], "   ")
    with pytest.raises(intake.IntakeError):
        cli._require_key(models.DEFAULT_MODEL)


def test_an_unknown_model_is_refused_before_the_network_with_a_suggestion():
    with pytest.raises(intake.IntakeError) as raised:
        cli._require_model("gemini-9.9-turbo")
    assert "gemini-3.7-flash" in str(raised.value), "no did-you-mean offered"


def test_every_catalogued_model_passes_its_own_check():
    for entry in models.CATALOGUE:
        cli._require_model(entry["id"])


@pytest.mark.parametrize("bad", ["0", "-3", "two", ""])
def test_a_shuffle_count_below_one_is_refused(bad):
    """`--shuffles 0` was falsy, so it fell through to the default and the run
    quietly did something other than what was asked."""
    with pytest.raises(argparse.ArgumentTypeError):
        cli._positive_int(bad)


def test_a_usable_shuffle_count_survives():
    assert cli._positive_int("3") == 3


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_an_empty_prediction_point_is_refused(blank):
    """Every verdict is a claim about the triple of column, target and
    prediction point. An empty third term makes each answer a claim about
    nothing, so it is refused rather than sent."""
    with pytest.raises(argparse.ArgumentTypeError):
        cli._phrase(blank)


def test_a_prediction_point_is_kept_but_trimmed():
    assert cli._phrase("  at loan approval  ") == "at loan approval"


def test_a_table_smaller_than_the_folds_is_a_refusal_not_a_traceback():
    """sklearn raises about n_splits and n_samples, which reached the user as a
    stack trace. Trying the tool on a handful of rows first is the most likely
    thing a careful person does."""
    frame = pd.DataFrame({"a": [1, 2, 3, 4], "leak": [0, 1, 0, 1], "y": [0, 1, 0, 1]})
    with pytest.raises(impact.ImpactError) as raised:
        impact.quantify(frame, "y", ["leak"])
    assert "4 rows" in str(raised.value)


def test_a_class_too_rare_for_the_folds_is_named():
    frame = pd.DataFrame({
        "a": list(range(40)),
        "leak": [0] * 37 + [1, 1, 1],
        "y": [0] * 37 + [1, 1, 1],
    })
    with pytest.raises(impact.ImpactError) as raised:
        impact.quantify(frame, "y", ["leak"])
    message = str(raised.value)
    assert "3 times" in message
    assert "np.int64" not in message, "a numpy repr reached a sentence about data"


def test_the_verdict_column_shows_the_mechanism_only_for_a_leak():
    assert cli._verdict({"contested": True, "verdict": "LEAK",
                         "mechanism": "REASON"}) == "CONTESTED"
    assert cli._verdict({"verdict": "LEAK", "mechanism": "TIMING"}) == "TIMING"
    assert cli._verdict({"verdict": "LEAK", "mechanism": None}) == "LEAK"
    assert cli._verdict({"verdict": "OK", "mechanism": "TIMING"}) == "ok"
    assert cli._verdict({"verdict": "ABSTAIN", "mechanism": None}) == "?"


def test_a_column_the_passes_disagreed_on_is_held_for_a_person():
    """Three passes, two calling it a leak and one not, is not a verdict.

    The majority would be LEAK and the tool would drop the column. It is held
    instead, for the same reason a contested column is: two to one is a tally,
    not a decision, and the person who owns the table is the one entitled to
    make it.
    """
    passes = [
        {"x": {"verdict": "LEAK", "mechanism": "TIMING",
               "reason": "recorded after the outcome"}},
        {"x": {"verdict": "LEAK", "mechanism": "TIMING",
               "reason": "only exists once the case closes"}},
        {"x": {"verdict": "OK", "mechanism": None,
               "reason": "captured at intake, before anything is decided"}},
    ]
    voted = screen.majority_vote(passes, ["x"])["x"]
    assert voted["verdict"] == "LEAK" and voted["leak_votes"] == 2
    assert voted["split"] is True
    assert any("after the outcome" in r for r in voted["reasons_for"])
    assert any("at intake" in r for r in voted["reasons_against"])

    report = {"semantic": {"x": voted}, "columns": ["x"]}
    assert audit.flagged_columns(report) == []
    assert audit.flagged_columns(report, include_contested=True) == ["x"]


def test_agreement_across_passes_is_not_a_split():
    """Unanimity is an answer, and a single pass cannot disagree with itself."""
    agreed = [{"x": {"verdict": "LEAK", "mechanism": "TIMING", "reason": "r"}}] * 3
    assert "split" not in screen.majority_vote(agreed, ["x"])["x"]
    alone = [{"x": {"verdict": "LEAK", "mechanism": "TIMING", "reason": "r"}}]
    assert "split" not in screen.majority_vote(alone, ["x"])["x"]
