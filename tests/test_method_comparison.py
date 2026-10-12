"""Comparison gates are exercised on synthetic fixtures, never paper result cells."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from specrsi.statistics import compare_registered_methods, holm_adjust, paired_sign_flip


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compare_methods_script", ROOT / "scripts/compare_methods.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def fixture(*, seeds=5, baselines=("janus", "hgm", "metarsi", "flashevolve"), direction="higher"):
    registration = cli.schema_example()["registration"]
    profile = registration["profiles"][0]
    profile["baselines"] = list(baselines)
    profile["training_seeds"] = list(range(seeds))
    profile["direction"] = direction
    registration["bootstrap_resamples"] = 1000
    rows = []
    for method in ["specrsi", *baselines]:
        for training in range(seeds):
            for search in range(3):
                rows.append({"profile_id": "example", "method": method, "training_seed": training,
                             "search_seed": search, "run_id": f"{method}-{training}-{search}",
                             "status": "complete", "contract": copy.deepcopy(profile["contract"]),
                             "score": float((10 if method == "specrsi" else 0) + search), "actual_cost": 10.0})
    return registration, rows


def test_holm_preserves_original_order_and_controls_full_family():
    assert holm_adjust([.04, .001, .03]) == pytest.approx([.06, .003, .06])
    with pytest.raises(ValueError):
        holm_adjust([float("nan")])


def test_exact_sign_flip_counts_seed_pairs_and_includes_ties():
    result = paired_sign_flip([1] * 5)
    assert result["exact"] and result["assignments"] == 32
    assert result["p_value"] == 1 / 32
    assert paired_sign_flip([0] * 5)["p_value"] == 1
    assert paired_sign_flip([-1] * 5)["p_value"] == 1


def test_monte_carlo_sign_flip_is_reproducible_and_never_claims_zero_p_value():
    first = paired_sign_flip([1] * 17, resamples=1000, seed=3)
    assert first == paired_sign_flip([1] * 17, resamples=1000, seed=3)
    assert not first["exact"] and first["p_value"] >= 1 / 1001


def test_nested_searches_do_not_create_independent_training_seeds():
    registration, rows = fixture()
    result = compare_registered_methods(registration, list(reversed(rows)))
    assert result["family_size"] == 4
    assert not result["all_registered_comparisons_supported"]
    assert all(row["estimate"]["independent_units"] == 5 for row in result["comparisons"])
    assert all(row["test"]["holm_p_value"] == .125 for row in result["comparisons"])
    assert all(row["estimate"]["expected_samples_per_family_tail"] >= 20 for row in result["comparisons"])


def test_consistent_gain_with_sufficient_independent_pairs_supports_registered_family():
    registration, rows = fixture(seeds=9)
    result = compare_registered_methods(registration, rows)
    assert result["all_registered_comparisons_supported"]
    assert "not a global SOTA claim" in result["scope"]


def test_smoke_registration_cannot_be_promoted_to_inference_with_more_replicas():
    registration, rows = fixture(seeds=9)
    registration["engineering_smoke_only"] = True
    with pytest.raises(ValueError, match="smoke registrations"):
        compare_registered_methods(registration, rows)


def test_paper_dispersion_uses_sample_sd_of_seed_means_not_nested_searches():
    registration, rows = fixture(seeds=3, baselines=("baseline",))
    for row in rows:
        if row["method"] == "specrsi":
            row["score"] += 2 * row["training_seed"]
    comparison = compare_registered_methods(registration, rows)["comparisons"][0]
    assert comparison["candidate_mean"] == 13
    assert comparison["candidate_sample_sd"] == 2
    assert comparison["baseline_sample_sd"] == 0
    assert comparison["paired_difference_sample_sd"] == 2
    assert comparison["estimate"]["independent_units"] == 3


def test_lower_is_better_and_minimum_effect_are_honored():
    registration, rows = fixture(seeds=9, direction="lower")
    assert not compare_registered_methods(registration, rows)["all_registered_comparisons_supported"]
    for row in rows:
        row["score"] *= -1
    assert compare_registered_methods(registration, rows)["all_registered_comparisons_supported"]
    registration["profiles"][0]["minimum_effect"] = 11
    assert not compare_registered_methods(registration, rows)["all_registered_comparisons_supported"]


@pytest.mark.parametrize("mutation,match", [
    (lambda rows: rows.pop(), "Missing"),
    (lambda rows: rows.append(copy.deepcopy(rows[0])), "Duplicate"),
    (lambda rows: rows[0].update(status="interrupted"), "Incomplete"),
    (lambda rows: rows[0].update(score=float("nan")), "finite observed score"),
    (lambda rows: rows[0].update(actual_cost=100001), "exceeds"),
    (lambda rows: rows[0]["contract"].update(permission_band="D/H/M"), "Unmatched"),
    (lambda rows: rows[0]["contract"]["budget"].update(unit="gpu_seconds"), "Unmatched"),
    (lambda rows: rows[0].update(search_seed=99), "Unregistered"),
    (lambda rows: rows[0].update(run_id=rows[1]["run_id"]), "unique"),
])
def test_invalid_or_selective_comparisons_are_rejected(mutation, match):
    registration, rows = fixture()
    mutation(rows)
    with pytest.raises(ValueError, match=match):
        compare_registered_methods(registration, rows)


def test_audit_cannot_use_selection_manifest():
    registration, rows = fixture()
    contract = registration["profiles"][0]["contract"]
    contract["evaluation_manifest_sha256"] = contract["selection_manifest_sha256"]
    with pytest.raises(ValueError, match="different split"):
        compare_registered_methods(registration, rows)


def test_family_includes_every_registered_profile():
    registration, rows = fixture(baselines=("janus",))
    second = copy.deepcopy(registration["profiles"][0])
    second["id"] = "another_dataset"
    second["contract"]["dataset"] = "another_dataset"
    registration["profiles"].append(second)
    second_rows = copy.deepcopy(rows)
    for row in second_rows:
        row["profile_id"] = second["id"]
        row["contract"] = copy.deepcopy(second["contract"])
        row["run_id"] += "-second"
    result = compare_registered_methods(registration, rows + second_rows)
    assert result["family_size"] == 2
    assert not result["all_registered_comparisons_supported"]


def write_artifact(path, value):
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_observed_scores_check_hashes_and_do_not_drop_unknown_audits(tmp_path):
    path = tmp_path / "audit.json"
    digest = write_artifact(path, [{"score": 1}, {"score": None}])
    source = {"path": "audit.json", "sha256": digest, "pointer": "", "item_pointer": "/score", "reduction": "mean"}
    with pytest.raises(ValueError, match="missing audit"):
        cli.observed_value(source, tmp_path, {})
    source["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        cli.observed_value(source, tmp_path, {})
    source["sha256"] = write_artifact(path, [{"score": 1, "status": "interrupted"}])
    with pytest.raises(ValueError, match="Incomplete audit"):
        cli.observed_value(source, tmp_path, {})


@pytest.mark.parametrize("document,pointer,item_pointer", [
    ({"status": "interrupted", "score": 1}, "/score", None),
    ([{"status": "failed", "metrics": {"score": 1}}], "/0/metrics/score", None),
    ({"status": "complete", "audit": {"status": "interrupted", "score": 1}}, "/audit/score", None),
    ({"status": "failed", "audit": [{"score": 1}]}, "/audit", "/score"),
    ([{"status": "complete", "metrics": {"status": "failed", "score": 1}}], "", "/metrics/score"),
])
def test_numeric_json_pointers_cannot_bypass_incomplete_ancestor_status(tmp_path, document, pointer, item_pointer):
    digest = write_artifact(tmp_path / "raw.json", document)
    source = {"path": "raw.json", "sha256": digest, "pointer": pointer}
    if item_pointer is not None:
        source.update(reduction="mean", item_pointer=item_pointer)
    with pytest.raises(ValueError, match="Incomplete audit"):
        cli.observed_value(source, tmp_path, {})


def test_completion_status_checks_only_the_selected_outcomes(tmp_path):
    digest = write_artifact(tmp_path / "raw.json", [{"status": "failed", "score": 0},
                                                  {"status": "complete", "score": 1}])
    source = {"path": "raw.json", "sha256": digest, "pointer": "/1/score"}
    assert cli.observed_value(source, tmp_path, {})[0] == 1


def test_cli_checks_raw_observations_and_preserves_existing_reports(tmp_path):
    registration, rows = fixture(baselines=("janus",))
    registration_path = tmp_path / "registration.json"
    registration_path.write_text(json.dumps(registration))
    digest = write_artifact(tmp_path / "observed.json", [{"score": row["score"], "cost": row["actual_cost"]} for row in rows])
    for index, row in enumerate(rows):
        row["evidence"] = {
            "score": {"path": "observed.json", "sha256": digest, "pointer": f"/{index}/score"},
            "actual_cost": {"path": "observed.json", "sha256": digest, "pointer": f"/{index}/cost"}}
    records_path = tmp_path / "runs.jsonl"
    records_path.write_text("\n".join(json.dumps(row) for row in rows))
    output = tmp_path / "comparison.json"
    command = [sys.executable, str(ROOT / "scripts/compare_methods.py"), str(registration_path), str(records_path), "--output", str(output)]
    first = subprocess.run(command, text=True, capture_output=True)
    assert first.returncode == 0, first.stderr
    saved = output.read_bytes()
    assert json.loads(saved)["all_registered_comparisons_supported"]
    second = subprocess.run(command, text=True, capture_output=True)
    assert second.returncode == 2 and output.read_bytes() == saved
    rows[0]["score"] = 999
    records_path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="differs from its raw artifact"):
        cli.load_observed_records(records_path)


def test_same_raw_outcome_cannot_be_reused_as_another_seed(tmp_path):
    registration, rows = fixture(baselines=("janus",))
    digest = write_artifact(tmp_path / "observed.json", {"score": 10, "cost": 10})
    for row in rows[:2]:
        row["score"] = 10
        row["evidence"] = {
            "score": {"path": "observed.json", "sha256": digest, "pointer": "/score"},
            "actual_cost": {"path": "observed.json", "sha256": digest, "pointer": "/cost"}}
    path = tmp_path / "runs.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows[:2]))
    with pytest.raises(ValueError, match="duplicated"):
        cli.load_observed_records(path)


def test_jsonl_provenance_metadata_preserves_unicode_line_separators(tmp_path):
    digest = write_artifact(tmp_path / "source.json", {"score": 1, "cost": 2})
    row = {"profile_id": "example", "note": "first\u2028second\u2029third\u0085last",
           "evidence": {field: {"path": "source.json", "sha256": digest, "pointer": pointer}
                        for field, pointer in (("score", "/score"), ("actual_cost", "/cost"))}}
    path = tmp_path / "runs.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    loaded, _ = cli.load_observed_records(path)
    assert len(loaded) == 1 and loaded[0]["note"] == row["note"]
