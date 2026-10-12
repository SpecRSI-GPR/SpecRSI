"""No model calls: prospective selection, journals and paid failure boundaries."""
import importlib.util
import hashlib
import json
from pathlib import Path

import pytest

from specrsi.benchmarks import BenchmarkExample
from specrsi.core.execution import ExecutionResult
from specrsi.core.models import Outcome


SPEC = importlib.util.spec_from_file_location("performance_campaign", Path(__file__).parents[1] / "scripts/run_performance_campaign.py")
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


class Executor:
    def __init__(self, scores=None, costs=None, behavior=None):
        self.scores, self.costs, self.behavior = scores or {}, costs or {}, behavior
        self.physical_records, self.calls = [], []

    def validate_prompt_candidate(self, *args):
        pass

    def run(self, candidate, task, scope, cap):
        split = "development" if scope.split == "dev" else scope.split
        key = split, candidate.name, task.name
        self.calls.append(key)
        cost = self.costs.get(candidate.name, 2)
        record = {"candidate": candidate.uid, "task": task.uid, "scope": scope.uid,
            "request_id": ":".join(key), "prompt_tokens": 1, "completion_tokens": cost - 1,
            "elapsed_seconds": .25, "gpu_count": 1, "finish_reason": "stop"}
        self.physical_records.append(record)
        if self.behavior:
            result = self.behavior(self, key, record, cost, cap)
            if result is not None:
                return result
        score = self.scores.get((split, candidate.name), .5)
        return ExecutionResult("complete", cost, Outcome(score, {"answer": candidate.name}), {"request_id": record["request_id"]})


@pytest.fixture
def setup(tmp_path, monkeypatch):
    lock = tmp_path / "model.json"
    lock.write_text(json.dumps({"revision": "frozen-model", "local_path": "/not-loaded"}))
    config = {"schema": "inference_performance_v1", "output": str(tmp_path / "campaign"),
        "model_lock": str(lock), "dataset": "hotpotqa", "sample_seed": 7, "decode_seed": 9,
        "counts": {"development": 2, "selection": 2, "audit": 2},
        "execution_cap": 10, "total_token_budget": 1000, "finalists": 1,
        "baseline": "baseline", "audit_controls": ["control"],
        "candidates": [{"name": name, "artifacts": {"system_prompt": name},
                        "generation": {"max_tokens": 3}} for name in ("baseline", "challenger", "control", "excluded")]}
    loaded = {split: [BenchmarkExample(f"{split}-{i}", "hotpotqa", split,
                                      f"Unique public question {split} {i}", "PRIVATE_REFERENCE") for i in range(3)]
              for split in config["counts"]}
    monkeypatch.setattr(campaign, "source_hashes", lambda: {"src/registered.py": "fixed"})
    return config, loaded, Path(config["output"])


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_audit_cannot_select_recipe_and_finalists_follow_development(setup):
    config, loaded, output = setup
    scores = {("development", "challenger"): .9, ("development", "excluded"): .8,
        ("selection", "challenger"): .8, ("selection", "baseline"): .5,
        ("selection", "control"): .6, ("audit", "challenger"): 0.,
        ("audit", "baseline"): 1., ("audit", "control"): .9}
    executor = Executor(scores)
    result = campaign.run(config, executor=executor, loaded=loaded)
    assert result["selected"] == "challenger"  # Despite losing every audit comparison.
    assert set(result["confirmation"]) == {"baseline", "control", "challenger"}
    assert result["selected_vs_controls"]["baseline"]["mean_difference"] == -1.
    frozen = json.loads((output / "frozen_choice.json").read_text())
    assert frozen["selected"] == "challenger" and "audit" not in frozen["confirmation"]
    assert all(name != "excluded" for split, name, _ in executor.calls if split != "development")
    assert result["sota_claimed"] is False and result["statistical_superiority_claimed"] is False


def test_confirmed_winner_uses_lower_cost_then_name_for_exact_score_ties(setup):
    config, loaded, _ = setup
    config["finalists"] = 4
    config["audit_controls"] = []
    result = campaign.run(config, executor=Executor(costs={"baseline": 4, "challenger": 3, "control": 2, "excluded": 2}),
                          loaded=loaded, stop_after="selection")
    assert result["selected"] == "control"


def test_interruption_and_unknown_usage_stay_in_denominator_and_costs(setup):
    config, loaded, output = setup
    def behavior(executor, key, record, cost, cap):
        if key[1] == "challenger":
            return ExecutionResult("interrupted", cost, error="timeout")
        if key[1] == "excluded":
            record.clear()
            record.update({"request_id": ":".join(key), "metering_status": "unavailable_reserved_upper_bound",
                           "accounted_tokens": 7, "reserved_tokens": 7})
            return ExecutionResult("error", 7, error="transport failure")
    result = campaign.run(config, executor=Executor(behavior=behavior), loaded=loaded)
    assert result["development"]["challenger"]["count"] == 2
    assert result["development"]["challenger"]["mean_score"] == 0.
    assert result["development"]["challenger"]["charged_tokens"] == 4
    unknown = result["development"]["excluded"]
    assert unknown["complete"] == 0 and unknown["count"] == 2
    assert unknown["charged_tokens"] == unknown["reserved_unmetered_tokens"] == 14
    assert unknown["measured_tokens"] == 0 and unknown["metering_complete"] is False
    assert set(result["confirmation"]) == {"baseline", "control"}
    assert result["complete"] is False
    rows = read_rows(output / "results.jsonl")
    assert result["total_charged_tokens"] == sum(r["charged_tokens"] for r in rows)
    assert all(r["score"] == 0 for r in rows if r["status"] != "complete")


def test_resume_reuses_paid_records_and_frozen_choice_without_new_calls(setup):
    config, loaded, output = setup
    first = Executor()
    frozen = campaign.run(config, executor=first, loaded=loaded, stop_after="selection")
    second = Executor()
    result = campaign.run(config, executor=second, loaded=loaded)
    assert result["selected"] == frozen["selected"]
    assert second.calls and all(split == "audit" for split, _, _ in second.calls)
    third = Executor()
    repeated = campaign.run(config, executor=third, loaded=loaded)
    assert not third.calls and repeated == result
    assert len(read_rows(output / "results.jsonl")) == len(first.calls) + len(second.calls)


def test_lost_result_preserves_pending_and_physical_records_and_blocks_retry(setup):
    config, loaded, output = setup
    def fail(*args):
        raise ConnectionError("connection lost after admitted model request")
    first = Executor(behavior=fail)
    with pytest.raises(ConnectionError):
        campaign.run(config, executor=first, loaded=loaded)
    assert (output / "pending_request.json").is_file()
    assert len(read_rows(output / "physical_calls.jsonl")) == 1
    assert not read_rows(output / "results.jsonl")
    retry = Executor()
    with pytest.raises(RuntimeError, match="no free retry"):
        campaign.run(config, executor=retry, loaded=loaded)
    assert not retry.calls


@pytest.mark.parametrize("violation", ["undercharge", "overcap", "complete_unmetered"])
def test_invalid_executor_accounting_is_never_admitted_and_cannot_retry_free(setup, violation):
    config, loaded, output = setup
    def behavior(executor, key, record, cost, cap):
        charged = cost
        if violation == "undercharge":
            charged -= 1
        elif violation == "overcap":
            record["completion_tokens"] = cap
            charged = cap + 1
        else:
            record.clear()
            record.update({"request_id": ":".join(key), "metering_status": "unavailable_reserved_upper_bound",
                           "accounted_tokens": 2, "reserved_tokens": 2})
        return ExecutionResult("complete", charged, Outcome(1., {"answer": "untrusted"}))
    with pytest.raises(ValueError):
        campaign.run(config, executor=Executor(behavior=behavior), loaded=loaded)
    assert (output / "pending_request.json").is_file() and not read_rows(output / "results.jsonl")
    assert len(read_rows(output / "physical_calls.jsonl")) == 1


def test_changed_configuration_or_source_invalidates_registration_before_calls(setup, monkeypatch):
    config, loaded, output = setup
    campaign.run(config, executor=Executor(), loaded=loaded, stop_after="register")
    config["decode_seed"] += 1
    executor = Executor()
    with pytest.raises(ValueError, match="Frozen registration changed"):
        campaign.run(config, executor=executor, loaded=loaded)
    assert not executor.calls
    config["decode_seed"] -= 1
    monkeypatch.setattr(campaign, "source_hashes", lambda: {"src/registered.py": "changed"})
    with pytest.raises(ValueError, match="Frozen registration changed"):
        campaign.run(config, executor=executor, loaded=loaded)
    assert not executor.calls


@pytest.mark.parametrize("change", ["source", "model_lock"])
def test_source_or_lock_change_during_run_stops_after_preserving_first_paid_attempt(setup, monkeypatch, change):
    config, loaded, output = setup
    def behavior(*args):
        if change == "source":
            monkeypatch.setattr(campaign, "source_hashes", lambda: {"src/registered.py": "changed"})
        else:
            Path(config["model_lock"]).write_text(json.dumps({"revision": "new-model"}))
    executor = Executor(behavior=behavior)
    with pytest.raises(ValueError, match="changed during execution"):
        campaign.run(config, executor=executor, loaded=loaded)
    assert len(executor.calls) == len(read_rows(output / "physical_calls.jsonl")) == 1
    assert not read_rows(output / "results.jsonl") and (output / "pending_request.json").exists()


@pytest.mark.parametrize("field,value", [("candidate_uid", "wrong"), ("charged_tokens", 0), ("measured_tokens", 0)])
def test_tampered_resume_rows_are_rejected_before_more_paid_work(setup, field, value):
    config, loaded, output = setup
    campaign.run(config, executor=Executor(), loaded=loaded, stop_after="development")
    rows = read_rows(output / "results.jsonl")
    rows[0][field] = value
    (output / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    executor = Executor()
    with pytest.raises(ValueError):
        campaign.run(config, executor=executor, loaded=loaded)
    assert not executor.calls


def test_changed_frozen_choice_is_rejected_before_audit(setup):
    config, loaded, output = setup
    campaign.run(config, executor=Executor(), loaded=loaded, stop_after="selection")
    choice = json.loads((output / "frozen_choice.json").read_text())
    choice["selected"] = "excluded"
    (output / "frozen_choice.json").write_text(json.dumps(choice))
    executor = Executor()
    with pytest.raises(ValueError, match="Frozen choice changed"):
        campaign.run(config, executor=executor, loaded=loaded)
    assert not executor.calls


def test_budget_reserves_full_next_execution_and_never_spends_partial_headroom(setup):
    config, loaded, output = setup
    config["total_token_budget"] = 11
    executor = Executor()
    with pytest.raises(RuntimeError, match="cannot reserve"):
        campaign.run(config, executor=executor, loaded=loaded)
    assert len(executor.calls) == 1
    assert sum(row["charged_tokens"] for row in read_rows(output / "results.jsonl")) == 2
    assert not (output / "pending_request.json").exists()


def test_paired_comparison_rejects_different_task_orders():
    with pytest.raises(ValueError, match="same ordered"):
        campaign.paired_comparison([{"example_id": "one", "score": 1}], [{"example_id": "two", "score": 1}], 0)


def write_prior_registration(output, rows, *, dataset="hotpotqa", name="prior"):
    directory = output.parent / name
    directory.mkdir()
    path = directory / "registration.json"
    path.write_text(json.dumps({"config": {"schema": "inference_performance_v1", "dataset": dataset,
        "counts": {"audit": len(rows)}}, "examples": {"audit": rows}}))
    return path


def test_audit_exclusions_filter_full_pool_before_sampling_without_touching_other_splits(setup):
    config, loaded, output = setup
    # Exclude the item that would otherwise rank first, and force refill from the
    # remaining full pool rather than truncating a previously sampled subset.
    original = campaign.choose(loaded["audit"], 2, config["sample_seed"])
    protected = original[0]
    prior = write_prior_registration(output, [{"id": protected.id, "input_hash": protected.content_hash}])
    config["exclude_prior_campaigns"] = [str(prior.parent)]
    executor = Executor()
    campaign.run(config, executor=executor, loaded=loaded, stop_after="register")
    registration = json.loads((output / "registration.json").read_text())
    meta = registration["audit_exclusions"]
    assert meta["excluded_count"] == 1 and meta["audit_pool_after_count"] == 2
    assert meta["sources"][0]["registration_sha256"] == hashlib.sha256(prior.read_bytes()).hexdigest()
    assert meta["sources"][0]["registered_audit_count"] == 1
    assert meta["sources"][0]["audit_examples"] == [{"id": protected.id, "input_hash": protected.content_hash}]
    expected = campaign.choose([e for e in loaded["audit"] if e.id != protected.id], 2, config["sample_seed"])
    assert [row["id"] for row in registration["examples"]["audit"]] == [e.id for e in expected]
    for split in ("development", "selection"):
        assert registration["examples"][split] == [{"id": e.id, "input_hash": e.content_hash}
            for e in campaign.choose(loaded[split], config["counts"][split], config["sample_seed"])]
    assert not executor.calls and "PRIVATE_REFERENCE" not in (output / "registration.json").read_text()
    assert len(loaded["audit"]) == 3  # Caller-owned inputs were not mutated.


@pytest.mark.parametrize("matching", ["renamed_id", "changed_input_same_id", "renamed_dataset"])
def test_prior_inputs_match_by_qualified_id_or_cross_id_cross_dataset_hash(setup, matching):
    config, loaded, output = setup
    example = loaded["audit"][0]
    row = {"id": "historical-id" if matching != "changed_input_same_id" else example.id,
           "input_hash": "f" * 64 if matching == "changed_input_same_id" else example.content_hash}
    prior = write_prior_registration(output, [row], dataset="renamed_dataset" if matching == "renamed_dataset" else "hotpotqa")
    config["exclude_prior_campaigns"] = [str(prior)]
    campaign.run(config, executor=Executor(), loaded=loaded, stop_after="register")
    registration = json.loads((output / "registration.json").read_text())
    assert example.id not in {r["id"] for r in registration["examples"]["audit"]}
    assert registration["audit_exclusions"]["excluded_count"] == 1


def test_foreign_dataset_id_collision_does_not_exclude_different_input(setup):
    config, loaded, output = setup
    prior = write_prior_registration(output, [{"id": loaded["audit"][0].id, "input_hash": "f" * 64}], dataset="other")
    config["exclude_prior_campaigns"] = [str(prior)]
    eligible, provenance = campaign.exclude_prior_audit(config, loaded)
    assert eligible["audit"] == loaded["audit"] and provenance["excluded_count"] == 0


def test_repeated_protected_inputs_across_sources_are_counted_once_in_union(setup):
    config, loaded, output = setup
    example = loaded["audit"][0]
    rows = [{"id": example.id, "input_hash": example.content_hash}]
    first = write_prior_registration(output, rows)
    second = write_prior_registration(output, rows, name="second")
    config["exclude_prior_campaigns"] = [str(first), str(second)]
    eligible, provenance = campaign.exclude_prior_audit(config, loaded)
    assert len(provenance["sources"]) == 2
    assert provenance["unique_prior_audit_ids"] == provenance["unique_prior_audit_input_hashes"] == 1
    assert len(eligible["audit"]) == 2 and provenance["excluded_count"] == 1


@pytest.mark.parametrize("split", ["development", "selection"])
def test_prior_audit_cannot_be_reintroduced_as_candidate_selection_data(setup, split):
    config, loaded, output = setup
    example = loaded[split][0]
    prior = write_prior_registration(output, [{"id": "historical-id", "input_hash": example.content_hash}])
    config["exclude_prior_campaigns"] = [str(prior)]
    executor = Executor()
    with pytest.raises(ValueError, match=f"overlap {split}"):
        campaign.run(config, executor=executor, loaded=loaded, stop_after="register")
    assert not executor.calls and not (output / "registration.json").exists()


@pytest.mark.parametrize("change", ["bytes_only", "identity", "missing"])
def test_changed_exclusion_source_is_rejected_on_resume_before_calls(setup, change):
    config, loaded, output = setup
    example = loaded["audit"][0]
    prior = write_prior_registration(output, [{"id": example.id, "input_hash": example.content_hash}])
    config["exclude_prior_campaigns"] = [str(prior)]
    campaign.run(config, executor=Executor(), loaded=loaded, stop_after="register")
    if change == "missing":
        prior.unlink()
    elif change == "bytes_only":
        prior.write_text(prior.read_text() + "\n")
    else:
        previous = json.loads(prior.read_text())
        previous["examples"]["audit"][0]["id"] = "different-id"
        prior.write_text(json.dumps(previous))
    executor = Executor()
    with pytest.raises(ValueError):
        campaign.run(config, executor=executor, loaded=loaded)
    assert not executor.calls


def test_exclusion_source_mutation_during_execution_preserves_paid_journal(setup):
    config, loaded, output = setup
    prior = write_prior_registration(output, [{"id": "old", "input_hash": "f" * 64}])
    config["exclude_prior_campaigns"] = [str(prior)]
    def behavior(*args):
        prior.write_text(prior.read_text() + "\n")
    executor = Executor(behavior=behavior)
    with pytest.raises(ValueError, match="exclusion source changed"):
        campaign.run(config, executor=executor, loaded=loaded)
    assert len(executor.calls) == len(read_rows(output / "physical_calls.jsonl")) == 1
    assert not read_rows(output / "results.jsonl") and (output / "pending_request.json").exists()


def test_filtered_pool_cannot_silently_reduce_registered_audit_count(setup):
    config, loaded, output = setup
    prior = write_prior_registration(output, [{"id": e.id, "input_hash": e.content_hash} for e in loaded["audit"]])
    config["exclude_prior_campaigns"] = [str(prior)]
    with pytest.raises(ValueError, match="Invalid count 2 for 0 examples"):
        campaign.run(config, executor=Executor(), loaded=loaded, stop_after="register")
    assert not (output / "registration.json").exists()


@pytest.mark.parametrize("invalid", [None, "path", {}, [None], [""]])
def test_exclusion_configuration_requires_an_explicit_path_list(setup, invalid):
    config, loaded, output = setup
    config["exclude_prior_campaigns"] = invalid
    with pytest.raises(ValueError, match="exclude_prior_campaigns"):
        campaign.run(config, executor=Executor(), loaded=loaded, stop_after="register")
    assert not (output / "registration.json").exists()


@pytest.mark.parametrize("defect", ["missing_hash", "bad_hash", "duplicate_id", "count_mismatch", "empty_audit", "unknown_schema", "missing_file", "duplicate_source", "self_source"])
def test_malformed_or_ambiguous_prior_registrations_fail_closed(setup, defect):
    config, loaded, output = setup
    prior = write_prior_registration(output, [{"id": "old", "input_hash": "f" * 64}])
    previous = json.loads(prior.read_text())
    if defect == "missing_hash":
        del previous["examples"]["audit"][0]["input_hash"]
    elif defect == "bad_hash":
        previous["examples"]["audit"][0]["input_hash"] = "not-a-hash"
    elif defect == "duplicate_id":
        previous["examples"]["audit"] *= 2
        previous["config"]["counts"]["audit"] = 2
    elif defect == "count_mismatch":
        previous["config"]["counts"]["audit"] = 2
    elif defect == "empty_audit":
        previous["examples"]["audit"] = []
        previous["config"]["counts"]["audit"] = 0
    elif defect == "unknown_schema":
        previous["config"]["schema"] = "unrecognized"
    prior.write_text(json.dumps(previous))
    config["exclude_prior_campaigns"] = [str(prior)]
    if defect == "missing_file":
        prior.unlink()
    elif defect == "duplicate_source":
        config["exclude_prior_campaigns"].append(str(prior.parent))
    elif defect == "self_source":
        config["exclude_prior_campaigns"] = [str(output / "registration.json")]
    with pytest.raises(ValueError):
        campaign.run(config, executor=Executor(), loaded=loaded, stop_after="register")
    assert not (output / "registration.json").exists()


def test_exclusion_sources_do_not_read_prior_results_or_answers(setup, monkeypatch):
    config, loaded, output = setup
    example = loaded["audit"][0]
    prior = write_prior_registration(output, [{"id": example.id, "input_hash": example.content_hash}])
    config["exclude_prior_campaigns"] = [str(prior.parent)]
    original = Path.open
    def guarded_open(path, *args, **kwargs):
        if path.parent == prior.parent and path.name != "registration.json":
            raise AssertionError("Prior scores and answers must never be opened")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    campaign.run(config, executor=Executor(), loaded=loaded, stop_after="development")
    assert (output / "development_summary.json").exists()


def test_legacy_configuration_does_not_gain_an_exclusion_registration_field(setup):
    config, loaded, output = setup
    campaign.run(config, executor=Executor(), loaded=loaded, stop_after="register")
    registration = json.loads((output / "registration.json").read_text())
    assert "audit_exclusions" not in registration
    eligible, provenance = campaign.exclude_prior_audit(config, loaded)
    assert eligible is loaded and provenance is None


def test_explicit_empty_exclusions_preserve_samples_but_record_empty_policy(setup):
    config, loaded, output = setup
    config["exclude_prior_campaigns"] = []
    campaign.run(config, executor=Executor(), loaded=loaded, stop_after="register")
    registration = json.loads((output / "registration.json").read_text())
    assert registration["audit_exclusions"]["sources"] == []
    assert registration["audit_exclusions"]["audit_pool_before_count"] == registration["audit_exclusions"]["audit_pool_after_count"] == 3


def test_eligible_pool_drift_is_rejected_even_when_the_selected_sample_is_unchanged(setup):
    config, loaded, output = setup
    config["exclude_prior_campaigns"] = []
    campaign.run(config, executor=Executor(), loaded=loaded, stop_after="register")
    selected = campaign.choose(loaded["audit"], 2, config["sample_seed"])
    loaded["audit"] = list(selected)  # Drop only an unselected eligible example.
    executor = Executor()
    with pytest.raises(ValueError, match="Frozen registration changed"):
        campaign.run(config, executor=executor, loaded=loaded)
    assert not executor.calls


def test_excluded_references_never_reach_candidate_preflight_or_execution(setup):
    config, loaded, output = setup
    protected = loaded["audit"][0]
    prior = write_prior_registration(output, [{"id": protected.id, "input_hash": protected.content_hash}])
    class ProtectedInput:
        id, dataset, split = protected.id, protected.dataset, protected.split
        prompt, metadata, content_hash = protected.prompt, protected.metadata, protected.content_hash
        @property
        def reference(self):
            raise AssertionError("Protected reference must never be accessed")
    loaded["audit"][0] = ProtectedInput()
    config["exclude_prior_campaigns"] = [str(prior)]
    executor = Executor()
    result = campaign.run(config, executor=executor, loaded=loaded)
    assert result["complete"]
    assert protected.id not in {example_id for split, _, example_id in executor.calls if split == "audit"}


def test_exclusion_registration_resumes_unchanged_without_repeating_paid_calls(setup):
    config, loaded, output = setup
    protected = loaded["audit"][0]
    prior = write_prior_registration(output, [{"id": protected.id, "input_hash": protected.content_hash}])
    config["exclude_prior_campaigns"] = [str(prior)]
    first = Executor()
    campaign.run(config, executor=first, loaded=loaded, stop_after="selection")
    before = (output / "registration.json").read_bytes()
    second = Executor()
    campaign.run(config, executor=second, loaded=loaded)
    assert second.calls and all(split == "audit" for split, _, _ in second.calls)
    assert protected.id not in {example_id for _, _, example_id in second.calls}
    assert (output / "registration.json").read_bytes() == before
