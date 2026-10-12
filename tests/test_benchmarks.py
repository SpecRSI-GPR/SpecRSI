import dataclasses
import json
from pathlib import Path
import shutil

import pytest

from specrsi.benchmarks.base import BenchmarkExample, MissingArtifactError
from specrsi.benchmarks.data import assign_split, exclude_audit_overlap, sha256_file, validate_disjoint_splits, verify_manifest
from specrsi.benchmarks.external import NativeHarness, mbpp_bubblewrap_score, require_scientific_harness, swebench_score
from specrsi.benchmarks.planning import UnsupportedPDDL, planbench_strips_score
from specrsi.benchmarks.science import aqm_score, fdtd_score, irp_score, pybamm_score, reactor_score
from specrsi.benchmarks.scorers import aime_score, game24_score, hotpotqa_score, hover_retrieval_score, hover_score, pupa_score


def test_hotpot_native_answer_special_cases_and_joint_evidence():
    assert hotpotqa_score("The Eiffel Tower.", "Eiffel Tower").score == 1
    assert hotpotqa_score("yes", "yes indeed").score == 0
    gold = {"answer": "red car", "supporting_facts": [["A", 0], ["B", 1]]}
    pred = {"answer": "red car", "supporting_facts": [["A", 0]]}
    result = hotpotqa_score(pred, gold, metric="joint_f1")
    assert result.score == pytest.approx(2 / 3)
    assert result.metrics["joint_em"] == 0


def test_hover_retrieval_is_distinct_from_label_accuracy():
    gold = {"label": "SUPPORTED", "supporting_facts": [["The First", 0], ["Second", 1]]}
    assert hover_score("SUPPORTED", gold).score == 1
    assert hover_retrieval_score(["First | text"], gold).score == 0
    assert hover_retrieval_score(["First | text", "SECOND | more"], gold).score == 1


@pytest.mark.parametrize("prediction,score", [(r"Reasoning \boxed{007}", 1), ("### 7", 1),
                                            ("17 is close to 7", 0), ("7.0", 0), ("1007", 0)])
def test_aime_exact_final_integer(prediction, score):
    assert aime_score(prediction, 7).score == score


@pytest.mark.parametrize("expression,numbers,score", [
    ("(6-2)*(5+1)=24", [1, 2, 5, 6], 1),
    ("8/(3-8/3)", [3, 3, 8, 8], 1),
    ("24", [1, 2, 3, 4], 0),
    ("6*4", [1, 2, 3, 4], 0),
    ("__import__('os').system('true')", [1, 2, 3, 4], 0),
    ("2**3*3", [2, 3, 3, 1], 0),
    ("(1+2+3)/0", [0, 1, 2, 3], 0),
])
def test_game24_exact_arithmetic_and_number_conservation(expression, numbers, score):
    assert game24_score(expression, numbers).score == score


def test_pupa_requires_named_judge_and_preserves_native_formula():
    ref = {"user_query": "question", "target_response": "answer", "pii_units": "name"}
    pred = {"response": "response", "llm_request": "redacted"}
    with pytest.raises(MissingArtifactError):
        pupa_score(pred, ref)
    result = pupa_score(pred, ref, judge=lambda **_: {"quality": True, "leakage": 0.2}, judge_identity="registered-test-judge")
    assert result.score == pytest.approx(0.9)


def test_split_leakage_uses_inputs_independent_of_ids():
    first = BenchmarkExample("1", "set", "selection", "  Same Input ", "a")
    second = BenchmarkExample("2", "set", "audit", "same  input", "b")
    assert first.content_hash == second.content_hash
    assert assign_split(first.content_hash) == assign_split(second.content_hash)
    with pytest.raises(ValueError, match="leakage"):
        validate_disjoint_splits({"selection": [first], "audit": [second]})
    kept, excluded = exclude_audit_overlap([first], [second])
    assert not kept and excluded == ["2"]


def test_manifest_detects_corruption_and_missing_artifacts(tmp_path):
    file = tmp_path / "dataset.json"
    file.write_text("{}")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": [{"path": "dataset.json", "sha256": sha256_file(file), "bytes": 2}]}))
    verify_manifest(manifest, root=tmp_path)
    file.write_text("[]")
    with pytest.raises(ValueError, match="hash/size mismatch"):
        verify_manifest(manifest, root=tmp_path)
    file.unlink()
    with pytest.raises(MissingArtifactError):
        verify_manifest(manifest, root=tmp_path)


DOMAIN = """(define (domain blocks) (:requirements :strips :typing)
 (:types block) (:predicates (clear ?b - block) (table ?b - block) (held ?b - block) (empty) (on ?a ?b - block))
 (:action pickup :parameters (?x - block) :precondition (and (clear ?x) (table ?x) (empty))
  :effect (and (held ?x) (not (table ?x)) (not (clear ?x)) (not (empty))))
 (:action stack :parameters (?x ?y - block) :precondition (and (held ?x) (clear ?y))
  :effect (and (on ?x ?y) (clear ?x) (empty) (not (held ?x)) (not (clear ?y)))))"""
PROBLEM = """(define (problem p) (:domain blocks) (:objects a b - block)
 (:init (clear a) (clear b) (table a) (table b) (empty)) (:goal (on a b)))"""


def test_strips_checks_transitions_goal_and_types():
    assert planbench_strips_score("(pickup a)\n(stack a b)", DOMAIN, PROBLEM).score == 1
    assert planbench_strips_score("(stack a b)", DOMAIN, PROBLEM).score == 0
    assert planbench_strips_score("(pickup a)\n(stack a a)", DOMAIN, PROBLEM).score == 0
    assert planbench_strips_score("(pickup c)", DOMAIN, PROBLEM).score == 0
    assert planbench_strips_score("", DOMAIN, PROBLEM).score == 0
    with pytest.raises(UnsupportedPDDL):
        planbench_strips_score("", DOMAIN.replace(":strips", ":adl"), PROBLEM)


def test_native_science_transforms_reject_partial_scenarios():
    assert pybamm_score([2000, 2000, 2000], valid=True, temperatures_c=[15, 25, 35]).score == 1
    run = dict(normalized_goodput=1, delay_p95_seconds=0.1, loss_rate=0, jain_fairness=1)
    assert aqm_score([run] * 6, valid=True).score == pytest.approx(0.9)
    with pytest.raises(ValueError):
        aqm_score([run] * 5, valid=True)
    assert fdtd_score(dict(T1A=1, T2B=1, T1B=0, T2A=0), valid=True).score == 1
    assert reactor_score([dict(n_C=85, n_D=0)] * 3, valid=True).score == 1
    assert irp_score([dict(total_cost=12, total_demand=1)] * 36, valid=True, ddof=0).score == 0
    with pytest.raises(MissingArtifactError):
        require_scientific_harness("pybamm", "/nonexistent/specrsi/scenario.json")


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap unavailable")
def test_mbpp_real_sandbox_correct_wrong_timeout_and_early_exit():
    reference = {"test_list": ["assert add(1,2)==3", "assert add(-1,1)==0"]}
    assert mbpp_bubblewrap_score("def add(a,b): return a+b", reference).score == 1
    assert mbpp_bubblewrap_score("def add(a,b): return a-b", reference).score == 0
    assert mbpp_bubblewrap_score("import os; os._exit(0)", reference).score == 0
    assert mbpp_bubblewrap_score("while True: pass", reference, timeout_seconds=0.3).score == 0


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap unavailable")
def test_mbpp_sandbox_hides_workspace_and_environment(monkeypatch):
    monkeypatch.setenv("SPECRSI_TEST_SECRET", "not-visible")
    code = "import os\nassert 'SPECRSI_TEST_SECRET' not in os.environ\nassert not os.path.exists('/root/.ssh')\nassert not os.path.exists('/private/research_workspace')\ndef add(a,b): return a+b"
    assert mbpp_bubblewrap_score(code, {"test_list": ["assert add(1,2)==3"]}).score == 1


def test_swebench_requires_real_complete_report():
    assert swebench_score({"x": {"resolved": True}}, "x").score == 1
    with pytest.raises(ValueError):
        swebench_score({"resolved": ["x"]}, "x")


def test_livebench_never_uses_implicit_external_judge(tmp_path, monkeypatch):
    import types
    import specrsi.benchmarks.livebench as bridge
    source = tmp_path / "data/raw/livebench_harness/source/LiveBench-test"
    source.mkdir(parents=True)
    attempted = []
    module = types.SimpleNamespace(__file__=str(source / "utils.py"))
    module.is_equiv_llm = lambda *args: attempted.append("external-call")
    # Match the pinned harness's module-global lookup. The bridge clones that
    # function's globals, preserving the original module and ambient environment.
    namespace = {"is_equiv_llm": module.is_equiv_llm}
    exec("def amps_hard_process_results(truth, answer):\n    return is_equiv_llm(truth, answer)\n", namespace)
    module.amps_hard_process_results = namespace["amps_hard_process_results"]
    monkeypatch.setattr(bridge.importlib, "import_module", lambda name: module)
    with pytest.raises(MissingArtifactError, match="explicitly registered judge"):
        bridge.livebench_math_score("answer", {"task": "amps_hard_std", "ground_truth": "x"}, root=tmp_path, verify=False)
    assert not attempted
    assert module.is_equiv_llm is namespace["is_equiv_llm"]
