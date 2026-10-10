import importlib.util
import json
from pathlib import Path

import pytest
import torch

from specrsi.paper_profile import require_role_checkpoint


def checkpoint(tmp_path, version=3):
    manifest = {"backend": "hf", "architecture_version": version, "feature_dim": 12,
                "option_dim": 12, "frozen_text_snapshot": "frozen_node_text_v1:sha256:" + "a" * 64}
    (tmp_path / "model_config.json").write_text(json.dumps(manifest))
    (tmp_path / "adapters").mkdir()
    heads = {f"{role}_head.node_text_projection.weight": torch.zeros(8, 16)
             for role in ("value", "policy")}
    torch.save(heads, tmp_path / "heads.pt")
    return manifest, heads, {"algorithm": "vats_pdf", "controller": "learned", "role_checkpoint": str(tmp_path)}


def test_explicit_v3_checkpoint_passes_early_identity_and_head_presence_checks(tmp_path):
    _, _, settings = checkpoint(tmp_path)
    assert require_role_checkpoint(settings) == tmp_path


@pytest.mark.parametrize("snapshot", [None, "", "wrong", "frozen_node_text_v1:sha256:" + "z" * 64])
def test_v3_cannot_enter_paid_evaluation_without_valid_encoder_identity(tmp_path, snapshot):
    manifest, _, settings = checkpoint(tmp_path)
    manifest["frozen_text_snapshot"] = snapshot
    (tmp_path / "model_config.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="frozen_text_snapshot"):
        require_role_checkpoint(settings)


@pytest.mark.parametrize("role", ["value", "policy"])
def test_v3_cannot_be_old_heads_under_new_version_label(tmp_path, role):
    _, heads, settings = checkpoint(tmp_path)
    del heads[f"{role}_head.node_text_projection.weight"]
    torch.save(heads, tmp_path / "heads.pt")
    with pytest.raises(ValueError, match="node text projections"):
        require_role_checkpoint(settings)


def test_existing_v2_preflight_does_not_require_v3_features(tmp_path):
    manifest, _, settings = checkpoint(tmp_path, 2)
    del manifest["frozen_text_snapshot"]
    (tmp_path / "model_config.json").write_text(json.dumps(manifest))
    torch.save({}, tmp_path / "heads.pt")
    assert require_role_checkpoint(settings) == tmp_path


def training_script():
    path = Path(__file__).resolve().parents[1] / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("semantic_profile_training_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("requested,saved", [(3, 2), (2, 3)])
def test_explicit_resume_version_mismatch_is_rejected_before_loading(tmp_path, monkeypatch, requested, saved):
    module = training_script()
    (tmp_path / "model_config.json").write_text(json.dumps({"backend": "tiny", "architecture_version": saved}))
    def forbidden(*args, **kwargs):
        raise AssertionError("Model load must not run on a declared architecture mismatch")
    monkeypatch.setattr(module.TinyRoleModel, "load", forbidden)
    with pytest.raises(ValueError, match="architecture_version differs"):
        module.create_model({"model": {"architecture_version": requested}}, tmp_path)


def test_undeclared_resume_version_uses_the_saved_checkpoint(tmp_path, monkeypatch):
    module = training_script()
    (tmp_path / "model_config.json").write_text(json.dumps({"backend": "tiny", "architecture_version": 2}))
    sentinel = object()
    monkeypatch.setattr(module.TinyRoleModel, "load", lambda *args, **kwargs: sentinel)
    assert module.create_model({"model": {}}, tmp_path) is sentinel


@pytest.mark.parametrize("field,value", [("semantic_budget_mode", "redistribute"),
    ("semantic_global_tokens", 512), ("semantic_node_min_tokens", 64),
    ("semantic_node_max_tokens", 2048), ("semantic_total_tokens", 8192),
    ("semantic_truncation", "prefix"), ("max_length", 8192)])
def test_explicit_resume_semantic_budget_cannot_silently_override_legacy(tmp_path, monkeypatch, field, value):
    from specrsi.learning.hf import HFRoleModel
    module = training_script()
    (tmp_path / "model_config.json").write_text(json.dumps({"backend": "hf", "architecture_version": 3,
                                                          "max_length": 2048}))
    def forbidden(*args, **kwargs):
        raise AssertionError("Mismatched observation profile must fail before model allocation")
    monkeypatch.setattr(HFRoleModel, "load", forbidden)
    with pytest.raises(ValueError, match=field + " differs"):
        module.create_model({"model": {field: value}}, tmp_path)


def test_undeclared_or_matching_resume_semantic_profile_uses_saved_contract(tmp_path, monkeypatch):
    from specrsi.learning.hf import HFRoleModel
    module = training_script()
    fields = {"semantic_budget_mode": "redistribute", "semantic_global_tokens": 512,
              "semantic_node_min_tokens": 64, "semantic_node_max_tokens": 2048,
              "semantic_total_tokens": 8192, "semantic_truncation": "head_tail", "max_length": 2048}
    (tmp_path / "model_config.json").write_text(json.dumps({"backend": "hf", "architecture_version": 3, **fields}))
    sentinel = object()
    monkeypatch.setattr(HFRoleModel, "load", lambda *args: sentinel)
    assert module.create_model({"model": {}}, tmp_path) is sentinel
    assert module.create_model({"model": {"backend": "hf", **fields}}, tmp_path) is sentinel


def test_v3_cost_disclosure_preserves_v2_and_separates_quota_from_physical():
    from specrsi.paper_profile import vats_cost_accounting
    old = vats_cost_accounting("tokens", 64)
    assert old == vats_cost_accounting("tokens", 64, 2)
    assert old["scorer_charge"] == "observed_prompt_tokens"
    new = vats_cost_accounting("tokens", 64, 3)
    assert new["scorer_charge"] == "fixed_record_context_plus_unique_node_input_quota"
    assert new["cache_independent_scorer_token_quota"]
    assert new["frozen_node_physical_charge"] == "actual_forward_cache_misses_only"
    wall = vats_cost_accounting("wall-ms", 64, 3)
    assert not wall["cache_independent_scorer_token_quota"]
    assert not wall["physical_bill_equals_budget_spend"]
