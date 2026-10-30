import copy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("aime_full", Path(__file__).parents[1] / "scripts/aime2025_full.py")
full = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(full)


@pytest.fixture
def prepared(tmp_path):
    config = json.loads(full.DEFAULT_CONFIG.read_text())
    raw = b"".join((json.dumps({"id": str(i), "problem": f"Fixture question {i}", "answer": i}) + "\n").encode()
                   for i in range(30))
    config["files"] = [{"filename": "test.jsonl", "bytes": len(raw), "sha256": full.sha256(raw)}]
    path = tmp_path / "raw" / config["dataset"] / "test.jsonl"
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    full.prepare(config, tmp_path, offline=True)
    return config, tmp_path


def test_full_profile_is_evaluation_only_and_compatible_with_loader(prepared):
    from specrsi.benchmarks.data import load_examples
    config, root = prepared
    examples, manifest = full.verify(config, root)
    assert manifest["counts"] == {"development": 0, "selection": 0, "audit": 30}
    assert len(examples) == len(load_examples(config["dataset"], "audit", data_root=root)) == 30
    assert all(e.metadata["year"] == 2025 for e in examples)
    assert not load_examples(config["dataset"], "selection", data_root=root)
    assert full.prepare(config, root, offline=True)["verified"]


def test_wrong_year_and_duplicate_or_missing_source_ids_are_rejected(prepared):
    config, root = prepared
    raw = (root / "raw" / config["dataset"] / "test.jsonl").read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    for modified in (rows[:-1], rows[:-1] + [rows[0]], rows + [dict(rows[0], id="30")]):
        with pytest.raises(ValueError, match="exactly once"):
            full.normalize("\n".join(json.dumps(r) for r in modified).encode(), config)
    with pytest.raises(ValueError, match="2025 profile"):
        full.verify(dict(config, year=2026), root)


def test_verification_rebuilds_expected_bytes_instead_of_trusting_rewritten_manifest(prepared):
    config, root = prepared
    path = root / "processed" / config["dataset"] / "audit.jsonl"
    path.write_text(path.read_text().replace('"reference": "0"', '"reference": "9"'))
    manifest = root / "manifests" / (config["dataset"] + ".json")
    doc = json.loads(manifest.read_text())
    for record in doc["files"]:
        if record["path"].endswith("audit.jsonl"):
            record["sha256"] = full.sha256(path.read_bytes())
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="missing or changed"):
        full.verify(config, root)
    with pytest.raises(ValueError, match="changed"):
        full.prepare(config, root, offline=True)


def test_complete_predictions_score_each_seed_before_sd(prepared):
    config, root = prepared
    rows = [{"seed": seed, "id": f"aime2025:{i}", "prediction": str(i) if seed == 1 else "",
             "candidate_sha256": "a" * 64} for seed in (1, 2) for i in range(30)]
    result = full.score_predictions(config, root, rows)
    assert result["mean_percent"] == 50
    assert result["sd_percent"] == pytest.approx(70.710678)
    assert [s["correct"] for s in result["seeds"]] == [30, 0]
    for modified, error in ((rows[:-1], "all 30"), (rows + [rows[0]], "Duplicate"),
                            ([dict(rows[0], id="2026:0")] + rows[1:], "registered ID"),
                            ([dict(rows[0], candidate_sha256="b" * 64)] + rows[1:], "Candidate changed")):
        with pytest.raises(ValueError, match=error):
            full.score_predictions(config, root, modified)


def test_changed_raw_bytes_are_rejected(prepared):
    config, root = prepared
    path = root / "raw" / config["dataset"] / "test.jsonl"
    path.write_bytes(path.read_bytes().replace(b"Fixture question 0", b"Altered question 0"))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        full.verify(config, root)
