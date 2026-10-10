import hashlib
import json

import pytest

from specrsi.benchmarks import BenchmarkExample
from specrsi.benchmarks.data import load_examples, sha256_file


def dataset(tmp_path, contents):
    root = tmp_path / "data"
    path = root / "processed" / "hotpotqa" / "development.jsonl"
    path.parent.mkdir(parents=True)
    path.write_bytes(contents.encode("utf-8"))
    manifest = root / "manifests" / "hotpotqa.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"files": [{"path": str(path.relative_to(tmp_path)),
                        "bytes": path.stat().st_size, "sha256": sha256_file(path)}]}))
    return root, path, manifest


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_jsonl_preserves_non_ascii_separators_without_changing_artifacts(tmp_path, separator):
    examples = [BenchmarkExample("one", "hotpotqa", "development", f"中文 left{separator}right", "答案"),
                BenchmarkExample("two", "hotpotqa", "development", "Other question", f"a{separator}b")]
    contents = "\r\n" + "\r\n".join(json.dumps(example.to_dict(), ensure_ascii=False) for example in examples)
    root, path, manifest = dataset(tmp_path, contents)
    original = {file: (file.read_bytes(), sha256_file(file)) for file in (path, manifest)}
    loaded = load_examples("hotpotqa", "development", data_root=root)
    assert loaded == examples
    assert loaded[0].content_hash == examples[0].content_hash
    for file, (raw, digest) in original.items():
        assert file.read_bytes() == raw and sha256_file(file) == digest
        assert hashlib.sha256(raw).hexdigest() == digest


def test_jsonl_parse_error_identifies_physical_line_after_unicode_separator(tmp_path):
    example = BenchmarkExample("one", "hotpotqa", "development", "before\u2028after", "answer")
    contents = json.dumps(example.to_dict(), ensure_ascii=False) + '\n\n{"prompt": "unterminated}\n'
    root, path, _ = dataset(tmp_path, contents)
    with pytest.raises(ValueError) as captured:
        load_examples("hotpotqa", "development", data_root=root)
    assert str(path) in str(captured.value)
    assert "physical line 3" in str(captured.value)
    assert "JSON column" in str(captured.value)
    assert isinstance(captured.value.__cause__, json.JSONDecodeError)
    assert captured.value.__cause__.msg in str(captured.value)


def test_jsonl_record_schema_error_has_file_and_physical_line(tmp_path):
    root, path, _ = dataset(tmp_path, '\n{"id": "missing-fields"}\n')
    with pytest.raises(ValueError) as captured:
        load_examples("hotpotqa", "development", data_root=root)
    assert str(path) in str(captured.value)
    assert "physical line 2" in str(captured.value)
