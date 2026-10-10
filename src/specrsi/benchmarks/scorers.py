"""Scorers preserve each task's metric; failures never become fabricated labels.

HotpotQA equations follow hotpotqa/hotpot/hotpot_evaluate_v1.py. IFBench
delegates constraint semantics to its official package. PUPA requires the
registered judge; a local replacement judge is a different experiment.
"""
from __future__ import annotations

import ast
import collections
from fractions import Fraction
import re
import string
from typing import Callable, Mapping, Sequence

from .base import MissingArtifactError, ScoreResult


def normalize_answer(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def _f1(prediction: str, answer: str) -> tuple[float, float, float]:
    p, g = normalize_answer(prediction), normalize_answer(answer)
    if (p in {"yes", "no", "noanswer"} or g in {"yes", "no", "noanswer"}) and p != g:
        return 0.0, 0.0, 0.0
    common = sum((collections.Counter(p.split()) & collections.Counter(g.split())).values())
    if not common:
        return 0.0, 0.0, 0.0
    precision, recall = common / len(p.split()), common / len(g.split())
    return 2 * precision * recall / (precision + recall), precision, recall


def hotpotqa_score(prediction: str | Mapping, reference: str | Mapping, *, metric: str = "answer_f1") -> ScoreResult:
    pred = {"answer": prediction} if isinstance(prediction, str) else prediction
    gold = {"answer": reference} if isinstance(reference, str) else reference
    f1, precision, recall = _f1(str(pred["answer"]), str(gold["answer"]))
    metrics = {"answer_em": float(normalize_answer(str(pred["answer"])) == normalize_answer(str(gold["answer"]))),
               "answer_f1": f1, "answer_precision": precision, "answer_recall": recall}
    if "supporting_facts" in gold:
        expected = {tuple(x) for x in gold["supporting_facts"]}
        actual = {tuple(x) for x in pred.get("supporting_facts", [])}
        n = len(actual & expected)
        sp_p, sp_r = n / len(actual) if actual else 0.0, n / len(expected) if expected else 0.0
        metrics.update(sp_em=float(actual == expected), sp_precision=sp_p, sp_recall=sp_r,
                       sp_f1=2 * sp_p * sp_r / (sp_p + sp_r) if sp_p + sp_r else 0.0)
        joint_p, joint_r = precision * sp_p, recall * sp_r
        metrics.update(joint_em=metrics["answer_em"] * metrics["sp_em"],
                       joint_f1=2 * joint_p * joint_r / (joint_p + joint_r) if joint_p + joint_r else 0.0)
    if metric not in metrics:
        raise ValueError(f"Unavailable HotpotQA metric: {metric}")
    return ScoreResult(metrics[metric], metrics)


def hover_score(prediction: str | Mapping, reference: str | Mapping, *, metric: str = "label_accuracy") -> ScoreResult:
    """Claim accuracy, with exact evidence completion reported separately."""
    p = {"label": prediction} if isinstance(prediction, str) else prediction
    g = {"label": reference} if isinstance(reference, str) else reference
    def label(v):
        return str(v).strip().upper().replace("_", " ")
    metrics = {"label_accuracy": float(label(p["label"]) == label(g["label"]))}
    if "supporting_facts" in g:
        expected = {tuple(x) for x in g["supporting_facts"]}
        actual = {tuple(x) for x in p.get("supporting_facts", [])}
        metrics["all_supporting_facts"] = float(expected <= actual)
        metrics["label_evidence_completion"] = metrics["label_accuracy"] * metrics["all_supporting_facts"]
    if metric not in metrics:
        raise ValueError(f"Unavailable HoVer metric: {metric}")
    return ScoreResult(metrics[metric], metrics)


def hover_retrieval_score(retrieved_docs: Sequence[str], reference: Mapping) -> ScoreResult:
    """GEPA's discrete document-retrieval objective, separate from claim labels."""
    # DSPy normalize_text lowercases, removes punctuation/articles and whitespace.
    facts = reference["supporting_facts"]
    titles = [fact["key"] if isinstance(fact, Mapping) else fact[0] for fact in facts]
    expected = {normalize_answer(title) for title in titles}
    if not expected:
        raise ValueError("HoVer retrieval requires nonempty gold supporting titles")
    actual = {normalize_answer(doc.split(" | ")[0]) for doc in retrieved_docs}
    score = float(expected <= actual)
    return ScoreResult(score, {"all_documents_retrieved": score, "gold_documents": len(expected),
                               "documents_retrieved": len(expected & actual)})


def extract_final_answer(text: str) -> str:
    """Use the last balanced boxed answer, explicit answer line, or full text."""
    starts = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    if starts:
        start = starts[-1].end()
        depth = 1
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index].strip()
    lines = re.findall(r"(?:^|\n)\s*(?:###\s*|(?:final\s+)?answer\s*[:=]\s*)([^\n]+)", text, re.I)
    return (lines[-1] if lines else text).strip().strip("$").strip()


def aime_score(prediction: str, reference: str | int) -> ScoreResult:
    answer = extract_final_answer(prediction)
    valid = bool(re.fullmatch(r"\d{1,3}", answer))
    score = float(valid and int(answer) == int(reference))
    return ScoreResult(score, {"accuracy": score, "extracted_answer": answer, "valid_integer": valid})


def game24_score(prediction: str, numbers: Sequence[int]) -> ScoreResult:
    """Evaluate only four-number +,-,*,/ arithmetic, with exact rationals."""
    if len(numbers) != 4 or any(type(number) is not int or number < 0 for number in numbers):
        raise ValueError("Game24 requires exactly four nonnegative integers")
    expression = extract_final_answer(prediction).strip()
    if "=" in expression:
        expression, rhs = expression.rsplit("=", 1)
        if rhs.strip() != "24":
            return ScoreResult(0.0, {"reason": "wrong equality target"})
    if len(expression) > 256:
        return ScoreResult(0.0, {"reason": "expression too long"})
    leaves: list[int] = []
    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) is int and node.value >= 0:
            leaves.append(node.value)
            return Fraction(node.value)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            return left / right
        raise ValueError("Only nonnegative integer leaves and + - * / are allowed")
    try:
        result = evaluate(ast.parse(expression.strip(), mode="eval").body)
        valid = collections.Counter(leaves) == collections.Counter(numbers)
        score = float(valid and result == 24)
        return ScoreResult(score, {"accuracy": score, "valid_number_use": valid, "value": str(result)})
    except (ValueError, SyntaxError, ZeroDivisionError, RecursionError):
        return ScoreResult(0.0, {"reason": "invalid arithmetic expression"})


def ifbench_score(prediction: str, reference: Mapping, *, metric: str = "prompt_loose") -> ScoreResult:
    try:
        from pathlib import Path
        import nltk
        resource_path = Path(__file__).resolve().parents[3] / "data/raw/nltk/source"
        if resource_path.is_dir() and str(resource_path) not in nltk.data.path:
            nltk.data.path.insert(0, str(resource_path))
        from ifbench import instructions_registry
    except ImportError as exc:
        raise MissingArtifactError("Install the pinned official ifbench package in the benchmark environment") from exc
    strict, loose = [], []
    lines = prediction.split("\n")
    variants = [prediction, prediction.replace("*", ""), "\n".join(lines[1:]).strip(),
                "\n".join(lines[:-1]).strip(), "\n".join(lines[1:-1]).strip()]
    variants += [v.replace("*", "") for v in variants[2:]]
    ids, kwargs = reference["instruction_id_list"], reference["kwargs"]
    if not ids or len(ids) != len(kwargs):
        raise ValueError("IFBench needs aligned nonempty instruction IDs and kwargs")
    for instruction_id, values in zip(ids, kwargs):
        cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        checker = cls(instruction_id)
        checker.build_description(**{k: v for k, v in values.items() if v is not None})
        if "prompt" in (checker.get_instruction_args() or {}):
            checker.build_description(prompt=reference["prompt"])
        strict.append(bool(prediction.strip()) and bool(checker.check_following(prediction)))
        loose.append(any(bool(v.strip()) and bool(checker.check_following(v)) for v in variants))
    metrics = {"prompt_strict": float(all(strict)), "prompt_loose": float(all(loose)),
               "instruction_strict": sum(strict) / len(strict), "instruction_loose": sum(loose) / len(loose)}
    return ScoreResult(metrics[metric], metrics)


def pupa_score(prediction: Mapping, reference: Mapping, *, judge: Callable | None = None, judge_identity: str | None = None) -> ScoreResult:
    """PAPILLON's mean of quality and 1-leakage, using an explicit judge."""
    if judge is None or not judge_identity:
        raise MissingArtifactError("PUPA requires the registered quality/leakage judge and its checkpoint identity")
    result = judge(user_query=reference["user_query"], target_response=reference["target_response"],
                   response=prediction["response"], llm_request=prediction["llm_request"], pii_units=reference["pii_units"])
    quality, leakage = float(result["quality"]), float(result["leakage"])
    if quality not in (0.0, 1.0) or not 0.0 <= leakage <= 1.0:
        raise ValueError("PUPA judge returned invalid quality/leakage")
    return ScoreResult((quality + 1 - leakage) / 2, {"quality": quality, "leakage": leakage, "judge_identity": judge_identity})
