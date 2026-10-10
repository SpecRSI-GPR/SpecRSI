"""Exact plan validator for the explicitly supported conjunctive STRIPS scope.

Full PDDL, temporal/numeric planning and unsupported expressions fail closed.
Use PlanBench's native VAL harness for those domains.
"""
from __future__ import annotations

import re
from .base import MissingArtifactError, ScoreResult


class UnsupportedPDDL(MissingArtifactError):
    pass


def parse_pddl(text: str) -> list:
    text = re.sub(r";[^\n]*", "", text.lower())
    tokens = re.findall(r"\(|\)|[^\s()]+", text)
    stack, root = [], []
    current = root
    for token in tokens:
        if token == "(":
            child = []
            current.append(child)
            stack.append(current)
            current = child
        elif token == ")":
            if not stack:
                raise ValueError("Unbalanced PDDL parentheses")
            current = stack.pop()
        else:
            current.append(token)
    if stack or len(root) != 1 or not isinstance(root[0], list):
        raise ValueError("Expected one complete PDDL expression")
    return root[0]


def _typed_symbols(tokens: list) -> dict[str, str]:
    result, pending = {}, []
    index = 0
    while index < len(tokens):
        if tokens[index] == "-":
            if not pending or index + 1 == len(tokens) or isinstance(tokens[index + 1], list):
                raise UnsupportedPDDL("Malformed or compound PDDL type")
            for name in pending:
                result[name] = tokens[index + 1]
            pending = []
            index += 2
        else:
            if not isinstance(tokens[index], str):
                raise UnsupportedPDDL("Compound typed symbol")
            pending.append(tokens[index])
            index += 1
    result.update({name: "object" for name in pending})
    return result


def _literals(expression: list, binding: dict) -> tuple[set, set]:
    if not expression:
        return set(), set()
    terms = expression[1:] if expression[0] == "and" else [expression]
    positive, negative = set(), set()
    for term in terms:
        if not isinstance(term, list) or not term:
            raise UnsupportedPDDL("Malformed literal")
        negate = term[0] == "not"
        if negate and len(term) != 2:
            raise ValueError("Negation must contain exactly one atom")
        atom = term[1] if negate and len(term) == 2 else term
        if not isinstance(atom, list) or not atom or any(isinstance(x, list) for x in atom):
            raise UnsupportedPDDL("Only flat positive/negative atoms are supported")
        if atom[0] in {"or", "forall", "exists", "when", "increase", "decrease", "assign", ">", "<"}:
            raise UnsupportedPDDL(f"Unsupported PDDL expression: {atom[0]}")
        instantiated = tuple(binding.get(v, v) for v in atom)
        if any(x.startswith("?") for x in instantiated):
            raise ValueError("Unbound action variable")
        (negative if negate else positive).add(instantiated)
    return positive, negative


def planbench_strips_score(plan: str, domain_pddl: str, problem_pddl: str) -> ScoreResult:
    domain, problem = parse_pddl(domain_pddl), parse_pddl(problem_pddl)
    if domain[0] != "define" or problem[0] != "define":
        raise ValueError("Expected PDDL define forms")
    sections = {s[0]: s[1:] for s in problem[1:] if isinstance(s, list) and s}
    domain_names = [s[1] for s in domain[1:] if isinstance(s, list) and s and s[0] == "domain"]
    if len(domain_names) != 1 or sections.get(":domain") != domain_names:
        raise ValueError("Problem and evaluator domain identities disagree")
    objects = _typed_symbols(sections.get(":objects", []))
    actions = {}
    type_parents = {"object": "object"}
    for section in domain[1:]:
        if not isinstance(section, list) or not section:
            continue
        if section[0] == ":requirements":
            unsupported = set(section[1:]) - {":strips", ":typing", ":negative-preconditions", ":equality"}
            if unsupported:
                raise UnsupportedPDDL(f"Unsupported requirements: {sorted(unsupported)}")
        if section[0] == ":types":
            type_parents.update(_typed_symbols(section[1:]))
        if section[0] == ":functions" and len(section) == 1:
            continue
        if section[0] not in {"domain", ":requirements", ":types", ":constants", ":predicates", ":action"}:
            raise UnsupportedPDDL(f"Unsupported domain section {section[0]}")
        if section[0] == ":constants":
            objects.update(_typed_symbols(section[1:]))
        if section[0] == ":action":
            if len(section[2:]) % 2:
                raise ValueError("Malformed action definition")
            fields = dict(zip(section[2::2], section[3::2]))
            if set(fields) - {":parameters", ":precondition", ":effect"}:
                raise UnsupportedPDDL("Unsupported action fields")
            actions[section[1]] = fields
    state, negative_initial = _literals(["and", *sections.get(":init", [])], {})
    if negative_initial:
        raise UnsupportedPDDL("Explicit negative initial literals are unsupported")
    goals = sections.get(":goal")
    if not goals or len(goals) != 1:
        raise ValueError("Missing single goal expression")
    goal_pos, goal_neg = _literals(goals[0], {})
    def holds(atom):
        if atom[0] == "=":
            if len(atom) != 3:
                raise ValueError("Equality requires two arguments")
            return atom[1] == atom[2]
        return atom in state
    def satisfied(positive, negative):
        return all(holds(a) for a in positive) and not any(holds(a) for a in negative)
    def subtype(actual, expected):
        seen = set()
        while actual not in seen:
            if actual == expected or expected == "object":
                return True
            seen.add(actual)
            if actual not in type_parents:
                raise ValueError(f"Unknown declared PDDL type: {actual}")
            actual = type_parents[actual]
        return False
    # Validation expects explicit actions, not natural-language heuristics.
    clean = re.sub(r";[^\n]*", "", plan.lower()).strip()
    tokens = re.findall(r"\([^()]+\)", clean)
    residue = re.sub(r"\([^()]+\)", "", clean).strip()
    if residue:
        return ScoreResult(0.0, {"valid_plan": False, "reason": "Use only parenthesized PDDL actions"})
    for step, token in enumerate(tokens):
        items = token[1:-1].split()
        if not items or items[0] not in actions:
            return ScoreResult(0.0, {"valid_plan": False, "failed_step": step, "reason": "unknown action"})
        action = actions[items[0]]
        parameters = _typed_symbols(action.get(":parameters", []))
        if len(items) - 1 != len(parameters):
            return ScoreResult(0.0, {"valid_plan": False, "failed_step": step, "reason": "arity mismatch"})
        for argument, parameter_type in zip(items[1:], parameters.values()):
            if argument not in objects or not subtype(objects[argument], parameter_type):
                return ScoreResult(0.0, {"valid_plan": False, "failed_step": step, "reason": "unknown/incorrectly typed object"})
        binding = dict(zip(parameters, items[1:]))
        pre_pos, pre_neg = _literals(action.get(":precondition", []), binding)
        if not satisfied(pre_pos, pre_neg):
            return ScoreResult(0.0, {"valid_plan": False, "failed_step": step, "reason": "unsatisfied precondition"})
        add, delete = _literals(action.get(":effect", []), binding)
        if any(atom[0] == "=" for atom in add | delete):
            raise UnsupportedPDDL("Equality is immutable and cannot be an effect")
        state = (state - delete) | add
    valid = satisfied(goal_pos, goal_neg)
    return ScoreResult(float(valid), {"valid_plan": bool(valid), "steps": len(tokens), "native_unit": "strips_plan_validity"})


def planbench_example_score(example, prediction: str, *, root=".") -> ScoreResult:
    """Score pinned PlanBench task 1, including official object name encoding."""
    from pathlib import Path
    import yaml
    if example.metadata.get("task") != "task_1_plan_generation":
        raise MissingArtifactError("Only native plan-generation validity is registered; other tasks need their own native metric")
    source = Path(root) / example.metadata["source_file"]
    benchmark_root = source.parents[2]
    config = yaml.safe_load((benchmark_root / "configs" / f"{example.metadata['domain']}.yaml").read_text())
    domain = benchmark_root / "instances" / config["domain_file"]
    problem = benchmark_root / "instances" / config["instance_dir"] / config["instances_template"].format(example.reference["instance_id"])
    # Accept native symbolic plans and the benchmark's exact controlled-English
    # action templates. Arbitrary prose is not interpreted as executable actions.
    value = prediction.strip()
    if "[PLAN]" in value:
        value = value.rsplit("[PLAN]", 1)[-1].split("[PLAN END]", 1)[0].strip()
    encoded = config.get("encoded_objects", {})
    aliases = {str(description).lower(): str(symbol) for symbol, description in encoded.items()}
    aliases.update({str(description).lower().removesuffix(" block"): str(symbol) for symbol, description in encoded.items()})
    converted = []
    for line in value.lower().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("(") and line.endswith(")"):
            parts = line[1:-1].split()
            converted.append("(" + " ".join([parts[0], *(aliases.get(p, p) for p in parts[1:])]) + ")")
            continue
        matched = False
        for action, template in config.get("actions", {}).items():
            pattern = "(.*?)".join(re.escape(piece) for piece in template.lower().split("{}"))
            match = re.fullmatch(pattern, line.rstrip("."))
            if match:
                arguments = [aliases.get(argument, argument) for argument in match.groups()]
                converted.append("(" + " ".join([action, *arguments]) + ")")
                matched = True
                break
        if not matched:
            return ScoreResult(0.0, {"valid_plan": False, "reason": "Unrecognized action text"})
    return planbench_strips_score("\n".join(converted), domain.read_text(), problem.read_text())
