"""Statistical analysis of paired run records."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Sequence
import numpy as np


@dataclass(frozen=True)
class PairedEstimate:
    mean_difference: float
    interval: tuple[float, float]
    family_interval: tuple[float, float]
    independent_units: int
    bootstrap_resamples: int
    family_size: int
    expected_samples_per_family_tail: float
    seed: int


def paired_bootstrap(left: Sequence[float], right: Sequence[float], *,
                     clusters: Sequence[str] | None = None, resamples: int = 10000,
                     family_size: int = 13, alpha: float = 0.05, seed: int = 0) -> PairedEstimate:
    """Equal-weight independent units; average nested runs within training replica first.

    The caller must align paired records by identity, not by incidental file order.
    Percent-valued inputs yield percentage-point differences; fractions stay fractions.
    """
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 2 or not np.isfinite([a, b]).all():
        raise ValueError("Need at least two aligned finite paired observations")
    if (type(resamples) is not int or type(family_size) is not int
            or resamples < 1 or family_size < 1 or not 0 < alpha < 1):
        raise ValueError("Invalid bootstrap settings")
    differences = a - b
    if clusters is not None:
        if len(clusters) != len(differences):
            raise ValueError("Cluster identities must align with pairs")
        grouped = defaultdict(list)
        for key, value in zip(clusters, differences):
            grouped[str(key)].append(value)
        differences = np.array([np.mean(grouped[key]) for key in sorted(grouped)])
    if len(differences) < 2:
        raise ValueError("Need at least two independent clusters")
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples)
    for start in range(0, resamples, 1000):
        size = min(1000, resamples - start)
        indexes = rng.integers(0, len(differences), size=(size, len(differences)))
        samples[start:start + size] = differences[indexes].mean(axis=1)
    ci = tuple(float(x) for x in np.quantile(samples, [alpha / 2, 1 - alpha / 2]))
    adjusted = tuple(float(x) for x in np.quantile(samples, [alpha / (2 * family_size), 1 - alpha / (2 * family_size)]))
    return PairedEstimate(float(differences.mean()), ci, adjusted, len(differences), resamples,
                          family_size, resamples * alpha / (2 * family_size), seed)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm's step-down correction across the complete registered hypothesis family."""
    values = np.asarray(p_values, dtype=float)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or np.any((values < 0) | (values > 1))):
        raise ValueError("Need finite p-values in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, float(values[index]) * (len(values) - rank))
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def paired_sign_flip(differences: Sequence[float], *, resamples: int = 10000,
                     seed: int = 0) -> dict:
    """One-sided test of a positive mean over independent paired seed differences.

    Enumerates every assignment up to 16 independent seeds; larger samples use
    Monte Carlo signs with a +1 correction. This test assumes exchangeability
    of method labels within independent training-seed pairs under the null;
    nesting searches within a seed does not create additional independent units.
    """
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Need at least two finite independent seed differences")
    if type(resamples) is not int or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    observed = float(values.mean())
    tolerance = np.finfo(float).eps * max(1.0, float(np.abs(values).mean())) * 16
    exact = len(values) <= 16
    draws = (1 << len(values)) if exact else resamples
    rng = np.random.default_rng(seed)
    extreme = 0
    for start in range(0, draws, 1024):
        count = min(1024, draws - start)
        if exact:
            assignments = np.arange(start, start + count, dtype=np.uint64)
            bits = (assignments[:, None] >> np.arange(len(values), dtype=np.uint64)) & 1
            signs = bits.astype(float) * 2 - 1
        else:
            signs = rng.integers(0, 2, size=(count, len(values))) * 2 - 1
        extreme += int(np.sum((signs * values).mean(axis=1) >= observed - tolerance))
    return {"p_value": extreme / draws if exact else (extreme + 1) / (draws + 1),
            "exact": exact, "assignments": draws, "independent_units": len(values),
            "alternative": "candidate_better", "assumption": "paired_label_exchangeability"}


def validate_comparison_registration(registration: dict) -> dict:
    """Validate a prospective protocol before acquiring any model observations."""
    if registration.get("schema_version") != 1:
        raise ValueError("Unsupported comparison registration schema")
    candidate = registration.get("candidate")
    if not isinstance(candidate, str) or not candidate:
        raise ValueError("Register a nonempty candidate method")
    profiles = registration.get("profiles", [])
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("Register at least one comparison profile")
    smoke = registration.get("engineering_smoke_only", False)
    if type(smoke) is not bool:
        raise ValueError("engineering_smoke_only must be a boolean")
    alpha = registration.get("alpha", .05)
    resamples = registration.get("bootstrap_resamples", 10000)
    seed = registration.get("random_seed", 0)
    if (not isinstance(alpha, (int, float)) or not 0 < alpha < 1
            or type(resamples) is not int or resamples < 1000 or type(seed) is not int or seed < 0):
        raise ValueError("Invalid registered inference settings (at least 1000 resamples required)")
    required_contract = {"dataset", "model_revision", "scorer", "permission_band", "endpoint",
                         "evaluation_split", "selection_manifest_sha256", "evaluation_manifest_sha256",
                         "initial_artifact_sha256", "execution_contract_sha256", "resource_profile", "budget"}
    by_profile = {}
    expected = set()
    family_size = 0
    for profile in profiles:
        name = profile.get("id")
        if not isinstance(name, str) or not name or name in by_profile:
            raise ValueError("Profile IDs must be nonempty and unique")
        baselines = profile.get("baselines", [])
        if (not isinstance(baselines, list) or not baselines
                or any(not isinstance(value, str) or not value for value in baselines)
                or len(set(baselines)) != len(baselines) or candidate in baselines):
            raise ValueError(f"Invalid baseline registration in {name}")
        training = profile.get("training_seeds", [])
        searches = profile.get("search_seeds", [])
        for keys, minimum, label in ((training, 1 if smoke else 2, "training"), (searches, 1, "search")):
            if (not isinstance(keys, list) or len(keys) < minimum
                    or any(type(value) is not int for value in keys) or len(set(keys)) != len(keys)):
                raise ValueError(f"Register distinct integer {label} seeds in {name}")
        if profile.get("direction") not in {"higher", "lower"}:
            raise ValueError(f"Register metric direction in {name}")
        margin = profile.get("minimum_effect", 0.0)
        if (not isinstance(margin, (int, float)) or isinstance(margin, bool)
                or not np.isfinite(margin) or margin < 0):
            raise ValueError("minimum_effect must be finite and nonnegative")
        contract = profile.get("contract")
        if not isinstance(contract, dict) or not required_contract <= contract.keys():
            raise ValueError(f"Incomplete matched-comparison contract in {name}")
        if any(not isinstance(contract[key], str) or not contract[key]
               for key in required_contract - {"budget"}):
            raise ValueError(f"Contract identities must be nonempty strings in {name}")
        for field in ("selection_manifest_sha256", "evaluation_manifest_sha256",
                      "initial_artifact_sha256", "execution_contract_sha256"):
            value = contract[field]
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"Invalid manifest sha256 in {name}")
        if contract["evaluation_split"] not in {"selection", "validation", "audit"}:
            raise ValueError(f"Unknown evaluation split in {name}")
        if (contract["evaluation_split"] == "audit"
                and contract["selection_manifest_sha256"] == contract["evaluation_manifest_sha256"]):
            raise ValueError("An audit must declare a different split from selection")
        budget = contract["budget"]
        if (not isinstance(budget, dict) or not {"cap", "unit", "accounting"} <= budget.keys()
                or not isinstance(budget["cap"], (int, float)) or isinstance(budget["cap"], bool)
                or not np.isfinite(budget["cap"]) or budget["cap"] <= 0
                or not isinstance(budget["unit"], str) or not budget["unit"]
                or not isinstance(budget["accounting"], str) or not budget["accounting"]):
            raise ValueError(f"Invalid budget contract in {name}")
        by_profile[name] = profile
        family_size += len(baselines)
        expected.update((name, method, train, search) for method in [candidate, *baselines]
                        for train in training for search in searches)
    return {"candidate": candidate, "profiles": profiles, "alpha": alpha, "resamples": resamples,
            "seed": seed, "by_profile": by_profile, "expected": expected, "family_size": family_size,
            "engineering_smoke_only": smoke}


def compare_registered_methods(registration: dict, records: Sequence[dict]) -> dict:
    """Compare all registered paired runs, without dropping incomplete pairs.

    Raw artifact verification is performed by scripts/compare_methods.py.
    Pair membership, training-seed clusters, and the comparison family are fixed
    by registration. Smoke registrations bypass inferential analysis.
    """
    settings = validate_comparison_registration(registration)
    if settings["engineering_smoke_only"]:
        raise ValueError("Engineering smoke registrations cannot support statistical inference")
    candidate, profiles = settings["candidate"], settings["profiles"]
    alpha, resamples, seed = settings["alpha"], settings["resamples"], settings["seed"]
    by_profile, expected, family_size = settings["by_profile"], settings["expected"], settings["family_size"]
    indexed = {}
    run_ids = set()
    for row in records:
        key = (row.get("profile_id"), row.get("method"), row.get("training_seed"), row.get("search_seed"))
        if type(key[2]) is not int or type(key[3]) is not int or key not in expected:
            raise ValueError(f"Unregistered run identity: {key}")
        if key in indexed:
            raise ValueError(f"Duplicate run identity: {key}")
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id or (key[0], run_id) in run_ids:
            raise ValueError("Every run needs a unique nonempty run_id within its profile")
        run_ids.add((key[0], run_id))
        contract = by_profile[key[0]]["contract"]
        if row.get("contract") != contract:
            raise ValueError(f"Unmatched evaluation/budget/permission contract: {key}")
        if row.get("status") != "complete":
            raise ValueError(f"Incomplete run cannot be silently omitted: {key}")
        for field in ("score", "actual_cost"):
            if (not isinstance(row.get(field), (int, float)) or isinstance(row[field], bool)
                    or not np.isfinite(row[field])):
                raise ValueError(f"Need a finite observed {field}: {key}")
        if not 0 <= row["actual_cost"] <= contract["budget"]["cap"]:
            raise ValueError(f"Run exceeds the registered budget: {key}")
        indexed[key] = row
    missing = expected - indexed.keys()
    if missing:
        raise ValueError(f"Missing {len(missing)} registered runs; refusing selective complete-case analysis: {sorted(missing)[:5]}")
    # Corrected percentile tails need enough Monte Carlo mass to be resolved.
    # Increase simulation effort, never the number of claimed experimental units.
    effective_resamples = max(resamples, int(np.ceil(40 * family_size / alpha)))
    comparisons = []
    for profile in profiles:
        name = profile["id"]
        for baseline in profile["baselines"]:
            training = sorted(profile["training_seeds"])
            searches = sorted(profile["search_seeds"])
            left = [float(np.mean([indexed[name, candidate, train, search]["score"] for search in searches]))
                    for train in training]
            right = [float(np.mean([indexed[name, baseline, train, search]["score"] for search in searches]))
                     for train in training]
            direction = 1 if profile["direction"] == "higher" else -1
            estimate = paired_bootstrap([direction * value for value in left],
                                        [direction * value for value in right],
                                        resamples=effective_resamples, family_size=family_size, alpha=alpha, seed=seed)
            margin = profile.get("minimum_effect", 0.0)
            test = paired_sign_flip([direction * (a - b) - margin for a, b in zip(left, right)],
                                    resamples=resamples, seed=seed)
            comparisons.append({"profile_id": name, "baseline": baseline,
                                "candidate_mean": float(np.mean(left)), "baseline_mean": float(np.mean(right)),
                                "candidate_sample_sd": float(np.std(left, ddof=1)),
                                "baseline_sample_sd": float(np.std(right, ddof=1)),
                                "paired_difference_sample_sd": float(np.std(np.asarray(left) - np.asarray(right), ddof=1)),
                                "direction": profile["direction"], "minimum_effect": margin,
                                "seed_means": {"training_seeds": training, "candidate": left, "baseline": right},
                                "nested_searches_per_seed": len(searches), "estimate": asdict(estimate), "test": test})
    adjusted = holm_adjust([row["test"]["p_value"] for row in comparisons])
    for row, p_value in zip(comparisons, adjusted):
        row["test"]["holm_p_value"] = p_value
        row["superiority_supported"] = bool(
            row["estimate"]["family_interval"][0] > row["minimum_effect"] and p_value <= alpha)
    return {"schema_version": 1, "candidate": candidate, "family_size": family_size, "alpha": alpha,
            "requested_bootstrap_resamples": resamples, "effective_bootstrap_resamples": effective_resamples,
            "all_registered_comparisons_supported": all(row["superiority_supported"] for row in comparisons),
            "scope": "Only the registered methods, profiles, contracts, and supplied run records; not a global SOTA claim.",
            "audit_isolation": "Split identities are checked; temporal isolation and absence of tuning on audit require independent provenance review.",
            "run_provenance": "Declared training-seed independence, baseline fidelity, and full cost accounting require independent provenance review.",
            "inference": "Equal-weight training-seed means; Bonferroni percentile bootstrap intervals and Holm-adjusted one-sided paired sign-flip tests.",
            "comparisons": comparisons}


def aubc(events: Sequence[tuple[float, float]], cap: float, *, normalize: bool = False) -> float:
    """Integral of the saved complete incumbent; right-continuous step interpolation."""
    if not events or events[0][0] != 0 or cap <= 0:
        raise ValueError("An initial complete incumbent at cost zero and positive cap are required")
    array = np.asarray(events, dtype=float)
    if not np.isfinite(array).all() or np.any(np.diff(array[:, 0]) < 0) or array[-1, 0] > cap:
        raise ValueError("Invalid charged event trajectory")
    area = float(np.sum(np.diff(np.append(array[:, 0], cap)) * array[:, 1]))
    return area / cap if normalize else area


def calibration(probabilities: Sequence[float], labels: Sequence[int], bins: int = 10) -> dict:
    p, y = np.asarray(probabilities, dtype=float), np.asarray(labels)
    if p.ndim != 1 or p.shape != y.shape or not len(p) or not np.isfinite(p).all():
        raise ValueError("Need aligned completed binary audits and pre-check probabilities")
    if np.any((p < 0) | (p > 1)) or not np.isin(y, [0, 1]).all() or bins < 1:
        raise ValueError("Interrupted or unexecuted checker attempts have no binary labels")
    groups = np.minimum((p * bins).astype(int), bins - 1)
    ece = sum(float((groups == j).mean()) * abs(float(p[groups == j].mean() - y[groups == j].mean()))
              for j in range(bins) if np.any(groups == j))
    bounded = np.clip(p, 1e-12, 1 - 1e-12)
    return {"n_completed": len(p), "brier": float(np.mean((p - y) ** 2)), "ece": ece,
            "log_loss": float(-np.mean(y * np.log(bounded) + (1 - y) * np.log(1 - bounded))),
            "false_P_forecast": int(np.sum((p >= .5) & (y == 0))),
            "false_U_forecast": int(np.sum((p < .5) & (y == 1)))}


def cost_endpoints(actual_paid: float, cap: float, hit_cost: float | None) -> dict:
    if actual_paid < 0 or cap <= 0 or not np.isfinite([actual_paid, cap]).all():
        raise ValueError("Costs must be finite and nonnegative")
    if hit_cost is not None and (not np.isfinite(hit_cost) or not 0 <= hit_cost <= min(actual_paid, cap)):
        raise ValueError("A hit requires a physical cost event within the cap")
    return {"actual_C": actual_paid, "hit_H": int(hit_cost is not None),
            "deadline_L": cap if hit_cost is None else hit_cost}
