"""Persistent root-lineage partitions shared by SFT, GVO and every epoch."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class LineageSplitRegistry:
    def __init__(self, path, *, validation_fraction=.2, seed=0):
        if not 0 <= validation_fraction < 1:
            raise ValueError("Invalid lineage validation fraction")
        self.path = Path(path)
        self.settings = {"schema": "stable_lineage_split_v1", "validation_fraction": validation_fraction, "seed": seed}
        self.assignments = {}
        if self.path.exists():
            saved = json.loads(self.path.read_text())
            if any(saved.get(key) != value for key, value in self.settings.items()):
                raise ValueError("A registered lineage split cannot change between epochs or stages")
            self.assignments = saved["assignments"]

    def assign(self, lineage, *, audit_only=False):
        if not lineage:
            raise ValueError("Every collected record requires its real root lineage")
        existing = self.assignments.get(lineage)
        if audit_only:
            if existing == "train":
                raise ValueError("Cannot declare a previously trained lineage untouched audit data")
            target = "audit"
        elif existing:
            return existing
        else:
            key = f"{self.settings['seed']}:{lineage}".encode()
            uniform = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
            target = "validation" if uniform < self.settings["validation_fraction"] else "train"
        self.assignments[lineage] = target
        return target

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps({**self.settings, "assignments": self.assignments}, indent=2, sort_keys=True))
        temporary.replace(self.path)

    def split_records(self, records):
        # Protect known audit roots before assigning any ordinary record from
        # that root, even if the audit row occurs last in the supplied file.
        for record in records:
            if record.get("audit_only", False):
                self.assign(record["lineage_id"], audit_only=True)
        subsets = {key: [] for key in ("train", "validation", "audit")}
        for record in records:
            subsets[self.assign(record["lineage_id"])].append(record)
        self.save()
        return tuple(subsets[key] for key in ("train", "validation", "audit"))

    def split_collection(self, collection):
        groups = collection["groups"]
        forced = collection.get("forced_trajectories", ())
        pairs = collection.get("pairs", ())
        all_records = [t for group in groups for t in group] + list(forced) + list(pairs) + list(collection.get("audits", ()))
        for item in all_records:
            if item.audit_only:
                self.assign(item.lineage_id, audit_only=True)
        train_groups, heldout = [], []
        for group in groups:
            if len({t.lineage_id for t in group}) != 1:
                raise ValueError("Independent searches in a group must share one root lineage")
            target = self.assign(group[0].lineage_id)
            if target == "train":
                train_groups.append(group)
            elif target == "validation":
                heldout.extend(group)
        result = {**collection, "groups": train_groups}
        for key in ("forced_trajectories", "pairs", "audits"):
            result[key] = [item for item in collection.get(key, ()) if self.assign(item.lineage_id) == "train"]
        self.save()
        return result, heldout
