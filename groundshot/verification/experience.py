"""Generation experience X (Sec. 3.4): a persistent JSONL log of what happened per
shot — references used, issues reported, repairs attempted and their outcomes.
Retrieval is lightweight keyword/type matching over past records; matched records
become textual advice eta_i for prompt building and repair ordering, and flag
high-risk shots for risk-gated verification.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


class ExperienceStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[dict] = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self.records.append(json.loads(line))

    def record(self, *, script_id: str, shot_id: int, shot_type: str,
               entity_types: list[str], mode: str, attempts: int,
               issues: list[str], repairs: list[str], passed: bool,
               final_quality: float) -> None:
        rec = dict(ts=time.time(), script_id=script_id, shot_id=shot_id,
                   shot_type=shot_type, entity_types=sorted(set(entity_types)),
                   mode=mode, attempts=attempts, issues=issues, repairs=repairs,
                   passed=passed, final_quality=final_quality)
        self.records.append(rec)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def retrieve(self, shot_type: str, entity_types: list[str], k: int = 5) -> list[dict]:
        """Most similar past records: same shot type and overlapping entity types."""
        et = set(entity_types)

        def score(r: dict) -> float:
            s = len(et & set(r.get("entity_types", []))) / (len(et) + 1e-6)
            if r.get("shot_type") == shot_type:
                s += 0.5
            return s

        ranked = sorted(self.records, key=score, reverse=True)
        return [r for r in ranked[:k] if score(r) > 0.3]

    def advice(self, shot_type: str, entity_types: list[str]) -> tuple[str, bool]:
        """Returns (textual advice for the prompt, high_risk flag)."""
        similar = self.retrieve(shot_type, entity_types)
        if not similar:
            return "", False
        fail_rate = sum(1 for r in similar if not r.get("passed", True)) / len(similar)
        issue_counts: dict[str, int] = {}
        for r in similar:
            for i in r.get("issues", []):
                issue_counts[i] = issue_counts.get(i, 0) + 1
        common = sorted(issue_counts, key=issue_counts.get, reverse=True)[:2]
        # These strings are appended verbatim to the VIDEO GENERATION prompt, so they
        # must read as prompt constraints, not as instructions to a prompt writer.
        hints = {
            "count_error": "Exactly the stated number of people, nobody else in frame.",
            "identity_mismatch": "Each subject matches the described appearance and wardrobe exactly.",
            "style_drift": "One uniform visual style throughout the shot.",
            "missing_entity": "Every described subject clearly visible in frame.",
        }
        advice = " ".join(hints[c] for c in common if c in hints)
        return advice, fail_rate >= 0.5

    def repair_order(self, issue_types: list[str]) -> list[str]:
        """Order repair actions by past success rate for the same issue type."""
        stats: dict[str, list[int]] = {}
        for r in self.records:
            for i in r.get("issues", []):
                stats.setdefault(i, []).append(1 if r.get("passed") else 0)
        return sorted(issue_types,
                      key=lambda t: -(sum(stats.get(t, [0])) / max(len(stats.get(t, [1])), 1)))
