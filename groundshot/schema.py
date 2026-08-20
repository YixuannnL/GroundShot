"""Core data structures shared across the GroundShot pipeline.

Terminology follows the paper:
  - Entity: a recurring character / object / location with a stable ID.
  - ShotSpec: one shot of the script with its parsed entity set and layered prompt.
  - Reference: one admitted visual reference (crop or scene image) in memory.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional


class EntityType(str, Enum):
    CHARACTER = "character"
    OBJECT = "object"
    LOCATION = "location"


# Scheduling priority: characters > objects > locations (Sec. 3.2).
ENTITY_PRIORITY = {
    EntityType.CHARACTER: 3,
    EntityType.OBJECT: 2,
    EntityType.LOCATION: 1,
}


@dataclass
class Entity:
    entity_id: str                 # e.g. "char_alex", "obj_briefcase", "loc_station"
    entity_type: EntityType
    name: str                      # short grounding-friendly name, e.g. "bald man in tan jacket"
    description: str               # full appearance description used for prompts & grounding
    aliases: list[str] = field(default_factory=list)
    priority: int = 0              # filled from ENTITY_PRIORITY at parse time
    gender: str | None = None      # characters only: "male"/"female" when the script states it

    def to_dict(self) -> dict:
        d = asdict(self)
        d["entity_type"] = self.entity_type.value
        return d

    @staticmethod
    def from_dict(d: dict) -> "Entity":
        d = dict(d)
        d["entity_type"] = EntityType(d["entity_type"])
        return Entity(**d)


@dataclass
class LayeredPrompt:
    """Layered prompt (Sec. 3.4): global style layer + entity layer + shot action layer."""
    style: str                     # style / lighting / tone cues distilled from global caption
    entities: str                  # descriptions of ONLY the entities present in this shot
    action: str                    # the shot text itself (what happens, camera, framing)

    def compose(self, extra: str = "") -> str:
        parts = [p.strip() for p in (self.style, self.entities, self.action, extra) if p and p.strip()]
        return " ".join(parts)


@dataclass
class ShotSpec:
    shot_id: int                   # narrative index (1-based)
    text: str
    entity_ids: list[str] = field(default_factory=list)
    prompt: Optional[LayeredPrompt] = None
    duration: float = 4.0
    shot_type: str = ""            # close-up / medium / wide / ...
    expected_char_count: Optional[int] = None   # explicit or strongly implied character count
    # qsrc(s_i, e) per entity (Eq. 1), filled by the parser's reference-quality estimator
    ref_quality: dict[str, float] = field(default_factory=dict)


@dataclass
class ParsedScript:
    script_id: str
    entities: dict[str, Entity]
    shots: list[ShotSpec]
    global_caption: str = ""
    style_layer: str = ""          # extracted global visual context (Supp. 6.1)
    seed: Optional[int] = None

    def entities_of(self, shot: ShotSpec) -> list[Entity]:
        return [self.entities[eid] for eid in shot.entity_ids if eid in self.entities]

    def save(self, path: Path) -> None:
        data = {
            "script_id": self.script_id,
            "global_caption": self.global_caption,
            "style_layer": self.style_layer,
            "seed": self.seed,
            "entities": {k: v.to_dict() for k, v in self.entities.items()},
            "shots": [
                {**asdict(s), "prompt": asdict(s.prompt) if s.prompt else None}
                for s in self.shots
            ],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    @staticmethod
    def load(path: Path) -> "ParsedScript":
        data = json.loads(path.read_text())
        shots = []
        for s in data["shots"]:
            p = s.pop("prompt", None)
            shot = ShotSpec(**s)
            if p:
                shot.prompt = LayeredPrompt(**p)
            shots.append(shot)
        return ParsedScript(
            script_id=data["script_id"],
            entities={k: Entity.from_dict(v) for k, v in data["entities"].items()},
            shots=shots,
            global_caption=data.get("global_caption", ""),
            style_layer=data.get("style_layer", ""),
            seed=data.get("seed"),
        )


@dataclass
class Reference:
    """One admitted visual reference in entity-level memory."""
    entity_id: str
    entity_type: EntityType
    image_path: str                # local path of the crop / scene image
    quality: float                 # composite q(c) in [0,1]
    source_shot: int               # narrative shot id it was grounded from
    is_canonical: bool = False
    is_provisional: bool = False   # provisional canonical (our deadlock fix; see README)
    face_conf: float = 0.0
    frontality: float = 0.0        # 1 - |yaw|/90, characters only
    sharpness: float = 0.0
    meta: dict = field(default_factory=dict)   # viewpoint/lighting/pose tags for retrieval
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["entity_type"] = self.entity_type.value
        return d

    @staticmethod
    def from_dict(d: dict) -> "Reference":
        d = dict(d)
        d["entity_type"] = EntityType(d["entity_type"])
        return Reference(**d)


class IssueType(str, Enum):
    """Structured issues reported by the VLM critic (Sec. 3.4)."""
    IDENTITY_MISMATCH = "identity_mismatch"
    COUNT_ERROR = "count_error"
    MISSING_ENTITY = "missing_entity"
    EXTRA_ENTITY = "extra_entity"
    CLOTHING_MISMATCH = "clothing_mismatch"
    STYLE_DRIFT = "style_drift"
    LIGHTING_INCONSISTENCY = "lighting_inconsistency"
    UNNATURAL_POSE = "unnatural_pose"
    MOTION_ARTIFACT = "motion_artifact"
    RENDERING_ARTIFACT = "rendering_artifact"
    PROMPT_MISMATCH = "prompt_mismatch"
    QUALITY_DEGRADATION = "quality_degradation"


SEVERE_ISSUES = {
    IssueType.IDENTITY_MISMATCH,
    IssueType.COUNT_ERROR,
    IssueType.MISSING_ENTITY,
    IssueType.MOTION_ARTIFACT,
    IssueType.RENDERING_ARTIFACT,
}


@dataclass
class CriticIssue:
    issue_type: IssueType
    severity: str                  # "minor" | "severe"
    detail: str = ""
    entity_id: str = ""


@dataclass
class Verdict:
    quality: float                 # Q(v) in [0,1]
    issues: list[CriticIssue] = field(default_factory=list)
    passed: bool = True
    raw: dict = field(default_factory=dict)

    @property
    def severe_issues(self) -> list[CriticIssue]:
        return [i for i in self.issues if i.severity == "severe" or i.issue_type in SEVERE_ISSUES]


@dataclass
class ShotResult:
    shot_id: int
    video_path: str
    mode: str                      # "t2v" | "ref2v"
    attempts: int
    verdict: Optional[Verdict] = None
    references_used: list[str] = field(default_factory=list)   # image paths
    prompt_used: str = ""
    gen_seed: Optional[int] = None
