"""Targeted repair (Sec. 3.4): map each reported issue type to a concrete action for
the next attempt — prompt augmentation, reference change, or reseed."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..llm.prompts import REPAIR_HINTS
from ..schema import Entity, IssueType, ShotSpec, Verdict


@dataclass
class RepairPlan:
    extra_prompt: str = ""
    reseed: bool = False
    reselect_references: bool = False
    strengthen_references: bool = False   # drop auxiliaries, keep canonical only
    actions: list[str] = field(default_factory=list)


def plan_repair(verdict: Verdict, shot: ShotSpec, entities: list[Entity],
                style_layer: str) -> RepairPlan:
    plan = RepairPlan()
    hints: list[str] = []
    char_names = ", ".join(e.name for e in entities if e.entity_type.value == "character")
    fmt = dict(
        expected_count=shot.expected_char_count or len([e for e in entities
                                                        if e.entity_type.value == "character"]),
        char_names=char_names or "the described subjects",
        style_layer=style_layer,
        entity_layer=shot.prompt.entities if shot.prompt else "",
        action_layer=shot.prompt.action if shot.prompt else shot.text,
        entity_name="",
    )
    for issue in verdict.issues:
        it = issue.issue_type
        plan.actions.append(it.value)
        if it == IssueType.IDENTITY_MISMATCH:
            plan.strengthen_references = True
            plan.reselect_references = True
        elif it in (IssueType.MOTION_ARTIFACT, IssueType.RENDERING_ARTIFACT,
                    IssueType.QUALITY_DEGRADATION, IssueType.UNNATURAL_POSE):
            plan.reseed = True
        elif it.value in REPAIR_HINTS:
            f = dict(fmt)
            if issue.entity_id:
                match = [e for e in entities if e.entity_id == issue.entity_id]
                f["entity_name"] = match[0].name if match else issue.entity_id
            hints.append(REPAIR_HINTS[it.value].format(**f))
    if not verdict.issues and not verdict.passed:
        plan.reseed = True   # low score with no structured issue -> try a new seed
        plan.actions.append("low_score_reseed")
    plan.extra_prompt = " ".join(dict.fromkeys(hints))   # dedupe, keep order
    return plan
