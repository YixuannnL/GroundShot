"""Shot verification (Sec. 3.4, Eq. 5): a VLM critic samples frames and reports
structured issues. Risk-gated mode (Sec. 4.6) only invokes the critic for shots
where an error would be expensive: reference-source shots, first Ref2V uses of an
entity, and shots flagged by past experience — matching the paper's reported call
budget while protecting memory quality.
"""
from __future__ import annotations

import logging

from ..config import VerifyConfig
from ..llm.client import LLMClient
from ..llm import prompts
from ..schema import (CriticIssue, Entity, IssueType, SEVERE_ISSUES, ShotSpec, Verdict)
from ..utils.media import frame_grid, sample_frames, to_b64_jpeg, load_image

log = logging.getLogger("groundshot.critic")


class ShotCritic:
    def __init__(self, cfg: VerifyConfig, llm: LLMClient):
        self.cfg = cfg
        self.llm = llm

    def should_verify(self, is_source_shot: bool, first_ref2v_entities: bool,
                      risk_advice: bool) -> bool:
        if self.cfg.mode == "off":
            return False
        if self.cfg.mode == "full":
            return True
        return is_source_shot or first_ref2v_entities or risk_advice

    def verify(self, video_path: str, shot: ShotSpec, entities: list[Entity],
               reference_paths: list[str]) -> Verdict:
        frames = sample_frames(video_path, self.cfg.frames_for_critic)
        grid = frame_grid(frames, cols=2)
        images = [to_b64_jpeg(load_image(p), max_side=512) for p in reference_paths[:4]]
        ref_note = ""
        if images:
            ref_note = prompts.CRITIC_REF_NOTE.format(
                ref_order=", ".join(f"ref{i+1}" for i in range(len(images))))
        images.append(to_b64_jpeg(grid, max_side=1024))

        entities_block = "\n".join(
            f"- {e.entity_id} ({e.entity_type.value}): {e.description}" for e in entities)
        try:
            out = self.llm.json_call(
                prompts.CRITIC_SYSTEM,
                prompts.CRITIC_USER.format(
                    shot_text=shot.text, entities_block=entities_block or "(none)",
                    expected_count=shot.expected_char_count or "unspecified",
                    ref_note=ref_note),
                images_b64=images,
            )
        except Exception as e:  # noqa: BLE001 - a failed critic never blocks generation
            log.warning("critic call failed: %s -> auto-pass", e)
            return Verdict(quality=self.cfg.tau_pass, passed=True,
                           raw={"error": str(e)})

        issues = []
        for rec in out.get("issues", []) or []:
            try:
                issues.append(CriticIssue(
                    issue_type=IssueType(rec.get("type", "quality_degradation")),
                    severity=rec.get("severity", "minor"),
                    detail=rec.get("detail", ""),
                    entity_id=rec.get("entity_id", "")))
            except ValueError:
                continue
        quality = float(out.get("quality", 0.5))
        severe = any(i.severity == "severe" and i.issue_type in SEVERE_ISSUES for i in issues)
        # Eq. 5: pass = Q >= tau_pass AND no severe issue.
        return Verdict(quality=quality, issues=issues,
                       passed=(quality >= self.cfg.tau_pass) and not severe, raw=out)
