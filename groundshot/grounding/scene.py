"""Location (scene) reference construction (Sec. 3.3 "Location grounding", Supp. 8.2).

Pipeline: pick the frame with the most usable background -> union-mask all grounded
foreground entities -> dilate -> instruction-following editor removes the masked
foreground and reconstructs the background -> VLM validates the reconstruction.

The editor is the fal `object-removal/mask` endpoint by default (pluggable).
When foreground dominates (bg fraction below scene_min_bg_frac) reconstruction is
skipped entirely: the paper (Sec. 4.6) prefers no scene update over a weak one.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np

from ..config import GroundingConfig
from ..llm.client import LLMClient
from ..llm import prompts
from ..schema import Entity
from ..utils import falapi
from ..utils.media import save_image, load_image, to_b64_jpeg
from .grounder import Detection, Grounder
from .quality import CandidateQuality, score_scene_image

log = logging.getLogger("groundshot.scene")


class SceneReconstructor:
    def __init__(self, cfg: GroundingConfig, llm: LLMClient, offline: bool = False):
        self.cfg = cfg
        self.llm = llm
        self.offline = offline   # mock backend runs: crop-only fallback, no fal call

    def build_scene_reference(self, frames: list[np.ndarray], dets: list[Detection],
                              location: Entity, grounder: Grounder,
                              out_path: Path) -> tuple[str | None, CandidateQuality | None]:
        """Returns (image_path, quality) or (None, None) when skipped/invalid."""
        # Rank frames by usable-background coverage; try the cleanest one first and,
        # after a VLM rejection, retry on the next-cleanest (scene_extra_frames).
        ranked = []
        for fi in range(len(frames)):
            H, W = frames[fi].shape[:2]
            mask = grounder.union_mask((H, W), dets, fi)
            ranked.append((1.0 - mask.mean() / 255.0, fi))
        ranked.sort(reverse=True)
        if not ranked:
            return None, None
        best_bg = ranked[0][0]
        if best_bg < self.cfg.scene_min_bg_frac:
            log.info("scene skip (%s): foreground dominates every frame (bg=%.2f)",
                     location.entity_id, best_bg)
            return None, None

        for bg_frac, fi in ranked[:1 + max(0, self.cfg.scene_extra_frames)]:
            if bg_frac < self.cfg.scene_min_bg_frac:
                break
            frame = frames[fi]
            mask = grounder.union_mask(frame.shape[:2], dets, fi)
            if not mask.any():
                edited = frame                  # nothing to remove
            elif self.offline:
                edited = self._cheap_inpaint(frame, mask)
            else:
                edited = self._fal_remove(frame, mask)
                if edited is None:
                    continue
            vlm_q = self._validate(edited, location)
            if vlm_q is None:
                continue    # rejected; the loop moves on to the next-cleanest frame
            q = score_scene_image(edited, bg_frac, vlm_q)
            path = save_image(edited, out_path)
            return path, q

        # Every reconstruction rejected: keep an almost-people-free raw frame at a
        # quality penalty rather than leaving the location reference-less.
        if self.cfg.scene_raw_fallback and best_bg >= self.cfg.scene_raw_fallback_bg_min:
            frame = frames[ranked[0][1]]
            q = score_scene_image(frame, best_bg, self.cfg.scene_raw_fallback_vlm_q)
            path = save_image(frame, out_path)
            log.info("scene raw-frame fallback (%s): bg=%.2f q=%.2f",
                     location.entity_id, best_bg, q.quality)
            return path, q
        return None, None

    def _validate(self, edited: np.ndarray, location: Entity) -> float | None:
        """VLM check of a reconstruction. Returns its quality, None when rejected.
        Offline (or on VLM failure) falls back to a conservative constant."""
        if self.offline:
            return 0.5
        try:
            out = self.llm.json_call(
                prompts.CROP_CHECK_SYSTEM,
                prompts.SCENE_CHECK_USER.format(
                    entity_name=location.name, entity_desc=location.description),
                images_b64=[to_b64_jpeg(edited)],
            )
            if not out.get("valid", False):
                log.info("scene reconstruction rejected by VLM: %s", out.get("reason"))
                return None
            return float(out.get("quality", 0.5))
        except Exception as e:  # noqa: BLE001
            log.warning("scene VLM check failed (%s); using conservative quality", e)
            return 0.5

    # ------------------------------------------------------------------ editors
    def _fal_remove(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
        try:
            with tempfile.TemporaryDirectory() as td:
                fp = save_image(frame, Path(td) / "frame.png")
                mp = save_image(np.stack([mask] * 3, -1), Path(td) / "mask.png")
                furl, murl = falapi.fal_upload(fp), falapi.fal_upload(mp)
                res = falapi.fal_run(self.cfg.scene_editor_endpoint,
                                     {"image_url": furl, "mask_url": murl,
                                      "mask_expansion": 10})
                images = res.get("images") or []
                if not images:
                    return None
                out = falapi.download(images[0]["url"], Path(td) / "edited.png")
                return load_image(out)
        except Exception as e:  # noqa: BLE001
            log.warning("fal object-removal failed: %s", e)
            return None

    @staticmethod
    def _cheap_inpaint(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        import cv2
        return cv2.inpaint(frame, (mask > 0).astype(np.uint8), 7, cv2.INPAINT_TELEA)
