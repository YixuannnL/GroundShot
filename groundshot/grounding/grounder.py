"""Entity-level grounding (Sec. 3.3): localize each entity in sampled frames of an
accepted shot and emit candidate crops.

The paper uses ReferDINO (referring video segmentation). We only need boxes/crops,
so the default implementation is open-vocabulary GroundingDINO (HF transformers),
which runs on CUDA / MPS / CPU. The interface is pluggable: point
grounding.model_id at any zero-shot-object-detection checkpoint, or subclass
`Grounder` with a ReferDINO server call for exact paper parity.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from PIL import Image

from ..config import GroundingConfig
from ..embeddings import pick_device
from ..schema import Entity, EntityType

log = logging.getLogger("groundshot.ground")


def _iom(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection over the smaller box's area (robust to nested boxes, unlike IoU)."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    min_area = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return (ix * iy) / (min_area + 1e-8)


@dataclass
class Detection:
    entity_id: str
    frame_idx: int
    box: tuple[int, int, int, int]      # x1,y1,x2,y2 in pixels
    score: float
    crop: np.ndarray


@lru_cache(maxsize=1)
def _gdino(model_id: str, device_pref: str):
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    device = pick_device(device_pref)
    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
    log.info("GroundingDINO %s loaded on %s", model_id, device)
    return proc, model, device


class Grounder:
    def __init__(self, cfg: GroundingConfig):
        self.cfg = cfg

    def query_for(self, entity: Entity) -> str:
        # GroundingDINO expects lowercase phrases terminated by periods.
        # Characters use a category noun, NOT the attribute phrase: GroundingDINO
        # binds phrases like "early 40s male wavy brown hair" to sub-parts (a box
        # around the hair) and to the wrong person; whole-person boxes come from
        # plain category queries, and identity assignment is handled downstream by
        # the gender/identity gates. Set grounding.char_query="name" to revert.
        if (entity.entity_type == EntityType.CHARACTER
                and self.cfg.char_query == "category"):
            noun = {"male": "man", "female": "woman"}.get(entity.gender or "", "person")
            return noun + "."
        return entity.name.lower().rstrip(".") + "."

    @torch.no_grad()
    def detect(self, frames: list[np.ndarray], entities: list[Entity]) -> list[Detection]:
        """Ground every foreground entity in every frame; returns all accepted boxes."""
        fg = [e for e in entities if e.entity_type != EntityType.LOCATION]
        if not fg:
            return []
        proc, model, device = _gdino(self.cfg.model_id, self.cfg.device)
        dets: list[Detection] = []
        for fi, frame in enumerate(frames):
            pil = Image.fromarray(frame)
            H, W = frame.shape[:2]
            for ent in fg:
                inputs = proc(images=pil, text=self.query_for(ent),
                              return_tensors="pt").to(device)
                out = model(**inputs)
                try:   # transformers >= 4.51 renamed box_threshold -> threshold
                    res = proc.post_process_grounded_object_detection(
                        out, inputs.input_ids,
                        threshold=self.cfg.box_threshold,
                        text_threshold=self.cfg.text_threshold,
                        target_sizes=[(H, W)])[0]
                except TypeError:
                    res = proc.post_process_grounded_object_detection(
                        out, inputs.input_ids,
                        box_threshold=self.cfg.box_threshold,
                        text_threshold=self.cfg.text_threshold,
                        target_sizes=[(H, W)])[0]
                for box, score in zip(res["boxes"], res["scores"]):
                    x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(W, x2), min(H, y2)
                    area_frac = ((x2 - x1) * (y2 - y1)) / (W * H + 1e-8)
                    if not (self.cfg.min_crop_area_frac <= area_frac <= self.cfg.max_crop_area_frac):
                        continue
                    dets.append(Detection(
                        entity_id=ent.entity_id, frame_idx=fi,
                        box=(x1, y1, x2, y2), score=float(score),
                        crop=frame[y1:y2, x1:x2].copy(),
                    ))
        if self.cfg.cross_entity_suppression:
            dets = self._suppress_cross_entity(dets)
        return dets

    def _suppress_cross_entity(self, dets: list[Detection]) -> list[Detection]:
        """Per-entity queries run independently, so two entities can claim the same
        region of a frame (e.g. both character phrases boxing the same person).
        Greedily keep detections by score; drop one whose box overlaps a kept box
        of a DIFFERENT entity with intersection-over-min-area > cross_entity_iom."""
        kept: list[Detection] = []
        dropped = 0
        for d in sorted(dets, key=lambda d: -d.score):
            clash = any(
                k.frame_idx == d.frame_idx and k.entity_id != d.entity_id
                and _iom(k.box, d.box) > self.cfg.cross_entity_iom
                for k in kept)
            if clash:
                dropped += 1
            else:
                kept.append(d)
        if dropped:
            log.info("cross-entity suppression dropped %d/%d boxes", dropped, len(dets))
        return kept

    def best_per_entity(self, dets: list[Detection], top_k: int = 3) -> dict[str, list[Detection]]:
        """Keep the top-k highest-score detections per entity across frames,
        at most one per frame."""
        by_ent: dict[str, list[Detection]] = {}
        for d in sorted(dets, key=lambda d: -d.score):
            lst = by_ent.setdefault(d.entity_id, [])
            if len(lst) < top_k and all(x.frame_idx != d.frame_idx for x in lst):
                lst.append(d)
        return by_ent

    def union_mask(self, frame_shape: tuple[int, int], dets: list[Detection],
                   frame_idx: int) -> np.ndarray:
        """Binary mask (255 = foreground) of all detections in one frame, dilated."""
        import cv2
        H, W = frame_shape
        mask = np.zeros((H, W), np.uint8)
        for d in dets:
            if d.frame_idx != frame_idx:
                continue
            x1, y1, x2, y2 = d.box
            mask[y1:y2, x1:x2] = 255
        if self.cfg.mask_dilate_px > 0 and mask.any():
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.cfg.mask_dilate_px * 2 + 1,) * 2)
            mask = cv2.dilate(mask, k)
        return mask
