"""Type-aware candidate quality scores q_cand(c, e) (Sec. 3.3, Supp. 8.1/9.1).

Characters use Eq. 7:  q = 0.4*sharpness + 0.4*face_conf + 0.2*frontality.
Objects/locations replace face terms with recognizability/completeness terms;
the semantic half comes from a VLM check performed by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..schema import EntityType
from ..utils.media import sharpness_score
from .faces import FaceInfo, best_face


@dataclass
class CandidateQuality:
    quality: float
    sharpness: float = 0.0
    face_conf: float = 0.0
    frontality: float = 0.0
    face: FaceInfo | None = None
    notes: dict = field(default_factory=dict)


def score_character_crop(crop: np.ndarray) -> CandidateQuality:
    sharp = sharpness_score(crop)
    face = best_face(crop)
    fconf = face.conf if face else 0.0
    front = face.frontality if face else 0.0
    q = 0.4 * sharp + 0.4 * fconf + 0.2 * front
    return CandidateQuality(quality=float(q), sharpness=sharp, face_conf=fconf,
                            frontality=front, face=face)


def score_object_crop(crop: np.ndarray, det_conf: float,
                      vlm_reliability: float | None = None) -> CandidateQuality:
    """Objects: sharpness + detection confidence + VLM recognizability (when available)."""
    sharp = sharpness_score(crop)
    rec = vlm_reliability if vlm_reliability is not None else det_conf
    q = 0.4 * sharp + 0.2 * det_conf + 0.4 * rec
    return CandidateQuality(quality=float(q), sharpness=sharp,
                            notes={"det_conf": det_conf, "vlm_reliability": rec})


def score_scene_image(img: np.ndarray, bg_frac: float,
                      vlm_quality: float | None = None) -> CandidateQuality:
    """Locations: sharpness + usable-background coverage + VLM reconstruction check."""
    sharp = sharpness_score(img)
    cov = float(np.clip(bg_frac / 0.8, 0.0, 1.0))
    vq = vlm_quality if vlm_quality is not None else 0.5
    q = 0.25 * sharp + 0.25 * cov + 0.5 * vq
    return CandidateQuality(quality=float(q), sharpness=sharp,
                            notes={"bg_frac": bg_frac, "vlm_quality": vq})


def score_crop(entity_type: EntityType, crop: np.ndarray, det_conf: float = 0.5,
               vlm_reliability: float | None = None) -> CandidateQuality:
    if entity_type == EntityType.CHARACTER:
        return score_character_crop(crop)
    return score_object_crop(crop, det_conf, vlm_reliability)
