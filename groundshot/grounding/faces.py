"""Face analysis via insightface (SCRFD detection + ArcFace embedding + pose).

Runs on CPU through onnxruntime, so it works on the Mac and on Linux servers alike.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

log = logging.getLogger("groundshot.faces")


@dataclass
class FaceInfo:
    conf: float
    bbox: tuple[float, float, float, float]
    yaw: float                      # degrees
    embedding: np.ndarray | None    # 512-d ArcFace, L2-normalized
    sex: str | None = None          # "male" / "female" from the genderage head, None if unknown

    @property
    def frontality(self) -> float:
        return max(0.0, 1.0 - abs(self.yaw) / 90.0)


@lru_cache(maxsize=1)
def _app():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    log.info("insightface buffalo_l ready (CPU)")
    return app


def analyze_faces(img: np.ndarray) -> list[FaceInfo]:
    """Detect faces in an RGB image; returns FaceInfo sorted by confidence desc."""
    import cv2
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    faces = _app().get(bgr)
    out = []
    for f in faces:
        emb = getattr(f, "normed_embedding", None)
        yaw = float(f.pose[1]) if getattr(f, "pose", None) is not None else 0.0
        sex = getattr(f, "sex", None)           # 'M' / 'F' in insightface releases
        if sex is None and getattr(f, "gender", None) is not None:
            sex = "M" if int(f.gender) == 1 else "F"
        sex = {"M": "male", "F": "female"}.get(sex)
        out.append(FaceInfo(conf=float(f.det_score), bbox=tuple(f.bbox), yaw=yaw,
                            embedding=np.asarray(emb) if emb is not None else None,
                            sex=sex))
    return sorted(out, key=lambda x: -x.conf)


def best_face(img: np.ndarray) -> FaceInfo | None:
    faces = analyze_faces(img)
    return faces[0] if faces else None


def face_sim(a: FaceInfo, b: FaceInfo) -> float:
    if a.embedding is None or b.embedding is None:
        return 0.0
    return float(np.dot(a.embedding, b.embedding))
