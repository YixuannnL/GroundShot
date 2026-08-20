"""Entity-level visual memory (Sec. 3.3, Supp. 8).

Gates on registration (in order):
  1. type-specific validation: q(c) >= tau_min (0.4); characters need face conf >= 0.3
  2. canonical init (Supp. 8.3): canonical slot filled only by q >= 0.85 +
     canonical-ready visibility; first such candidate wins, never overwritten
  3. identity gate: sim(c, r*) >= theta_id (ArcFace for two faced character crops,
     DINOv2 otherwise) — value UNSPECIFIED-IN-PAPER, configurable
  4. diversity gate: max CLIP cos to existing refs < 0.92
  5. quotas: <=2 refs per (shot, entity), <=6 active refs (1 canonical + 5 aux)

DEVIATION (documented, on by default, disable with strict_paper_mode):
  Supp. 8.3 rejects all sub-canonical candidates while the canonical slot is empty,
  which deadlocks entities that never yield a q>=0.85 crop into pure T2V. We keep a
  best-so-far *provisional* canonical (q >= provisional_quality_min) once the entity's
  designated source shot has been generated; a later true-canonical candidate that is
  identity-consistent with it upgrades/replaces the provisional anchor.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from ..config import MemoryConfig
from ..embeddings import clip_image_embed, dino_embed, cos
from ..grounding.faces import best_face, face_sim
from ..grounding.quality import CandidateQuality
from ..schema import Entity, EntityType, Reference
from ..utils.media import load_image

log = logging.getLogger("groundshot.memory")


class EntityVisualMemory:
    def __init__(self, cfg: MemoryConfig, store_dir: Path):
        self.cfg = cfg
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.refs: dict[str, list[Reference]] = {}      # entity_id -> [canonical?, aux...]
        self._clip_cache: dict[str, np.ndarray] = {}
        self._dino_cache: dict[str, np.ndarray] = {}
        self._shots_seen = 0

    # -------------------------------------------------------------- accessors
    def canonical(self, entity_id: str) -> Reference | None:
        for r in self.refs.get(entity_id, []):
            if r.is_canonical:
                return r
        return None

    def auxiliaries(self, entity_id: str) -> list[Reference]:
        return [r for r in self.refs.get(entity_id, []) if not r.is_canonical]

    def has_usable(self, entity_id: str) -> bool:
        return bool(self.refs.get(entity_id))

    # ------------------------------------------------------------ registration
    def register(self, entity: Entity, image_path: str, quality: CandidateQuality,
                 source_shot: int, contributed_this_shot: int, meta: dict | None = None) -> str:
        """Try to admit a candidate. Returns a status string for logging/experience."""
        eid = entity.entity_id
        img = load_image(image_path)

        # Gate 1: type-specific validation (Supp. 8.1).
        if quality.quality < self.cfg.tau_min:
            return f"reject:low_quality({quality.quality:.2f})"
        if entity.entity_type == EntityType.CHARACTER and quality.face_conf < self.cfg.char_face_conf_min:
            return f"reject:face_conf({quality.face_conf:.2f})"
        if (entity.entity_type == EntityType.CHARACTER and self.cfg.gender_gate
                and entity.gender in ("male", "female")
                and quality.face is not None and quality.face.sex is not None
                and quality.face.conf >= self.cfg.gender_gate_face_conf_min
                and quality.face.sex != entity.gender):
            return f"reject:gender_mismatch({quality.face.sex})"
        if contributed_this_shot >= self.cfg.max_refs_per_shot_per_entity:
            return "reject:per_shot_quota"

        ref = Reference(
            entity_id=eid, entity_type=entity.entity_type, image_path=image_path,
            quality=quality.quality, source_shot=source_shot,
            face_conf=quality.face_conf, frontality=quality.frontality,
            sharpness=quality.sharpness, meta=meta or {},
        )
        canonical = self.canonical(eid)

        if canonical is None:
            if self._canonical_ready(entity, quality):
                ref.is_canonical = True
                self.refs.setdefault(eid, []).insert(0, ref)
                return "admit:canonical"
            if self.cfg.provisional_canonical and quality.quality >= self.cfg.provisional_quality_min:
                # Keep only the single best provisional candidate.
                pool = self.refs.setdefault(eid, [])
                if pool and pool[0].is_provisional:
                    if quality.quality <= pool[0].quality:
                        return "reject:worse_than_provisional"
                    pool.pop(0)
                ref.is_canonical = True
                ref.is_provisional = True
                pool.insert(0, ref)
                return "admit:provisional_canonical"
            return "reject:pre_canonical"

        # Canonical exists. A true-canonical candidate may upgrade a provisional anchor.
        if canonical.is_provisional and self._canonical_ready(entity, quality):
            if self._identity_ok(entity, img, canonical):
                ref.is_canonical = True
                self.refs[eid][0] = ref
                self._revalidate_aux(entity)
                return "admit:canonical_upgrade"
            # Not consistent with provisional: trust the higher-quality evidence.
            ref.is_canonical = True
            self.refs[eid] = [ref]
            return "admit:canonical_replace_inconsistent_provisional"

        # Gate 3: identity gate against the canonical (Eq. 6).
        if not self._identity_ok(entity, img, canonical):
            return "reject:identity_gate"

        # Gate 4: diversity gate (Supp. 8.1, CLIP ViT-B/32).
        c_emb = self._clip_of(image_path, img)
        for r in self.refs[eid]:
            if float(np.dot(c_emb, self._clip_of(r.image_path))) >= self.cfg.clip_redundancy_max:
                return "reject:redundant"

        # Gate 5: pool size / eviction (Supp. 8.4).
        aux = self.auxiliaries(eid)
        if len(aux) >= self.cfg.max_active_refs - 1:
            worst = min(aux, key=lambda r: r.quality)
            if ref.quality <= worst.quality:
                return "reject:pool_full"
            self.refs[eid].remove(worst)
        self.refs[eid].append(ref)
        return "admit:auxiliary"

    def _canonical_ready(self, entity: Entity, q: CandidateQuality) -> bool:
        if q.quality < self.cfg.canonical_quality_min:
            return False
        if entity.entity_type == EntityType.CHARACTER:
            return (q.face_conf >= self.cfg.canonical_face_conf_min
                    and q.frontality >= self.cfg.canonical_frontality_min)
        return True  # objects/locations: quality gate + upstream VLM validation

    def _identity_ok(self, entity: Entity, img: np.ndarray, canonical: Reference) -> bool:
        can_img = load_image(canonical.image_path)
        if entity.entity_type == EntityType.CHARACTER:
            fa, fb = best_face(img), best_face(can_img)
            if fa and fb and fa.embedding is not None and fb.embedding is not None:
                return face_sim(fa, fb) >= self.cfg.theta_id_face
            thr = self.cfg.theta_id_dino_char
        elif entity.entity_type == EntityType.OBJECT:
            thr = self.cfg.theta_id_dino_obj
        else:
            thr = self.cfg.theta_id_dino_loc
        a = self._dino_of(canonical.image_path, can_img)
        b = dino_embed([img])[0]
        return cos(a, b) >= thr

    def _revalidate_aux(self, entity: Entity) -> None:
        """After a canonical upgrade, drop auxiliaries inconsistent with the new anchor."""
        eid = entity.entity_id
        canonical = self.canonical(eid)
        keep = [canonical]
        for r in self.auxiliaries(eid):
            if self._identity_ok(entity, load_image(r.image_path), canonical):
                keep.append(r)
        self.refs[eid] = keep

    # ---------------------------------------------------------------- eviction
    def on_shot_done(self) -> None:
        self._shots_seen += 1
        if self._shots_seen % self.cfg.cleanup_every_n_shots == 0:
            self.cleanup()

    def cleanup(self) -> None:
        for eid, pool in self.refs.items():
            kept = [r for r in pool if r.is_canonical or r.quality >= self.cfg.evict_low_quality]
            # redundancy sweep among auxiliaries
            final: list[Reference] = [r for r in kept if r.is_canonical]
            for r in sorted([r for r in kept if not r.is_canonical], key=lambda r: -r.quality):
                emb = self._clip_of(r.image_path)
                if all(float(np.dot(emb, self._clip_of(o.image_path))) < self.cfg.clip_redundancy_max
                       for o in final):
                    final.append(r)
            self.refs[eid] = final

    # ------------------------------------------------------------------ query
    def query(self, entity_id: str, k: int) -> list[Reference]:
        """Supp. 8.5: canonical first, then quality, shot id as final tie-breaker."""
        pool = sorted(self.refs.get(entity_id, []),
                      key=lambda r: (not r.is_canonical, -r.quality, r.source_shot))
        return pool[:k]

    # ------------------------------------------------------------- persistence
    def save(self) -> None:
        data = {eid: [r.to_dict() for r in pool] for eid, pool in self.refs.items()}
        (self.store_dir / "memory.json").write_text(json.dumps(data, indent=1))

    def load(self) -> None:
        p = self.store_dir / "memory.json"
        if p.exists():
            data = json.loads(p.read_text())
            self.refs = {eid: [Reference.from_dict(r) for r in pool]
                         for eid, pool in data.items()}

    # -------------------------------------------------------------- emb caches
    def _clip_of(self, path: str, img: np.ndarray | None = None) -> np.ndarray:
        if path not in self._clip_cache:
            self._clip_cache[path] = clip_image_embed([img if img is not None else load_image(path)])[0]
        return self._clip_cache[path]

    def _dino_of(self, path: str, img: np.ndarray | None = None) -> np.ndarray:
        if path not in self._dino_cache:
            self._dino_cache[path] = dino_embed([img if img is not None else load_image(path)])[0]
        return self._dino_cache[path]
