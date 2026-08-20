"""Agentic reference selection (Sec. 3.4 "Reference retrieval", Supp. 9).

Per entity: canonical by default; a VLM decides whether auxiliaries better cover the
target shot (agent mode), with the Eq. 7 deterministic scorer as fallback (hybrid).
Final per-entity subsets are packed under the backend's global reference-image limit
(4 for Vidu Q3), prioritizing character identity anchors.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..config import SelectorConfig
from ..llm.client import LLMClient, LLMError
from ..llm import prompts
from ..memory.memory import EntityVisualMemory
from ..schema import Entity, EntityType, Reference, ShotSpec
from ..utils.media import load_image, to_b64_jpeg

log = logging.getLogger("groundshot.select")


@dataclass
class SelectedRefs:
    per_entity: dict[str, list[Reference]] = field(default_factory=dict)

    @property
    def flat(self) -> list[Reference]:
        return [r for pool in self.per_entity.values() for r in pool]

    @property
    def image_paths(self) -> list[str]:
        return [r.image_path for r in self.flat]


class ReferenceSelector:
    def __init__(self, cfg: SelectorConfig, llm: LLMClient):
        self.cfg = cfg
        self.llm = llm

    def select(self, shot: ShotSpec, entities: list[Entity],
               memory: EntityVisualMemory, max_total: int) -> SelectedRefs:
        chosen: dict[str, list[Reference]] = {}
        for ent in entities:
            pool = memory.query(ent.entity_id, self.cfg.kmax)
            if not pool:
                continue
            canonical = pool[0] if pool[0].is_canonical else None
            aux = [r for r in pool if not r.is_canonical]
            if self.cfg.mode == "traditional" or (not aux and canonical):
                chosen[ent.entity_id] = [canonical] if canonical else aux[:1]
                continue
            picks = None
            if self.cfg.mode in ("agent", "hybrid"):
                picks = self._agent_pick(shot, ent, canonical, aux)
            if picks is None:  # agent failed or low-confidence -> traditional
                picks = self._traditional_pick(canonical, aux)
            if picks:
                chosen[ent.entity_id] = picks
        return self._pack(chosen, entities, max_total)

    # -------------------------------------------------------------- strategies
    def _traditional_pick(self, canonical: Reference | None,
                          aux: list[Reference]) -> list[Reference]:
        if canonical:
            return [canonical]
        return sorted(aux, key=lambda r: -r.quality)[:1]

    def _agent_pick(self, shot: ShotSpec, ent: Entity, canonical: Reference | None,
                    aux: list[Reference]) -> list[Reference] | None:
        candidates = ([canonical] if canonical else []) + aux
        if not candidates:
            return None
        try:
            images = [to_b64_jpeg(load_image(r.image_path), max_side=512) for r in candidates]
            aux_note = (f"; images 2-{len(candidates)} are auxiliary references"
                        if len(candidates) > 1 else "") if canonical else \
                       " (no canonical yet; all images are auxiliary candidates)"
            out = self.llm.json_call(
                prompts.SELECTOR_SYSTEM,
                prompts.SELECTOR_USER.format(
                    entity_name=ent.name, entity_type=ent.entity_type.value,
                    entity_desc=ent.description, shot_type=shot.shot_type or "unspecified",
                    shot_text=shot.text, n_images=len(candidates), aux_note=aux_note),
                images_b64=images,
            )
            if float(out.get("confidence", 0)) < self.cfg.min_agent_confidence:
                return None
            picks: list[Reference] = []
            if out.get("use_canonical", True) and canonical:
                picks.append(canonical)
            for idx in out.get("selected_indices", []):
                if isinstance(idx, int) and 0 <= idx < len(aux) and aux[idx] not in picks:
                    picks.append(aux[idx])
            if not picks and ent.entity_type == EntityType.CHARACTER and canonical:
                picks = [canonical]   # characters never drop their identity anchor silently
            return picks or None
        except (LLMError, Exception) as e:  # noqa: BLE001 - graceful degradation (Supp. 9.2)
            log.warning("agent selector failed for %s: %s", ent.entity_id, e)
            return None

    # ------------------------------------------------------------------ packing
    def _pack(self, chosen: dict[str, list[Reference]], entities: list[Entity],
              max_total: int) -> SelectedRefs:
        """Pack under the backend limit: character canonicals > object canonicals >
        location refs > character aux > other aux."""
        etype = {e.entity_id: e.entity_type for e in entities}

        def rank(item: tuple[str, Reference]) -> tuple:
            eid, r = item
            t = etype.get(eid, EntityType.OBJECT)
            type_rank = {EntityType.CHARACTER: 0, EntityType.OBJECT: 1, EntityType.LOCATION: 2}[t]
            return (0 if r.is_canonical else 1, type_rank, -r.quality)

        flat = [(eid, r) for eid, pool in chosen.items() for r in pool]
        flat.sort(key=rank)
        packed: dict[str, list[Reference]] = {}
        covered: set[str] = set()
        # First pass: one reference per entity (coverage before variety).
        for eid, r in flat:
            if sum(len(v) for v in packed.values()) >= max_total:
                break
            if eid in covered:
                continue
            packed.setdefault(eid, []).append(r)
            covered.add(eid)
        # Second pass: fill remaining slots with the best leftovers.
        for eid, r in flat:
            if sum(len(v) for v in packed.values()) >= max_total:
                break
            if r not in packed.get(eid, []):
                packed.setdefault(eid, []).append(r)
        return SelectedRefs(per_entity=packed)
