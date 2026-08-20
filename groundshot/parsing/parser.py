"""Script parsing (Sec. 3.2 "Script parsing" + Supp. 6.1).

One batched LLM call per script performs: entity extraction + cross-shot coreference,
per-shot entity sets, expected counts, layered prompt construction, and the
reference-quality estimation Qref for every (shot, entity) pair (Eq. 1).

Two input forms are supported:
  - GroundBench YAML (characters/settings/recurring_objects partially given: those
    definitions are passed to the LLM as authoritative so IDs stay stable, which is
    also what UED evaluation expects);
  - free text: a list of shot strings + optional global caption.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

from ..config import GroundShotConfig
from ..llm.client import LLMClient
from ..llm import prompts
from ..schema import Entity, EntityType, ENTITY_PRIORITY, LayeredPrompt, ParsedScript, ShotSpec

log = logging.getLogger("groundshot.parse")


def load_bench_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def bench_known_entities(doc: dict) -> dict[str, dict]:
    """Convert GroundBench YAML metadata into authoritative entity definitions."""
    known: dict[str, dict] = {}
    meta = doc.get("metadata", {})
    for ch in meta.get("characters", []) or []:
        known[ch["id"]] = {
            "type": "character",
            "name": _short_name(ch.get("description", ch["id"])),
            "description": ch.get("description", ""),
            "aliases": [],
            "gender": str(ch["gender"]).strip().lower() if ch.get("gender") else None,
        }
    for i, obj in enumerate(meta.get("recurring_objects", []) or []):
        oid = f"obj_{obj.get('name', f'{i:02d}').replace(' ', '_')}"
        known[oid] = {
            "type": "object",
            "name": obj.get("name", oid),
            "description": obj.get("description", obj.get("name", "")),
            "aliases": [],
        }
    for st in meta.get("settings", []) or []:
        lid = f"loc_{st['id']}" if not str(st["id"]).startswith("loc") else str(st["id"])
        known[lid] = {
            "type": "location",
            "name": st.get("name", lid),
            "description": st.get("description", ""),
            "aliases": [],
        }
    return known


def _short_name(desc: str, max_words: int = 8) -> str:
    return " ".join(desc.replace(",", " ").split()[:max_words])


class ScriptParser:
    def __init__(self, cfg: GroundShotConfig, llm: LLMClient):
        self.cfg = cfg
        self.llm = llm

    # ------------------------------------------------------------- entrypoints
    def parse_bench(self, yaml_path: str | Path) -> ParsedScript:
        doc = load_bench_yaml(yaml_path)
        meta = doc.get("metadata", {})
        shots_text = [s["text"] for s in doc["shots"]]
        known = bench_known_entities(doc)
        if self.llm.provider == "none":
            return self._parse_bench_offline(doc, known)
        parsed = self._parse(
            script_id=meta.get("script_id", Path(yaml_path).stem),
            global_caption=doc.get("global_caption", ""),
            shots_text=shots_text,
            known_entities=known,
        )
        parsed.seed = meta.get("generation_seed")
        # Merge authoritative per-shot character presence from the YAML.
        for spec, shot_doc in zip(parsed.shots, doc["shots"]):
            listed = shot_doc.get("characters_present") or []
            for cid in listed:
                if cid in parsed.entities and cid not in spec.entity_ids:
                    spec.entity_ids.append(cid)
            # The YAML's characters_present is authoritative: drop characters the
            # LLM placed in this shot that the script does not list.
            spec.entity_ids = [
                eid for eid in spec.entity_ids
                if not (eid in known and known[eid]["type"] == "character" and eid not in listed)
            ]
            spec.duration = float(shot_doc.get("duration_seconds", spec.duration))
            spec.shot_type = shot_doc.get("shot_type", spec.shot_type)
        return parsed

    def _parse_bench_offline(self, doc: dict, known: dict[str, dict]) -> ParsedScript:
        """Heuristic no-LLM parse of a GroundBench YAML (offline smoke tests and
        LLM-outage fallback): entities from metadata, per-shot presence from
        characters_present + keyword matches, ref_quality from shot_type."""
        meta = doc.get("metadata", {})
        entities: dict[str, Entity] = {}
        for eid, rec in known.items():
            etype = EntityType(rec["type"])
            entities[eid] = Entity(eid, etype, rec["name"], rec["description"],
                                   rec.get("aliases", []), ENTITY_PRIORITY[etype],
                                   gender=rec.get("gender"))
        settings = meta.get("settings", []) or []
        default_loc = None
        if len(settings) == 1:
            sid = settings[0]["id"]
            default_loc = sid if sid.startswith("loc") else f"loc_{sid}"

        char_q = {"close-up": 0.9, "medium": 0.7, "extreme-close-up": 0.45,
                  "wide": 0.35, "extreme-wide": 0.2}
        loc_q = {"wide": 0.85, "extreme-wide": 0.9, "establishing": 0.9,
                 "medium": 0.5, "close-up": 0.2, "extreme-close-up": 0.1}

        global_caption = doc.get("global_caption", "")
        style = global_caption.split(".")[0].strip() + "." if global_caption else ""
        specs: list[ShotSpec] = []
        for shot_doc in doc["shots"]:
            text_l = shot_doc["text"].lower()
            eids = [c for c in (shot_doc.get("characters_present") or []) if c in entities]
            for eid, ent in entities.items():
                if ent.entity_type == EntityType.OBJECT:
                    words = [w for w in ent.name.lower().split() if len(w) > 3]
                    if words and any(w in text_l for w in words):
                        eids.append(eid)
            if default_loc and default_loc in entities:
                eids.append(default_loc)
            elif settings:
                for st in settings:
                    lid = st["id"] if str(st["id"]).startswith("loc") else f"loc_{st['id']}"
                    words = [w for w in str(st.get("name", "")).lower().split() if len(w) > 3]
                    if lid in entities and words and any(w in text_l for w in words):
                        eids.append(lid)
                        break
            stype = shot_doc.get("shot_type", "medium")
            rq = {}
            for eid in eids:
                et = entities[eid].entity_type
                if et == EntityType.CHARACTER:
                    rq[eid] = char_q.get(stype, 0.5)
                elif et == EntityType.LOCATION:
                    rq[eid] = loc_q.get(stype, 0.5)
                else:
                    rq[eid] = 0.7 if stype in ("close-up", "medium") else 0.4
            n_chars = len([e for e in eids if entities[e].entity_type == EntityType.CHARACTER])
            specs.append(ShotSpec(
                shot_id=int(shot_doc["shot_id"]), text=shot_doc["text"],
                entity_ids=list(dict.fromkeys(eids)),
                duration=float(shot_doc.get("duration_seconds", 4.0)),
                shot_type=stype, expected_char_count=n_chars or None,
                prompt=LayeredPrompt(
                    style=style,
                    entities="; ".join(entities[e].description for e in eids
                                       if entities[e].entity_type != EntityType.LOCATION),
                    action=shot_doc["text"]),
                ref_quality=rq))
        parsed = ParsedScript(
            script_id=meta.get("script_id", "offline"), entities=entities,
            shots=specs, global_caption=global_caption, style_layer=style,
            seed=meta.get("generation_seed"))
        return parsed

    def parse_text(self, script_id: str, shots_text: list[str],
                   global_caption: str = "") -> ParsedScript:
        return self._parse(script_id, global_caption, shots_text, known_entities={})

    # ------------------------------------------------------------------ core
    def _parse(self, script_id: str, global_caption: str, shots_text: list[str],
               known_entities: dict[str, dict]) -> ParsedScript:
        shots_block = "\n".join(f"Shot {i + 1}: {t}" for i, t in enumerate(shots_text))
        keb = ""
        if known_entities:
            keb = prompts.KNOWN_ENTITIES_BLOCK.format(
                entities_json=json.dumps(known_entities, indent=1, ensure_ascii=False))
        out = self.llm.json_call(
            prompts.PARSE_SYSTEM,
            prompts.PARSE_USER.format(global_caption=global_caption or "(none)",
                                      shots_block=shots_block,
                                      known_entities_block=keb),
            role="parse", max_tokens=8192,
        )
        entities: dict[str, Entity] = {}
        for eid, rec in out.get("entities", {}).items():
            etype = EntityType(rec["type"])
            gender = str(rec.get("gender") or "").strip().lower() or None
            entities[eid] = Entity(
                entity_id=eid, entity_type=etype,
                name=rec.get("name", eid),
                description=rec.get("description", ""),
                aliases=rec.get("aliases", []),
                priority=ENTITY_PRIORITY[etype],
                gender=gender if gender in ("male", "female") else None,
            )
        # Authoritative definitions win over LLM rewrites for type; keep LLM name/desc
        # only when the known description is empty.
        for eid, rec in known_entities.items():
            etype = EntityType(rec["type"])
            if eid not in entities:
                entities[eid] = Entity(eid, etype, rec["name"], rec["description"],
                                       rec.get("aliases", []), ENTITY_PRIORITY[etype],
                                       gender=rec.get("gender"))
            else:
                entities[eid].entity_type = etype
                entities[eid].priority = ENTITY_PRIORITY[etype]
                entities[eid].gender = rec.get("gender") or entities[eid].gender
                if rec["description"]:
                    entities[eid].description = rec["description"]

        specs: list[ShotSpec] = []
        recs = {int(r["shot_id"]): r for r in out.get("shots", [])}
        for i, text in enumerate(shots_text, start=1):
            r = recs.get(i, {})
            eids = [e for e in r.get("entity_ids", []) if e in entities]
            spec = ShotSpec(
                shot_id=i, text=text, entity_ids=eids,
                expected_char_count=r.get("expected_char_count"),
                prompt=LayeredPrompt(
                    style=out.get("style_layer", ""),
                    entities=r.get("entity_layer", ""),
                    action=r.get("action_layer", text),
                ),
                ref_quality={k: float(v) for k, v in (r.get("ref_quality") or {}).items()
                             if k in entities},
            )
            specs.append(spec)
        parsed = ParsedScript(script_id=script_id, entities=entities, shots=specs,
                              global_caption=global_caption,
                              style_layer=out.get("style_layer", ""))
        _sanity_fill(parsed)
        _ensure_appearance_in_entity_layer(parsed)
        return parsed


def _ensure_appearance_in_entity_layer(parsed: ParsedScript) -> None:
    """The generator sees only the composed prompt, so a shot's entity layer must
    carry each present entity's appearance. LLM parses sometimes compress the layer
    into a scene sentence and drop wardrobe/appearance entirely (observed with
    GPT-4o), which the critic then correctly fails every attempt. When less than
    half of an entity's significant description words appear in the layer, append
    the authoritative description."""
    for spec in parsed.shots:
        if spec.prompt is None:
            continue
        layer_l = spec.prompt.entities.lower()
        missing = []
        for eid in spec.entity_ids:
            ent = parsed.entities[eid]
            if ent.entity_type == EntityType.LOCATION or not ent.description:
                continue
            words = [w.strip(".,;:()") for w in ent.description.lower().split()]
            words = [w for w in words if len(w) > 3]
            if words and sum(w in layer_l for w in words) / len(words) < 0.5:
                missing.append(ent.description)
        if missing:
            spec.prompt.entities = "; ".join(
                filter(None, [spec.prompt.entities, *missing]))


def _sanity_fill(parsed: ParsedScript) -> None:
    """Guarantee every (shot, entity) pair has a ref_quality value."""
    for spec in parsed.shots:
        for eid in spec.entity_ids:
            if eid not in spec.ref_quality:
                # Neutral default; wide shots get a small penalty for foreground entities.
                ent = parsed.entities[eid]
                base = 0.5
                if ent.entity_type != EntityType.LOCATION and spec.shot_type in ("wide", "extreme-wide"):
                    base = 0.35
                if ent.entity_type == EntityType.LOCATION and spec.shot_type in ("wide", "establishing"):
                    base = 0.7
                spec.ref_quality[eid] = base
