"""GroundShot pipeline — the full Algorithm 1 loop.

For each scheduled shot: retrieve experience advice -> retrieve references ->
select generator (T2V/Ref2V) -> generate -> verify & repair -> ground entities ->
update visual memory -> record experience. Finally shots are returned in
narrative order.

Run directory layout:
  runs/<run>/<script_id>/parsed.json          parsed script
  runs/<run>/<script_id>/schedule.json        generation order + source shots
  runs/<run>/<script_id>/shots/shot_<id>.mp4  accepted videos (narrative ids)
  runs/<run>/<script_id>/crops/               grounded candidate crops
  runs/<run>/<script_id>/memory/              admitted references + memory.json
  runs/<run>/<script_id>/results.json         per-shot metadata
  runs/<run>/experience.jsonl                 generation experience X (per run)
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from .config import GroundShotConfig
from .generation.base import VideoBackend, make_backend
from .grounding.grounder import Grounder
from .grounding.quality import score_crop
from .grounding.scene import SceneReconstructor
from .llm.client import LLMClient
from .llm import prompts
from .memory.memory import EntityVisualMemory
from .parsing.parser import ScriptParser
from .schema import Entity, EntityType, ParsedScript, ShotResult, ShotSpec, Verdict
from .scheduling.scheduler import schedule, source_shots
from .selection.selector import ReferenceSelector, SelectedRefs
from .utils.media import sample_frames, save_image, to_b64_jpeg
from .verification.critic import ShotCritic
from .verification.experience import ExperienceStore
from .verification.repair import plan_repair

log = logging.getLogger("groundshot.pipeline")


class GroundShotPipeline:
    def __init__(self, cfg: GroundShotConfig, run_name: str = "default"):
        self.cfg = cfg
        self.run_dir = Path(cfg.runs_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.llm = LLMClient(cfg.llm)
        self.backend: VideoBackend = make_backend(cfg.backend)
        self.parser = ScriptParser(cfg, self.llm)
        self.grounder = Grounder(cfg.grounding)
        self.selector = ReferenceSelector(cfg.selector, self.llm)
        self.critic = ShotCritic(cfg.verify, self.llm)
        self.scene = SceneReconstructor(cfg.grounding, self.llm,
                                        offline=(cfg.backend.name == "mock"))
        self.experience = ExperienceStore(self.run_dir / "experience.jsonl")

    # ------------------------------------------------------------------ public
    def run_bench_script(self, yaml_path: str | Path) -> dict:
        script_dir = self.run_dir / Path(yaml_path).stem
        parsed_path = script_dir / "parsed.json"
        if parsed_path.exists():
            parsed = ParsedScript.load(parsed_path)
            log.info("loaded cached parse for %s", parsed.script_id)
        else:
            parsed = self.parser.parse_bench(yaml_path)
            script_dir.mkdir(parents=True, exist_ok=True)
            parsed.save(parsed_path)
        return self.run_parsed(parsed, script_dir)

    def run_text_script(self, script_id: str, shots_text: list[str],
                        global_caption: str = "") -> dict:
        script_dir = self.run_dir / script_id
        parsed = self.parser.parse_text(script_id, shots_text, global_caption)
        script_dir.mkdir(parents=True, exist_ok=True)
        parsed.save(script_dir / "parsed.json")
        return self.run_parsed(parsed, script_dir)

    # -------------------------------------------------------------------- core
    def run_parsed(self, script: ParsedScript, script_dir: Path) -> dict:
        script_dir.mkdir(parents=True, exist_ok=True)
        order = schedule(script, self.cfg.scheduler)
        src_shots = source_shots(script, self.cfg.scheduler)
        (script_dir / "schedule.json").write_text(json.dumps(
            {"order": order, "source_shots": src_shots}, indent=1))

        memory = EntityVisualMemory(self.cfg.memory, script_dir / "memory")
        memory.load()   # resume support
        if self.cfg.t2i_bootstrap:
            self._t2i_bootstrap(script, memory)
        results: dict[int, ShotResult] = {}
        results_path = script_dir / "results.json"
        if results_path.exists():   # resume: skip completed shots
            for rec in json.loads(results_path.read_text()).get("shots", []):
                if Path(rec["video_path"]).exists():
                    results[rec["shot_id"]] = ShotResult(**{
                        k: v for k, v in rec.items()
                        if k in ShotResult.__dataclass_fields__ and k != "verdict"})

        conditioned_entities: set[str] = set()
        shots_by_id = {s.shot_id: s for s in script.shots}

        for t, sid in enumerate(order, 1):
            if sid in results:
                log.info("[%d/%d] shot %d already done, skipping", t, len(order), sid)
                continue
            shot = shots_by_id[sid]
            log.info("[%d/%d] generating narrative shot %d: %s",
                     t, len(order), sid, shot.text[:80])
            result = self._generate_shot(script, shot, memory, src_shots,
                                         conditioned_entities, script_dir)
            results[sid] = result
            if result.mode == "ref2v":
                conditioned_entities.update(shot.entity_ids)
            # Ground + update memory only from accepted (kept) videos.
            self._ground_and_update(script, shot, result, memory, script_dir)
            memory.on_shot_done()
            memory.save()
            self._save_results(script, results, results_path)

        self._save_results(script, results, results_path)
        log.info("script %s complete: %d shots", script.script_id, len(results))
        return json.loads(results_path.read_text())

    # ------------------------------------------------------------ shot generation
    def _generate_shot(self, script: ParsedScript, shot: ShotSpec,
                       memory: EntityVisualMemory, src_shots: dict[str, int],
                       conditioned: set[str], script_dir: Path) -> ShotResult:
        entities = script.entities_of(shot)
        fg = [e for e in entities if e.entity_type != EntityType.LOCATION]

        advice, high_risk = self.experience.advice(
            shot.shot_type, [e.entity_type.value for e in entities])

        refs = self.selector.select(shot, entities, memory,
                                    self.backend.max_reference_images)
        # Generation mode (Sec. 3.4): Ref2V when a usable foreground reference exists.
        # Optimization (non-strict mode): also use Ref2V when only a scene reference
        # exists, so location consistency benefits from visual conditioning.
        has_fg_ref = any(eid in refs.per_entity for eid in (e.entity_id for e in fg))
        use_ref2v = has_fg_ref or (bool(refs.flat) and not self.cfg.strict_paper_mode)
        if not use_ref2v:
            refs = SelectedRefs()

        base_prompt = shot.prompt.compose(extra=advice) if shot.prompt else shot.text
        shots_dir = script_dir / "shots"
        shots_dir.mkdir(parents=True, exist_ok=True)

        is_source = any(src == shot.shot_id for src in src_shots.values())
        first_ref2v = use_ref2v and any(
            eid not in conditioned for eid in refs.per_entity)
        want_verify = self.critic.should_verify(is_source, first_ref2v, high_risk)

        best: tuple[float, str, Verdict | None, int, list[str]] | None = None
        extra_prompt, cur_refs, seed_bump = "", refs, 0
        attempts = 0
        issues_seen: list[str] = []
        repairs_done: list[str] = []

        for attempt in range(1 + self.cfg.verify.retry_budget):
            attempts += 1
            seed = ((script.seed or 0) + shot.shot_id * 97 + seed_bump) % (2**31)
            out_path = shots_dir / f"shot_{shot.shot_id:03d}_attempt{attempt + 1}.mp4"
            prompt = (base_prompt + " " + extra_prompt).strip()[:1990]
            mode = "ref2v" if (use_ref2v and cur_refs.flat) else "t2v"
            try:
                if mode == "ref2v":
                    self.backend.ref2v(prompt, cur_refs.image_paths, out_path,
                                       int(shot.duration), seed)
                else:
                    self.backend.t2v(prompt, out_path, int(shot.duration), seed)
            except Exception as e:  # noqa: BLE001
                log.error("generation attempt %d failed: %s", attempt + 1, e)
                seed_bump += 1
                continue

            if not want_verify:
                verdict = None
                best = (1.0, str(out_path), None, attempt, cur_refs.image_paths)
                break
            verdict = self.critic.verify(str(out_path), shot, entities,
                                         cur_refs.image_paths)
            issues_seen += [i.issue_type.value for i in verdict.issues]
            score = verdict.quality - (0.5 if verdict.severe_issues else 0.0)
            if best is None or score > best[0]:
                best = (score, str(out_path), verdict, attempt, cur_refs.image_paths)
            if verdict.passed:
                break
            if attempt >= self.cfg.verify.retry_budget:
                break
            # Targeted repair for the next attempt (Sec. 3.4).
            plan = plan_repair(verdict, shot, entities, script.style_layer)
            repairs_done += plan.actions
            extra_prompt = plan.extra_prompt
            if plan.reseed:
                seed_bump += 1
            if plan.strengthen_references and use_ref2v:
                cur_refs = SelectedRefs(per_entity={
                    eid: [r for r in pool if r.is_canonical] or pool[:1]
                    for eid, pool in cur_refs.per_entity.items()})
            log.info("repair attempt %d: %s", attempt + 2, plan.actions)

        if best is None:
            raise RuntimeError(f"all generation attempts failed for shot {shot.shot_id}")

        _, video_path, verdict, best_attempt, used_refs = best
        final_path = shots_dir / f"shot_{shot.shot_id:03d}.mp4"
        Path(video_path).replace(final_path)

        self.experience.record(
            script_id=script.script_id, shot_id=shot.shot_id, shot_type=shot.shot_type,
            entity_types=[e.entity_type.value for e in entities],
            mode="ref2v" if used_refs else "t2v", attempts=attempts,
            issues=list(dict.fromkeys(issues_seen)),
            repairs=list(dict.fromkeys(repairs_done)),
            passed=bool(verdict.passed) if verdict else True,
            final_quality=verdict.quality if verdict else 1.0)

        return ShotResult(
            shot_id=shot.shot_id, video_path=str(final_path),
            mode="ref2v" if used_refs else "t2v", attempts=attempts,
            verdict=verdict, references_used=used_refs, prompt_used=base_prompt)

    # -------------------------------------------------- grounding + memory update
    def _ground_and_update(self, script: ParsedScript, shot: ShotSpec,
                           result: ShotResult, memory: EntityVisualMemory,
                           script_dir: Path) -> None:
        # Memory is built from ACCEPTED shots only (Sec. 3.3): a kept-best attempt
        # that still failed verification must not seed references.
        if result.verdict is not None and not result.verdict.passed:
            log.info("shot %d kept but not accepted; skipping memory update",
                     shot.shot_id)
            return
        entities = script.entities_of(shot)
        if not entities:
            return
        try:
            frames = sample_frames(result.video_path, self.cfg.grounding.frames_per_shot)
        except ValueError as e:
            log.warning("cannot sample frames for grounding: %s", e)
            return
        dets = self.grounder.detect(frames, entities)
        by_ent = self.grounder.best_per_entity(
            dets, top_k=self.cfg.memory.max_refs_per_shot_per_entity + 1)
        crops_dir = script_dir / "crops"

        for ent in entities:
            if ent.entity_type == EntityType.LOCATION:
                continue
            contributed = 0
            for k, det in enumerate(by_ent.get(ent.entity_id, [])):
                crop_path = save_image(
                    det.crop,
                    crops_dir / f"s{shot.shot_id:03d}_{ent.entity_id}_{k}.png")
                vlm_rel = None
                if ent.entity_type == EntityType.OBJECT:
                    vlm_rel = self._object_check(ent, det.crop)
                    if vlm_rel is not None and vlm_rel < 0.3:
                        continue
                q = score_crop(ent.entity_type, det.crop, det.score, vlm_rel)
                mem_path = save_image(
                    det.crop,
                    memory.store_dir / f"{ent.entity_id}_s{shot.shot_id:03d}_{k}.png")
                status = memory.register(ent, mem_path, q, shot.shot_id, contributed)
                log.info("memory %s <- shot %d crop%d: %s (q=%.2f)",
                         ent.entity_id, shot.shot_id, k, status, q.quality)
                if status.startswith("admit"):
                    contributed += 1

        # Location scene reference (Sec. 3.3 location grounding).
        for ent in entities:
            if ent.entity_type != EntityType.LOCATION:
                continue
            out = memory.store_dir / f"{ent.entity_id}_s{shot.shot_id:03d}_scene.png"
            path, q = self.scene.build_scene_reference(frames, dets, ent,
                                                       self.grounder, out)
            if path and q:
                status = memory.register(ent, path, q, shot.shot_id, 0)
                log.info("memory %s <- shot %d scene: %s (q=%.2f)",
                         ent.entity_id, shot.shot_id, status, q.quality)

    def _t2i_bootstrap(self, script: ParsedScript, memory: EntityVisualMemory) -> None:
        """Optional optimization: seed canonical references with T2I before any video
        generation (see GroundShotConfig.t2i_bootstrap)."""
        from .utils import falapi
        from .utils.media import load_image
        for ent in script.entities.values():
            if ent.entity_type == EntityType.LOCATION or memory.has_usable(ent.entity_id):
                continue
            prompt = (f"{script.style_layer} Studio-quality full reference portrait of "
                      f"{ent.description}. Front view, sharp focus, plain background."
                      if ent.entity_type == EntityType.CHARACTER else
                      f"{script.style_layer} Clear product-style reference photo of "
                      f"{ent.description}, complete and unoccluded.")
            try:
                res = falapi.fal_run(self.cfg.t2i_endpoint,
                                     {"prompt": prompt[:1500], "aspect_ratio": "1:1",
                                      "seed": (script.seed or 0) % (2**31)})
                path = falapi.download(
                    res["image"]["url"],
                    memory.store_dir / f"{ent.entity_id}_t2i_bootstrap.png")
            except Exception as e:  # noqa: BLE001 - bootstrap is best-effort
                log.warning("t2i bootstrap failed for %s: %s", ent.entity_id, e)
                continue
            img = load_image(path)
            vlm_rel = (self._object_check(ent, img)
                       if ent.entity_type == EntityType.OBJECT else None)
            q = score_crop(ent.entity_type, img, det_conf=0.9, vlm_reliability=vlm_rel)
            status = memory.register(ent, path, q, source_shot=0, contributed_this_shot=0,
                                     meta={"bootstrap": "t2i"})
            log.info("t2i bootstrap %s: %s (q=%.2f)", ent.entity_id, status, q.quality)

    def _object_check(self, ent: Entity, crop) -> float | None:
        """VLM semantic check for object crops (Supp. 8.1)."""
        if self.cfg.backend.name == "mock":
            return None
        try:
            out = self.llm.json_call(
                prompts.CROP_CHECK_SYSTEM,
                prompts.CROP_CHECK_USER.format(
                    entity_name=ent.name, entity_type=ent.entity_type.value,
                    entity_desc=ent.description),
                images_b64=[to_b64_jpeg(crop, max_side=512)])
            if not out.get("match", True):
                return 0.0
            return float(out.get("reliability", 0.5))
        except Exception as e:  # noqa: BLE001
            log.warning("object VLM check failed: %s", e)
            return None

    # --------------------------------------------------------------------- io
    @staticmethod
    def _save_results(script: ParsedScript, results: dict[int, ShotResult],
                      path: Path) -> None:
        recs = []
        for sid in sorted(results):
            r = results[sid]
            rec = asdict(r)
            if r.verdict is not None:
                rec["verdict"] = {
                    "quality": r.verdict.quality, "passed": r.verdict.passed,
                    "issues": [i.issue_type.value for i in r.verdict.issues]}
            recs.append(rec)
        path.write_text(json.dumps(
            {"script_id": script.script_id, "shots": recs}, indent=1))
