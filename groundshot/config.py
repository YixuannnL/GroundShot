"""Central configuration for GroundShot.

Every threshold from the paper (main text Sec. 3-4 and Supp. Sec. 8-9) lives here,
plus the values the paper leaves unspecified (marked UNSPECIFIED-IN-PAPER) with our
defaults. Override any field via a YAML file passed to `load_config`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _load_env() -> None:
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_env()


@dataclass
class MemoryConfig:
    # --- Multi-gate registration (Supp. 8.1) ---
    tau_min: float = 0.4            # composite quality floor q(c) >= 0.4
    char_face_conf_min: float = 0.3 # face detection confidence floor for character crops
    clip_redundancy_max: float = 0.92  # reject if max CLIP cos to existing refs >= 0.92
    max_refs_per_shot_per_entity: int = 2
    max_active_refs: int = 6        # 1 canonical + up to 5 auxiliaries

    # --- Canonical initialization (Supp. 8.3) ---
    canonical_quality_min: float = 0.85
    canonical_face_conf_min: float = 0.7
    canonical_frontality_min: float = 0.5   # "near-frontal": UNSPECIFIED-IN-PAPER, |yaw| <= 45 deg

    # --- Identity gate sim(c, r*) >= theta_id (Eq. 3/6). UNSPECIFIED-IN-PAPER values. ---
    theta_id_face: float = 0.40     # ArcFace cosine, characters with two valid faces
    theta_id_dino_char: float = 0.50  # DINOv2 cosine fallback for characters
    theta_id_dino_obj: float = 0.55
    theta_id_dino_loc: float = 0.50

    # --- Eviction (Supp. 8.4) ---
    evict_low_quality: float = 0.5
    cleanup_every_n_shots: int = 5

    # --- Gender gate (OUR FIX, see README "deviations"; not in the paper). ---
    # GroundingDINO binds attribute phrases weakly: with two characters in frame,
    # both queries can land on the same person, and during cold start (no canonical
    # yet) the identity gate has nothing to compare against, so a wrong-person crop
    # can poison the provisional canonical. When the script states a character's
    # gender, reject crops whose confidently-detected face disagrees.
    gender_gate: bool = True
    gender_gate_face_conf_min: float = 0.5  # trust genderage only on confident faces

    # --- Provisional-canonical fallback (OUR FIX, see README "deviations"). ---
    # Strict paper mode (Supp. 8.3) rejects every sub-canonical candidate while the
    # canonical slot is empty, which can deadlock an entity into pure-T2V forever.
    provisional_canonical: bool = True
    provisional_quality_min: float = 0.60


@dataclass
class SchedulerConfig:
    # Reference-quality estimation is batched into the parsing LLM call.
    # Cycle resolution order: entity priority > quality gain > narrative proximity (Sec. 3.2).
    min_quality_gain: float = 0.05  # ignore reorderings whose predicted qsrc gain is below this


@dataclass
class SelectorConfig:
    mode: str = "hybrid"            # "traditional" | "agent" | "hybrid" (Supp. 9.1)
    kmax: int = 3                   # canonical + up to Kmax-1 auxiliaries shown to the selector
    min_agent_confidence: float = 0.3
    # Eq. 7 weights (characters): 0.4 sharpness + 0.4 face conf + 0.2 frontality
    w_sharp: float = 0.4
    w_face: float = 0.4
    w_front: float = 0.2


@dataclass
class VerifyConfig:
    tau_pass: float = 0.70          # UNSPECIFIED-IN-PAPER; consistent with 83.1% first-pass rate
    retry_budget: int = 2           # UNSPECIFIED-IN-PAPER; consistent with 1.20 gens/shot
    mode: str = "risk_gated"        # "full" | "risk_gated" | "off"
    frames_for_critic: int = 4


@dataclass
class GroundingConfig:
    frames_per_shot: int = 8
    box_threshold: float = 0.30
    text_threshold: float = 0.25
    min_crop_area_frac: float = 0.003   # discard tiny boxes (< 0.3% of frame)
    max_crop_area_frac: float = 0.95
    # Cross-entity suppression (part of the ReferDINO->GroundingDINO adaptation):
    # per-entity queries run independently, so two entities can claim overlapping
    # boxes on the same person/object in a frame; keep only the highest-scoring
    # claim when intersection-over-min-area exceeds the threshold.
    cross_entity_suppression: bool = True
    cross_entity_iom: float = 0.7
    # Character grounding query: "category" grounds a plain noun (man/woman/person,
    # from Entity.gender) and leaves identity assignment to the gender/identity
    # gates; "name" grounds the attribute phrase (fails badly on GroundingDINO:
    # part-boxes and wrong-person binding — see README deviations).
    char_query: str = "category"
    mask_dilate_px: int = 15            # union-mask dilation before scene reconstruction
    model_id: str = "IDEA-Research/grounding-dino-base"
    device: str = "auto"                # "auto" -> cuda > mps > cpu
    # Scene reference admission (Supp. 8.2)
    scene_min_bg_frac: float = 0.35     # skip reconstruction when foreground dominates the frame
    scene_editor_endpoint: str = "fal-ai/object-removal/mask"
    # Scene reconstruction robustness (OUR ADDITIONS, README deviations): when the
    # VLM rejects the reconstruction, retry on the next-cleanest frame; if every
    # attempt fails but some frame is almost people-free, keep that raw frame at a
    # quality penalty rather than leaving the location reference-less (a missing
    # scene reference is the most visible cross-shot inconsistency).
    scene_extra_frames: int = 1          # extra next-best frames to try after a rejection
    scene_raw_fallback: bool = True
    scene_raw_fallback_bg_min: float = 0.80
    scene_raw_fallback_vlm_q: float = 0.35


@dataclass
class BackendConfig:
    name: str = "fal_vidu"          # "fal_vidu" | "phantom" | "mock"
    resolution: str = "720p"
    aspect_ratio: str = "16:9"
    duration: int = 4
    audio: bool = False
    # NOTE on endpoints: the paper's viduq3-mix maps to fal-ai/vidu/q3/reference-to-
    # video/mix (1-4 refs), but fal accounts restricted to "enterprise ready"
    # endpoints (like this user's) cannot access it; the accessible alternative is
    # fal-ai/vidu/reference-to-video (Vidu 2.0: fixed 4s/720p, $0.40/video, <=3 refs).
    # Swap back to the q3 endpoints below if your account allows them.
    max_reference_images: int = 3
    t2v_endpoint: str = "fal-ai/vidu/q3/text-to-video/turbo"
    ref2v_endpoint: str = "fal-ai/vidu/reference-to-video"
    poll_interval: float = 5.0
    timeout: float = 900.0
    # Phantom (Linux GPU server) settings
    phantom_ckpt_dir: str = "weights/Phantom-Wan-14B"
    wan_ckpt_dir: str = "weights/Wan2.1-T2V-14B"
    phantom_size: str = "1280*720"


@dataclass
class LLMConfig:
    provider: str = "auto"          # "openai" | "anthropic" | "auto" (first with a key)
    # Paper setup: GPT-4o parses, GPT-4.1 handles decisions/critique.
    openai_parse_model: str = "gpt-4o"
    openai_decision_model: str = "gpt-4.1"
    anthropic_parse_model: str = "claude-sonnet-5"
    anthropic_decision_model: str = "claude-sonnet-5"
    temperature: float = 0.2
    max_retries: int = 3


@dataclass
class EvalConfig:
    frames_per_shot: int = 8
    # UED grounding thresholds (Supp. 7.2 "fixed thresholds"; values UNSPECIFIED-IN-PAPER)
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    model_id: str = "IDEA-Research/grounding-dino-base"
    # UED grounds by YAML name+description phrase ("name"), independent of the
    # generation-side char_query setting: benchmark semantics must not shift when
    # the method's grounding is tuned. "category" is available but changes UED
    # numbers and breaks same-gender pairs (identical queries) — see README.
    char_query: str = "name"
    # Calibration mapping raw cosines to a shared same-entity score in [0,1]
    # score = clip((cos - lo) / (hi - lo)). Defaults from common operating ranges; recalibrate
    # with scripts/calibrate.py if you have held-out same/different-entity pairs.
    arcface_lo: float = 0.05
    arcface_hi: float = 0.75
    dino_lo: float = 0.30
    dino_hi: float = 0.95
    clip_eaf_lo: float = 0.15       # EAF rescale bounds for CLIP text-image cosine
    clip_eaf_hi: float = 0.40
    csc_backend: str = "clip_mean"  # "clip_mean" proxy (Mac) | "viclip" (server, if installed)
    enable_aesthetic: bool = True   # CLIP ViT-L/14 download (~1.7GB); disable for smoke runs


@dataclass
class GroundShotConfig:
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    verify: VerifyConfig = field(default_factory=VerifyConfig)
    grounding: GroundingConfig = field(default_factory=GroundingConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    runs_dir: str = str(ROOT / "runs")
    strict_paper_mode: bool = False  # disable all deviations (provisional canonical, etc.)
    # OPTIONAL OPTIMIZATION (off by default; not in the paper's text-only setting,
    # but allowed by its "memory can be initialized with user-provided references"):
    # before any video generation, synthesize one canonical reference per character/
    # object with a T2I model from the parsed description + style layer. Eliminates
    # the reference cold-start entirely.
    t2i_bootstrap: bool = False
    t2i_endpoint: str = "fal-ai/vidu/q2/text-to-image"

    def __post_init__(self):
        if self.strict_paper_mode:
            self.memory.provisional_canonical = False
            self.memory.gender_gate = False


def load_config(path: str | Path | None = None) -> GroundShotConfig:
    cfg = GroundShotConfig()
    if path:
        overrides = yaml.safe_load(Path(path).read_text()) or {}
        for section, values in overrides.items():
            if not hasattr(cfg, section):
                raise KeyError(f"Unknown config section: {section}")
            target = getattr(cfg, section)
            if isinstance(values, dict):
                for k, v in values.items():
                    if not hasattr(target, k):
                        raise KeyError(f"Unknown config key: {section}.{k}")
                    setattr(target, k, v)
            else:
                setattr(cfg, section, values)
        if cfg.strict_paper_mode:
            cfg.memory.provisional_canonical = False
            cfg.memory.gender_gate = False
    return cfg
