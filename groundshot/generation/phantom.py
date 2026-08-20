"""Open-source backend: Wan2.1 (T2V) + Phantom-Wan-14B (Ref2V), for a Linux GPU server.

This backend shells out to the official repos' generate.py, so it needs:

  1. git clone https://github.com/Wan-Video/Wan2.1          -> third_party/Wan2.1
  2. git clone https://github.com/Phantom-video/Phantom     -> third_party/Phantom
  3. Checkpoints (huggingface-cli download ...):
       Wan-AI/Wan2.1-T2V-14B          -> weights/Wan2.1-T2V-14B
       bytedance-research/Phantom     -> weights/Phantom-Wan-14B
  4. pip install -r third_party/Wan2.1/requirements.txt  (same env works for Phantom)

Hardware: 14B checkpoints want >=48GB VRAM single-GPU, or use --ulysses_size/--ring_size
for multi-GPU (see the repos). On smaller GPUs switch to the 1.3B variants by
pointing the ckpt dirs and templates below at them.

The exact CLI of both repos occasionally changes; the command templates are
config-visible strings so they can be fixed without touching code.
"""
from __future__ import annotations

import logging
import shlex
import subprocess
from pathlib import Path

from ..config import BackendConfig, ROOT
from .base import VideoBackend

log = logging.getLogger("groundshot.phantom")

WAN_T2V_TEMPLATE = (
    "python generate.py --task t2v-14B --size {size} --frame_num {frames} "
    "--ckpt_dir {wan_ckpt} --prompt {prompt} --base_seed {seed} "
    "--save_file {out}"
)
PHANTOM_S2V_TEMPLATE = (
    "python generate.py --task s2v-14B --size {size} --frame_num {frames} "
    "--ckpt_dir {wan_ckpt} --phantom_ckpt {phantom_ckpt} "
    "--ref_image {refs} --prompt {prompt} --base_seed {seed} "
    "--save_file {out}"
)


class PhantomBackend(VideoBackend):
    name = "phantom"

    def __init__(self, cfg: BackendConfig):
        self.cfg = cfg
        self.max_reference_images = 4   # Phantom supports up to 4 subject references
        self.wan_dir = ROOT / "third_party" / "Wan2.1"
        self.phantom_dir = ROOT / "third_party" / "Phantom"
        self.fps = 16

    def _frames(self, duration: int) -> int:
        # Wan/Phantom expect 4k+1 frame counts (e.g. 65 for ~4s @16fps).
        n = int(duration * self.fps)
        return (n // 4) * 4 + 1

    def _run(self, cmd: str, cwd: Path) -> None:
        log.info("phantom backend: %s (cwd=%s)", cmd, cwd)
        proc = subprocess.run(shlex.split(cmd), cwd=cwd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"generation failed (rc={proc.returncode}):\n{proc.stderr[-2000:]}")

    def t2v(self, prompt: str, out_path: Path, duration: int, seed: int | None = None) -> str:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = WAN_T2V_TEMPLATE.format(
            size=self.cfg.phantom_size, frames=self._frames(duration),
            wan_ckpt=shlex.quote(str(ROOT / self.cfg.wan_ckpt_dir)),
            prompt=shlex.quote(prompt), seed=seed or 42,
            out=shlex.quote(str(out_path)))
        self._run(cmd, self.wan_dir)
        return str(out_path)

    def ref2v(self, prompt: str, reference_images: list[str], out_path: Path,
              duration: int, seed: int | None = None) -> str:
        refs = reference_images[: self.max_reference_images]
        if not refs:
            return self.t2v(prompt, out_path, duration, seed)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = PHANTOM_S2V_TEMPLATE.format(
            size=self.cfg.phantom_size, frames=self._frames(duration),
            wan_ckpt=shlex.quote(str(ROOT / self.cfg.wan_ckpt_dir)),
            phantom_ckpt=shlex.quote(str(ROOT / self.cfg.phantom_ckpt_dir)),
            refs=shlex.quote(",".join(refs)),
            prompt=shlex.quote(prompt), seed=seed or 42,
            out=shlex.quote(str(out_path)))
        self._run(cmd, self.phantom_dir)
        return str(out_path)
