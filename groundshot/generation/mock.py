"""Offline mock backend for end-to-end pipeline testing without API keys or cost.

Rendering is deterministic per prompt/seed: each entity mentioned in the prompt is
drawn as a stable colored avatar; when reference images are provided they are pasted
(scaled, jittered) into the scene so that reference conditioning genuinely affects
the output and grounding/metrics respond to it. Use with configs/mock.yaml, which
relaxes face-based gates (mock avatars have no detectable faces).
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..config import BackendConfig
from ..utils.media import load_image, write_video
from .base import VideoBackend


def _h(s: str, mod: int, salt: str = "") -> int:
    return int(hashlib.md5((salt + s).encode()).hexdigest(), 16) % mod


class MockBackend(VideoBackend):
    name = "mock"

    def __init__(self, cfg: BackendConfig):
        self.cfg = cfg
        self.max_reference_images = cfg.max_reference_images
        self.size = (640, 360)
        self.fps = 8

    # ------------------------------------------------------------------ public
    def t2v(self, prompt: str, out_path: Path, duration: int, seed: int | None = None) -> str:
        return self._render(prompt, [], out_path, duration, seed or 0)

    def ref2v(self, prompt: str, reference_images: list[str], out_path: Path,
              duration: int, seed: int | None = None) -> str:
        return self._render(prompt, reference_images[: self.max_reference_images],
                            out_path, duration, seed or 0)

    # ---------------------------------------------------------------- rendering
    def _render(self, prompt: str, refs: list[str], out_path: Path,
                duration: int, seed: int) -> str:
        rng = np.random.default_rng(seed + _h(prompt, 10_000))
        W, H = self.size
        n_frames = max(4, int(self.fps * min(duration, 2)))

        bg_hue = _h(prompt.lower()[:120], 360, "bg")   # style+location words dominate
        bg = self._gradient(W, H, bg_hue)

        # Entities: words that look like subjects get avatars; refs replace avatars.
        subjects = self._subjects(prompt)
        frames = []
        for t in range(n_frames):
            img = bg.copy()
            draw = ImageDraw.Draw(img)
            x0 = 60
            for i, subj in enumerate(subjects[:4]):
                jitter = int(3 * np.sin(t / 3 + i))
                if i < len(refs):
                    ref = Image.fromarray(load_image(refs[i]))
                    ref.thumbnail((W // 3, int(H * 0.7)))
                    img.paste(ref, (x0, H - ref.height - 30 + jitter))
                    x0 += ref.width + 30
                else:
                    self._avatar(draw, subj, x0, H, jitter)
                    x0 += 130
            frames.append(np.array(img))
        write_video(frames, out_path, fps=self.fps)
        return str(out_path)

    @staticmethod
    def _subjects(prompt: str) -> list[str]:
        words = re.findall(r"[a-zA-Z]{4,}", prompt.lower())
        keys = [w for w in words if w in
                ("woman", "man", "person", "detective", "worker", "chef", "guard",
                 "briefcase", "jacket", "plant", "child", "girl", "boy", "figure")]
        seen, out = set(), []
        for k in keys:
            if k not in seen:
                seen.add(k)
                out.append(k)
        return out or ["figure"]

    @staticmethod
    def _gradient(W: int, H: int, hue: int) -> Image.Image:
        import colorsys
        top = colorsys.hsv_to_rgb(hue / 360, 0.35, 0.85)
        bot = colorsys.hsv_to_rgb(hue / 360, 0.55, 0.45)
        rows = np.linspace(top, bot, H)
        arr = (np.tile(rows[:, None, :], (1, W, 1)) * 255).astype(np.uint8)
        return Image.fromarray(arr)

    @staticmethod
    def _avatar(draw: ImageDraw.ImageDraw, key: str, x: int, H: int, jitter: int) -> None:
        import colorsys
        hue = _h(key, 360, "ent")
        col = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue / 360, 0.8, 0.9))
        y = H - 40 + jitter
        draw.rectangle([x, y - 120, x + 90, y], fill=col)                 # body
        draw.ellipse([x + 15, y - 180, x + 75, y - 120], fill=col)        # head
        draw.text((x, y + 6), key, fill="white")
