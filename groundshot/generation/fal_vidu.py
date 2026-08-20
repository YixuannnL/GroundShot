"""Vidu Q3 backend via fal.ai (paper's primary backend: viduq3-pro / viduq3-mix).

T2V:   fal-ai/vidu/q3/text-to-video
Ref2V: fal-ai/vidu/q3/reference-to-video/mix  (1-4 reference images)

Pricing (fal, 2026-08): $0.035/video-second at 360p/540p, x2.2 for 720p/1080p.
A 4s 720p shot therefore costs ~$0.31; the backend logs a running total.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..config import BackendConfig
from ..utils import falapi
from .base import VideoBackend

log = logging.getLogger("groundshot.vidu")

_RATE = {"360p": 0.035, "540p": 0.035, "720p": 0.077, "1080p": 0.077}


class FalViduBackend(VideoBackend):
    name = "fal_vidu"

    def __init__(self, cfg: BackendConfig):
        self.cfg = cfg
        self.max_reference_images = cfg.max_reference_images
        self.total_cost = 0.0
        self.n_calls = 0
        self._url_cache: dict[str, str] = {}

    def _common(self, prompt: str, duration: int, seed: int | None) -> dict:
        payload = {
            "prompt": prompt[:2000],
            "duration": int(duration),
            "aspect_ratio": self.cfg.aspect_ratio,
            "resolution": self.cfg.resolution,
            "audio": self.cfg.audio,
        }
        if seed is not None:
            payload["seed"] = int(seed) % (2**31)
        return payload

    def _track(self, duration: int) -> None:
        self.n_calls += 1
        self.total_cost += _RATE.get(self.cfg.resolution, 0.077) * duration
        log.info("vidu call #%d, est. total cost $%.2f", self.n_calls, self.total_cost)

    def _upload(self, path: str) -> str:
        if path not in self._url_cache:
            self._url_cache[path] = falapi.fal_upload(path)
        return self._url_cache[path]

    def t2v(self, prompt: str, out_path: Path, duration: int, seed: int | None = None) -> str:
        res = falapi.fal_run(self.cfg.t2v_endpoint, self._common(prompt, duration, seed),
                             poll_interval=self.cfg.poll_interval, timeout=self.cfg.timeout)
        self._track(duration)
        return falapi.download(res["video"]["url"], out_path)

    def ref2v(self, prompt: str, reference_images: list[str], out_path: Path,
              duration: int, seed: int | None = None) -> str:
        refs = reference_images[: self.max_reference_images]
        if not refs:
            return self.t2v(prompt, out_path, duration, seed)
        ep = self.cfg.ref2v_endpoint
        if "/q3/" in ep or "/q2/" in ep:
            payload = self._common(prompt, duration, seed)
        else:   # Vidu 2.0 reference-to-video: fixed 4s/720p, no duration/resolution
            payload = {"prompt": prompt[:1500],
                       "aspect_ratio": self.cfg.aspect_ratio,
                       "movement_amplitude": "auto"}
            if seed is not None:
                payload["seed"] = int(seed) % (2**31)
        payload["reference_image_urls"] = [self._upload(p) for p in refs]
        res = falapi.fal_run(ep, payload,
                             poll_interval=self.cfg.poll_interval, timeout=self.cfg.timeout)
        self.n_calls += 1
        self.total_cost += 0.40 if "/q3/" not in ep and "/q2/" not in ep else \
            _RATE.get(self.cfg.resolution, 0.154) * duration
        log.info("vidu call #%d, est. total cost $%.2f", self.n_calls, self.total_cost)
        return falapi.download(res["video"]["url"], out_path)
