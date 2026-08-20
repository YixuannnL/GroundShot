"""Video generation backend interface (Eq. 4: G_mi, mi in {T2V, Ref2V})."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class VideoBackend(ABC):
    name: str = "base"
    max_reference_images: int = 4

    @abstractmethod
    def t2v(self, prompt: str, out_path: Path, duration: int, seed: int | None = None) -> str:
        """Generate a video from text only; returns the local video path."""

    @abstractmethod
    def ref2v(self, prompt: str, reference_images: list[str], out_path: Path,
              duration: int, seed: int | None = None) -> str:
        """Generate a video conditioned on reference images; returns the local path."""


def make_backend(cfg) -> VideoBackend:
    from ..config import BackendConfig
    assert isinstance(cfg, BackendConfig)
    if cfg.name == "fal_vidu":
        from .fal_vidu import FalViduBackend
        return FalViduBackend(cfg)
    if cfg.name == "phantom":
        from .phantom import PhantomBackend
        return PhantomBackend(cfg)
    if cfg.name == "mock":
        from .mock import MockBackend
        return MockBackend(cfg)
    raise ValueError(f"Unknown backend: {cfg.name}")
