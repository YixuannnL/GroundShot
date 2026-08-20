"""Video / image IO helpers. Uses OpenCV for reading and imageio-ffmpeg for writing,
so no system ffmpeg is required."""
from __future__ import annotations

import base64
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def sample_frames(video_path: str | Path, n: int = 8) -> list[np.ndarray]:
    """Uniformly sample n RGB frames from a video."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"Cannot read frames from {video_path}")
    idxs = np.linspace(0, total - 1, min(n, total)).astype(int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"No decodable frames in {video_path}")
    return frames


def last_frame(video_path: str | Path) -> np.ndarray:
    return sample_frames(video_path, n=64)[-1]


def write_video(frames: list[np.ndarray], path: str | Path, fps: int = 16) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()


def save_image(img: np.ndarray | Image.Image, path: str | Path) -> str:
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))
    return str(path)


def load_image(path: str | Path) -> np.ndarray:
    return np.array(Image.open(str(path)).convert("RGB"))


def to_b64_jpeg(img: np.ndarray | Image.Image, max_side: int = 768, quality: int = 85) -> str:
    """Encode an image as base64 JPEG for VLM calls, downscaling to bound payload size."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def sharpness_score(img: np.ndarray) -> float:
    """Normalized Laplacian-variance sharpness in [0,1]."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Typical crisp crops land around var>=300; map log-scale to [0,1].
    return float(np.clip(np.log10(max(var, 1.0)) / np.log10(1500.0), 0.0, 1.0))


def frame_grid(frames: list[np.ndarray], cols: int = 2) -> Image.Image:
    """Tile frames into a labeled grid image (for VLM critique)."""
    ims = [Image.fromarray(f) for f in frames]
    w = min(im.width for im in ims)
    h = min(im.height for im in ims)
    ims = [im.resize((w, h)) for im in ims]
    rows = (len(ims) + cols - 1) // cols
    grid = Image.new("RGB", (cols * w, rows * h), "black")
    for i, im in enumerate(ims):
        grid.paste(im, ((i % cols) * w, (i // cols) * h))
    return grid
