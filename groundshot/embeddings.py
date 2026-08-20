"""Lazy-loaded embedding models: CLIP ViT-B/32 (redundancy gate + EAF), DINOv2
(identity/scene similarity), shared by the pipeline and the UED evaluator.

All models load on first use and cache on the best available device.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import torch
from PIL import Image

log = logging.getLogger("groundshot.embed")


def pick_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def _clip():
    from transformers import CLIPModel, CLIPProcessor
    device = pick_device()
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    log.info("CLIP ViT-B/32 loaded on %s", device)
    return model, proc, device


@lru_cache(maxsize=1)
def _dinov2():
    from transformers import AutoImageProcessor, AutoModel
    device = pick_device()
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    log.info("DINOv2-base loaded on %s", device)
    return model, proc, device


def _to_pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return Image.fromarray(np.asarray(img)).convert("RGB")


def _as_tensor(feats) -> torch.Tensor:
    # transformers>=5 returns BaseModelOutputWithPooling (pooler_output already
    # projected to the joint space); <5 returns the tensor directly.
    return feats if torch.is_tensor(feats) else feats.pooler_output


@torch.no_grad()
def clip_image_embed(images: list) -> np.ndarray:
    model, proc, device = _clip()
    inputs = proc(images=[_to_pil(i) for i in images], return_tensors="pt").to(device)
    feats = _as_tensor(model.get_image_features(**inputs))
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()


@torch.no_grad()
def clip_text_embed(texts: list[str]) -> np.ndarray:
    model, proc, device = _clip()
    inputs = proc(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    feats = _as_tensor(model.get_text_features(**inputs))
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()


@torch.no_grad()
def dino_embed(images: list) -> np.ndarray:
    model, proc, device = _dinov2()
    inputs = proc(images=[_to_pil(i) for i in images], return_tensors="pt").to(device)
    out = model(**inputs)
    feats = out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:, 0]
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
