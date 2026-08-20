"""Dual-provider (OpenAI / Anthropic) LLM+VLM client with JSON output handling.

Paper setup: GPT-4o for parsing, GPT-4.1 for scheduling / retrieval / critique
decisions. Set llm.provider="anthropic" in config to run on Claude instead
(results then are not directly comparable to the paper's numbers).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Optional

import requests

from ..config import LLMConfig

log = logging.getLogger("groundshot.llm")


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.provider = cfg.provider
        if self.provider == "auto":
            if os.environ.get("OPENAI_API_KEY"):
                self.provider = "openai"
            elif os.environ.get("ANTHROPIC_API_KEY"):
                self.provider = "anthropic"
            else:
                raise LLMError("No OPENAI_API_KEY or ANTHROPIC_API_KEY found in env/.env")
        # provider "none": every call raises; use with the offline heuristic parser,
        # selector.mode=traditional and verify.mode=off (see configs/mock.yaml).
        self.n_text_calls = 0
        self.n_vision_calls = 0

    # ---------------------------------------------------------------- public
    def json_call(self, system: str, user: str, images_b64: Optional[list[str]] = None,
                  role: str = "decision", max_tokens: int = 4096) -> dict:
        """Run a chat call and parse a JSON object from the reply."""
        text = self.call(system, user, images_b64, role, max_tokens)
        return extract_json(text)

    def call(self, system: str, user: str, images_b64: Optional[list[str]] = None,
             role: str = "decision", max_tokens: int = 4096) -> str:
        if self.provider == "none":
            raise LLMError("LLM provider is 'none' (offline mode)")
        if images_b64:
            self.n_vision_calls += 1
        else:
            self.n_text_calls += 1
        last: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                if self.provider == "openai":
                    return self._openai(system, user, images_b64, role, max_tokens)
                return self._anthropic(system, user, images_b64, role, max_tokens)
            except Exception as e:  # noqa: BLE001 - retry any transient failure
                last = e
                log.warning("LLM call failed (attempt %d): %s", attempt + 1, e)
                time.sleep(2 ** attempt)
        raise LLMError(f"LLM call failed after retries: {last}")

    # -------------------------------------------------------------- backends
    def _model(self, role: str) -> str:
        if self.provider == "openai":
            return self.cfg.openai_parse_model if role == "parse" else self.cfg.openai_decision_model
        return self.cfg.anthropic_parse_model if role == "parse" else self.cfg.anthropic_decision_model

    def _openai(self, system: str, user: str, images_b64, role: str, max_tokens: int) -> str:
        content: list | str = user
        if images_b64:
            content = [{"type": "text", "text": user}] + [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b}"}}
                for b in images_b64
            ]
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            json={
                "model": self._model(role),
                "temperature": self.cfg.temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
            },
            timeout=180,
        )
        if r.status_code >= 400:
            raise LLMError(f"openai {r.status_code}: {r.text[:400]}")
        return r.json()["choices"][0]["message"]["content"]

    def _anthropic(self, system: str, user: str, images_b64, role: str, max_tokens: int) -> str:
        content: list = [{"type": "text", "text": user}]
        if images_b64:
            content = [
                {"type": "image",
                 "source": {"type": "base64", "media_type": "image/jpeg", "data": b}}
                for b in images_b64
            ] + content
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": self._model(role),
                "max_tokens": max_tokens,
                "temperature": self.cfg.temperature,
                "system": system,
                "messages": [{"role": "user", "content": content}],
            },
            timeout=180,
        )
        if r.status_code >= 400:
            raise LLMError(f"anthropic {r.status_code}: {r.text[:400]}")
        return "".join(b.get("text", "") for b in r.json()["content"])


def extract_json(text: str) -> dict:
    """Parse the first JSON object in a reply, tolerating markdown fences."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    start = text.find("{")
    if start == -1:
        raise LLMError(f"No JSON object in LLM reply: {text[:200]}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise LLMError("Unbalanced JSON in LLM reply")
