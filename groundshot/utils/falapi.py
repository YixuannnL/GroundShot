"""Thin wrapper around the fal.ai queue API (requests-based, no fal_client dependency).

All fal calls in the project route through `fal_run` / `fal_upload` so that retries,
timeouts, and cost logging live in one place.
"""
from __future__ import annotations

import mimetypes
import os
import time
from pathlib import Path

import requests

QUEUE_BASE = "https://queue.fal.run"
REST_BASE = "https://rest.alpha.fal.ai"


class FalError(RuntimeError):
    pass


def _key() -> str:
    key = os.environ.get("FAL_KEY", "")
    if not key:
        raise FalError("FAL_KEY not set; put it in .env")
    return key


def _headers() -> dict:
    return {"Authorization": f"Key {_key()}"}


def fal_upload(path: str | Path) -> str:
    """Upload a local file to fal storage; returns a public URL usable as an input."""
    path = Path(path)
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    r = requests.post(
        f"{REST_BASE}/storage/upload/initiate",
        headers=_headers(),
        json={"file_name": path.name, "content_type": content_type},
        timeout=60,
    )
    r.raise_for_status()
    info = r.json()
    with open(path, "rb") as f:
        up = requests.put(info["upload_url"], data=f,
                          headers={"Content-Type": content_type}, timeout=300)
    up.raise_for_status()
    return info["file_url"]


def fal_run(endpoint: str, payload: dict, poll_interval: float = 5.0,
            timeout: float = 900.0, max_retries: int = 2) -> dict:
    """Submit to the fal queue, poll until completion, return the result JSON."""
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return _run_once(endpoint, payload, poll_interval, timeout)
        except FalError as e:
            last_err = e
            msg = str(e).lower()
            if any(k in msg for k in ("budget", "unauthorized", "403",
                                      "not enterprise ready", "401")):
                raise  # not transient; retrying would just resubmit and fail again
            time.sleep(5 * (attempt + 1))
    raise FalError(f"fal_run failed after retries: {last_err}")


def _run_once(endpoint: str, payload: dict, poll_interval: float, timeout: float) -> dict:
    r = requests.post(f"{QUEUE_BASE}/{endpoint}", headers=_headers(), json=payload, timeout=60)
    if r.status_code >= 400:
        raise FalError(f"submit {endpoint} -> {r.status_code}: {r.text[:500]}")
    sub = r.json()
    status_url = sub.get("status_url")
    response_url = sub.get("response_url")
    if not status_url:
        return sub  # synchronous response
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = requests.get(status_url, headers=_headers(), timeout=60)
        if s.status_code >= 400:
            raise FalError(f"status -> {s.status_code}: {s.text[:300]}")
        st = s.json()
        if st.get("status") == "COMPLETED":
            res = requests.get(response_url, headers=_headers(), timeout=60)
            if res.status_code >= 400:
                raise FalError(f"result -> {res.status_code}: {res.text[:500]}")
            return res.json()
        if st.get("status") in ("FAILED", "CANCELLED", "ERROR"):
            raise FalError(f"job failed: {st}")
        time.sleep(poll_interval)
    raise FalError(f"timed out after {timeout}s waiting for {endpoint}")


def download(url: str, dest: str | Path) -> str:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return str(dest)
