#!/usr/bin/env python3
"""Keeping an 8 GB card alive across a ten-hour day.

The production failure was not a crash: after a few hours (or within a minute
under four concurrent callers) every request returned
`{"detail":"Error: AcceleratorError: CUDA error: out of memory"}` in ~0.1s and
never recovered, while `/api/health` cheerfully answered `{"status":"ok"}` and
GPU *utilisation* read 5 %. Memory was full; the card was idle; every graph
looked fine.

This module holds the parts of the fix that are not specific to OmniVoice:

  * `guarded()` — inference_mode, an explicit `del`, `gc.collect()` and
    `empty_cache()` in that order, plus OOM retry and an optional reload.
  * `snapshot()` — allocated **and** reserved, so fragmentation is visible.
    `empty_cache()` before measuring is what makes a leak test blind to the
    exact bug it was written for.
  * `Readiness` — a cached self-test result. "Ready" has to mean *I generated
    words a minute ago*, not *the process is alive*.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from typing import Callable, Dict, Optional

log = logging.getLogger("omnivoice.gpu")

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


# ---------------------------------------------------------------------------
# Classifying the failure
# ---------------------------------------------------------------------------
def is_oom(exc: BaseException) -> bool:
    if torch is not None:
        oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_cls is not None and isinstance(exc, oom_cls):
            return True
    msg = str(exc).lower()
    return ("out of memory" in msg
            or "cuda error" in msg
            or type(exc).__name__ in ("OutOfMemoryError", "AcceleratorError"))


def is_allocation_failure(exc: BaseException) -> bool:
    """Did this fail because the card ran out of room, in any of its shapes?

    `torch.cuda.OutOfMemoryError` is only the tidiest one. A batch measured on
    Windows never raised it at all: allocations came back as plain
    RuntimeErrors from cuBLAS and cuDNN, whose workspaces are allocated
    outside torch's caching allocator and report failure in their own words.
    Matching only the tidy class is why a whole session recorded `oom_total: 0`
    while three segments died repeatedly.
    """
    if is_oom(exc):
        return True
    msg = str(exc).lower()
    return any(s in msg for s in (
        "cudaerrormemoryallocation", "cuda_error_out_of_memory",
        "cublas_status_alloc_failed", "cudnn_status_alloc_failed",
        "failed to allocate", "insufficient", "no kernel image",
    ))


def spilled_to_shared() -> bool:
    """Has CUDA started backing "VRAM" with system RAM?

    On Windows (WDDM) the driver will quietly satisfy allocations out of shared
    system memory once the card is full, so nothing raises — it just gets
    slower and slower until something gives up mid-request. The tell is
    `reserved` exceeding the card's own total, which is physically impossible
    otherwise. Measured on the batch that produced this: reserved 9008 MB on an
    8191 MB card, `free_mb: 0`, and every counter still reading zero.
    """
    snap = snapshot()
    total = snap.get("total_mb") or 0
    return bool(total and snap.get("reserved_mb", 0) > total * 1.02)


def reclaim() -> Dict:
    """Free what can be freed, and say how much came back."""
    before = snapshot().get("free_mb", 0)
    free_cuda()
    after = snapshot().get("free_mb", 0)
    return {"free_before_mb": before, "free_after_mb": after,
            "recovered_mb": round(after - before, 1)}


def headroom(need_mb: float, floor_mb: float = 0.0) -> Optional[Dict]:
    """Is there room to start? `None` means yes.

    Called before work begins rather than after it dies. The measured failure
    took 26-41 s to arrive and returned nothing at all — no status, no body —
    so the caller could not tell a crash from a hang. Refusing in 2 ms with a
    `Retry-After` is strictly more useful than that.
    """
    snap = snapshot()
    if snap.get("device") != "cuda":
        return None
    free = snap.get("free_mb", 0.0)
    want = max(float(need_mb), float(floor_mb))
    if free >= want and not spilled_to_shared():
        return None
    got = reclaim()                       # a full cache release may be enough
    free = got["free_after_mb"]
    if free >= want and not spilled_to_shared():
        return None
    return {
        "free_mb": free,
        "needed_mb": round(want, 1),
        "short_by_mb": round(want - free, 1),
        "fragmentation_mb": snapshot().get("fragmentation_mb", 0.0),
        "spilled_to_shared": spilled_to_shared(),
        "recovered_mb": got["recovered_mb"],
    }


def is_sticky_cuda_error(exc: BaseException) -> bool:
    """A raw `CUDA error: ...` usually poisons the context: every later call
    fails in milliseconds. That is the 0.1s-forever symptom, and retrying it
    is a waste — the process needs a reload or a restart."""
    msg = str(exc).lower()
    return "cuda error" in msg and "tried to allocate" not in msg


def free_cuda() -> None:
    """Order matters: `empty_cache()` does nothing while Python still holds a
    reference, so the collect has to come first."""
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:  # pragma: no cover
            pass


# ---------------------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------------------
def expandable_segments_state() -> Dict:
    """Did PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True actually take?

    It is the one setting the launcher names as the cure for "GPU at 5 % and
    out of memory", and on Windows torch accepts the variable, warns once into
    the log, and carries on with the ordinary allocator. Measured on
    torch 2.8.0+cu128 / Windows 11: the warning is "expandable_segments not
    supported on this platform".

    An inert setting that looks set is worse than one that is plainly off, so
    report it rather than leaving it to a UserWarning nobody reads.
    """
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    requested = "expandable_segments:true" in conf.lower().replace(" ", "")
    # Windows (WDDM) has no support for it in any torch release so far.
    supported = not sys.platform.startswith("win")
    return {
        "requested": requested,
        "effective": bool(requested and supported),
        "note": ("ignored on Windows — fragmentation is held down by "
                 "concurrency=1 and explicit cache release instead"
                 if requested and not supported else ""),
    }


def snapshot() -> Dict:
    """Both numbers, always. `reserved - allocated` is the fragmentation that
    makes a card OOM at 5 % utilisation."""
    if torch is None or not torch.cuda.is_available():
        return {"device": "cpu"}
    try:
        alloc = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        peak = torch.cuda.max_memory_allocated()
        try:
            free_b, total_b = torch.cuda.mem_get_info()
        except Exception:
            free_b, total_b = 0, torch.cuda.get_device_properties(0).total_memory
        mb = 1024.0 * 1024.0
        return {
            "device": "cuda",
            "name": torch.cuda.get_device_name(0),
            "allocated_mb": round(alloc / mb, 1),
            "reserved_mb": round(reserved / mb, 1),
            "peak_allocated_mb": round(peak / mb, 1),
            "fragmentation_mb": round((reserved - alloc) / mb, 1),
            "free_mb": round(free_b / mb, 1),
            "total_mb": round(total_b / mb, 1),
        }
    except Exception as e:  # pragma: no cover
        return {"device": "cuda", "error": f"{type(e).__name__}: {e}"}


@contextmanager
def inference():
    """`inference_mode` where available — without it autograd keeps every
    intermediate tensor of every request alive, which is the most common cause
    of exactly this failure shape."""
    if torch is None:
        yield
        return
    try:
        with torch.inference_mode():
            yield
    except AttributeError:  # pragma: no cover - very old torch
        with torch.no_grad():
            yield


def seed_everything(seed: Optional[int]) -> None:
    """Make `seed` mean something, so a golden-file test is not silently
    non-deterministic."""
    if seed is None or torch is None:
        return
    try:
        import random

        import numpy as np

        random.seed(int(seed))
        np.random.seed(int(seed) % (2 ** 32 - 1))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception as e:  # pragma: no cover
        log.warning("could not set seed %s: %s", seed, e)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
class GpuHealth:
    """Remembers whether generation is actually working right now."""

    def __init__(self, fail_threshold: int = 2):
        self._lock = threading.Lock()
        self.ok = True
        self.consecutive_failures = 0
        self.oom_total = 0
        self.last_error = ""
        self.last_error_at = 0.0
        self.last_ok_at = 0.0
        self.reloads = 0
        self.fail_threshold = fail_threshold

    def mark_ok(self) -> None:
        with self._lock:
            self.ok = True
            self.consecutive_failures = 0
            self.last_ok_at = time.time()

    def mark_failed(self, exc: BaseException) -> None:
        with self._lock:
            self.consecutive_failures += 1
            self.last_error = f"{type(exc).__name__}: {exc}"[:400]
            self.last_error_at = time.time()
            if is_allocation_failure(exc):
                self.oom_total += 1
            if is_sticky_cuda_error(exc) or self.consecutive_failures >= self.fail_threshold:
                self.ok = False

    def mark_reloaded(self) -> None:
        with self._lock:
            self.reloads += 1

    def as_dict(self) -> Dict:
        with self._lock:
            return {
                "generation_ok": self.ok,
                "consecutive_failures": self.consecutive_failures,
                "oom_total": self.oom_total,
                "model_reloads": self.reloads,
                "last_error": self.last_error or None,
                "last_error_age_s": (round(time.time() - self.last_error_at, 1)
                                     if self.last_error_at else None),
                "last_success_age_s": (round(time.time() - self.last_ok_at, 1)
                                       if self.last_ok_at else None),
            }


class Readiness:
    """`ready` means: words came out of the model recently.

    Reading VRAM and a flag is what let a dead server report `{"status":"ok"}`
    while every generation returned 500. Only an actual generation answers the
    question a client is asking.
    """

    def __init__(self, stale_after: float = 300.0):
        self._lock = threading.Lock()
        self.ok = False
        self.at = 0.0
        self.detail = "no self-test has run yet"
        self.stale_after = stale_after

    def record(self, ok: bool, detail: str) -> None:
        with self._lock:
            self.ok, self.at, self.detail = ok, time.time(), detail

    def as_dict(self) -> Dict:
        with self._lock:
            age = (time.time() - self.at) if self.at else None
            fresh = bool(self.ok and age is not None and age <= self.stale_after)
            return {
                "ready": fresh,
                "detail": self.detail,
                "last_selftest_age_s": round(age, 1) if age is not None else None,
                "stale_after_s": self.stale_after,
            }

    @property
    def is_ready(self) -> bool:
        return bool(self.as_dict()["ready"])


# ---------------------------------------------------------------------------
# Running work
# ---------------------------------------------------------------------------
def guarded(fn: Callable, *, retries: int = 1, health: Optional[GpuHealth] = None,
            on_reload: Optional[Callable[[], None]] = None, label: str = "generate"):
    """Run `fn()` under inference_mode with OOM recovery.

    On OOM: free properly, retry once. If the error is a sticky CUDA error, a
    retry cannot help — reload the model if a reload hook was given, otherwise
    mark the GPU unhealthy so `/api/ready` starts answering 503 instead of
    letting a caller waste a twenty-minute batch.
    """
    attempt = 0
    while True:
        result = None
        try:
            with inference():
                result = fn()
            if health:
                health.mark_ok()
            return result
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            del result
            if not is_allocation_failure(exc):
                raise
            if health:
                health.mark_failed(exc)
            free_cuda()
            sticky = is_sticky_cuda_error(exc)
            log.warning("%s hit OOM (attempt %d/%d, sticky=%s): %s",
                        label, attempt + 1, retries + 1, sticky, exc)
            if sticky and on_reload is not None:
                try:
                    on_reload()
                    if health:
                        health.mark_reloaded()
                except Exception as e:  # noqa: BLE001
                    log.error("model reload failed: %s", e)
                    raise exc from e
            elif sticky:
                raise
            attempt += 1
            if attempt > retries:
                raise
        finally:
            gc.collect()
