from __future__ import annotations

import threading
import queue
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FaissTask:
    """Single face submitted for async FAISS matching."""

    object_id: int  # nvtracker-assigned, stable across frames
    source_id: int
    frame_number: int
    embedding: np.ndarray  # shape (512,) float32


@dataclass
class MatchResult:
    """Cached FAISS match result for one tracked face."""

    name: str
    score: float
    frame_number: int  # frame at which the match was computed


def _cache_key(source_id: int, object_id: int) -> tuple[int, int]:
    """Cache key for per-stream face match; avoids cross-stream collision."""
    return (source_id, object_id)


class AsyncFaissMatcher:
    """
    Offloads FAISS search from the GStreamer streaming thread to a background
    worker, caching results by *(source_id, object_id)* so that the probe
    callback never blocks the pipeline and results do not cross streams.

    Typical usage inside a pad probe:

        # 1. check if result already cached
        cached = matcher.get_result(frame_meta.source_id, obj_meta.object_id)

        # 2. submit only when needed (new face or stale cache)
        if matcher.needs_submit(frame_meta.source_id, obj_meta.object_id, frame_number):
            matcher.submit(FaissTask(...))

        # 3. periodically clean up departed faces
        if frame_number % 300 == 0:
            matcher.cleanup(frame_number)
    """

    def __init__(
        self,
        faiss_index,
        max_queue_size: int = 8,
        stale_frames: int = 300,
        refresh_interval: int = 90,
    ):
        """
        :param faiss_index: IIndexFlatIP / IIndexFlatL2 with search_with_scores()
        :param max_queue_size: backlog limit; full queue -> task silently dropped
        :param stale_frames: evict cached entries not refreshed for this many frames
        :param refresh_interval: re-submit a tracked face after this many frames
        """
        self._index = faiss_index
        self._queue: queue.Queue[FaissTask] = queue.Queue(maxsize=max_queue_size)
        self._cache: dict[tuple[int, int], MatchResult] = (
            {}
        )  # (source_id, object_id) -> MatchResult
        self._lock = threading.Lock()
        self._stale_frames = stale_frames
        self._refresh_interval = refresh_interval
        self._running = True
        self._worker = threading.Thread(
            target=self._run, daemon=True, name="faiss-worker"
        )
        self._worker.start()
        logger.info(
            "AsyncFaissMatcher started (queue=%d, stale=%d, refresh=%d)",
            max_queue_size,
            stale_frames,
            refresh_interval,
        )

    # ------------------------------------------------------------------
    # Public API (called from streaming thread – must be non-blocking)
    # ------------------------------------------------------------------

    def submit(self, task: FaissTask) -> bool:
        """Non-blocking enqueue.  Returns False when the queue is full."""
        try:
            self._queue.put_nowait(task)
            return True
        except queue.Full:
            return False

    def get_result(self, source_id: int, object_id: int) -> MatchResult | None:
        with self._lock:
            return self._cache.get(_cache_key(source_id, object_id))

    def needs_submit(self, source_id: int, object_id: int, current_frame: int) -> bool:
        """True when (source_id, object_id) has no cached result or the cache is stale."""
        with self._lock:
            cached = self._cache.get(_cache_key(source_id, object_id))
            if cached is None:
                return True
            return (current_frame - cached.frame_number) > self._refresh_interval

    def cleanup(self, current_frame: int) -> int:
        """Remove cache entries older than *stale_frames*.  Returns count removed."""
        with self._lock:
            stale_keys = [
                key
                for key, r in self._cache.items()
                if (current_frame - r.frame_number) > self._stale_frames
            ]
            for key in stale_keys:
                del self._cache[key]
            if stale_keys:
                logger.debug(
                    "AsyncFaissMatcher: cleaned %d stale entries", len(stale_keys)
                )
            return len(stale_keys)

    def stop(self):
        self._running = False
        self._worker.join(timeout=3.0)

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _run(self):
        while self._running:
            try:
                task = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if task.embedding is None:
                continue
            try:
                names, scores = self._index.search_with_scores([task.embedding])
                name = names[0] if names else "Unknown"
                score = scores[0] if scores else 0.0
                result = MatchResult(name, score, task.frame_number)
                with self._lock:
                    self._cache[_cache_key(task.source_id, task.object_id)] = result
            except Exception as e:
                logger.warning("AsyncFaissMatcher search error: %s", e)
