"""Shared inference chain: mux → queue → preprocess → queue → pgie → sgie → tracker → tee."""

from __future__ import annotations

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # type: ignore

from eleproxy import (
    acquire_nvstreammux,
    acquire_nvdspreprocess,
    acquire_nvinfer,
    acquire_nvtracker,
    get_queue,
)
from logger.get_logger import get_logger

logger = get_logger(__name__)


class InferenceChain:
    """Build and own the shared inference sub-pipeline.

    The chain is: mux → queue → preprocess → queue → pgie → sgie → tracker → tee.
    Callers use :pyattr:`streammux` to request sink pads for each source and
    :pyattr:`tee` to attach downstream branches (message, demux, etc.).
    """

    def __init__(self, pipeline: Gst.Pipeline, config: dict):
        self._pipeline = pipeline
        self._cfg = config
        # inner ele
        self._streammux: Gst.Element | None = None
        self._pgie: Gst.Element | None = None
        self._sgie: Gst.Element | None = None
        self._tee: Gst.Element | None = None

    def build(self) -> Gst.Element:
        """Create all elements, add to *pipeline*, link, and return the tee."""
        cfg = self._cfg
        p = self._pipeline

        self._streammux = acquire_nvstreammux(index=0, args=cfg.get("nvstreammux"))
        q_pre = get_queue(index=10)
        preprocess = acquire_nvdspreprocess(args=cfg.get("nvdspreprocess"))
        q_post = get_queue(index=11)
        self._pgie = acquire_nvinfer(index=0, args=cfg.get("pgie"))
        self._sgie = acquire_nvinfer(index=1, args=cfg.get("sgie"))
        # TODO 多个 SGIE 怎么办
        tracker = acquire_nvtracker(index=0, args=cfg.get("nvtracker"))
        self._tee = Gst.ElementFactory.make("tee", "inference-tee")
        if not self._tee:
            raise RuntimeError("Unable to create tee")

        elements = [
            self._streammux,
            q_pre,
            preprocess,
            q_post,
            self._pgie,
            self._sgie,
            tracker,
            self._tee,
        ]
        for el in elements:
            p.add(el)

        self._streammux.link(q_pre)
        q_pre.link(preprocess)
        preprocess.link(q_post)
        q_post.link(self._pgie)
        self._pgie.link(self._sgie)
        self._sgie.link(tracker)
        tracker.link(self._tee)

        logger.info("InferenceChain built")
        return self._tee

    def update_batch_size(self, n: int):
        """Update batch-size on mux / pgie / sgie at runtime."""
        for el in (self._streammux, self._pgie, self._sgie):
            if el is not None:
                el.set_property("batch-size", n)
        logger.info("batch-size updated to %d", n)

    @property
    def streammux(self) -> Gst.Element:
        assert self._streammux is not None, "build() not called yet"
        return self._streammux

    @property
    def pgie(self) -> Gst.Element:
        assert self._pgie is not None, "build() not called yet"
        return self._pgie

    @property
    def sgie(self) -> Gst.Element:
        assert self._sgie is not None, "build() not called yet"
        return self._sgie

    @property
    def tee(self) -> Gst.Element:
        assert self._tee is not None, "build() not called yet"
        return self._tee
