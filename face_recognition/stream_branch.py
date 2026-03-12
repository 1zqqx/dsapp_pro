"""Per-stream branch: demux src_N → valve → queue → nvvidconv → nvosd → rtsp_client_bin."""

from __future__ import annotations

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # type: ignore

from eleproxy import (
    acquire_nvvideoconvert,
    acquire_nvdsosd,
    get_queue,
    get_rtsp_client_bin,
)
from logger.get_logger import get_logger

logger = get_logger(__name__)


class StreamBranch:
    """Owns the sub-pipeline from *nvstreamdemux.src_N* to the RTSP output.

    ``valve`` gates the data flow: ``drop=True`` disables RTSP output with
    near-zero overhead; ``drop=False`` re-enables it.
    """

    def __init__(
        self,
        pipeline: Gst.Pipeline,
        demux: Gst.Element,
        pad_index: int,
        rtsp_cfg: dict | None = None,
    ):
        self._pipeline = pipeline
        self._demux = demux
        self._pad_index = pad_index
        self._rtsp_cfg = rtsp_cfg or {}

        self._queue: Gst.Element | None = None
        self._nvvidconv: Gst.Element | None = None
        self._nvosd: Gst.Element | None = None
        self._valve: Gst.Element | None = None
        self._rtsp_bin: Gst.Bin | None = None
        self._demux_src_pad: Gst.Pad | None = None
        self._elements: list[Gst.Element] = []

    def build(self):
        """Create elements, add to pipeline, link, and connect to demux."""
        idx = self._pad_index
        p = self._pipeline

        self._queue = get_queue(index=200 + idx)
        self._nvvidconv = acquire_nvvideoconvert(index=100 + idx)
        self._nvosd = acquire_nvdsosd(index=100 + idx)

        self._valve = Gst.ElementFactory.make("valve", f"valve-{idx:03d}")
        if not self._valve:
            raise RuntimeError(f"Unable to create valve-{idx:03d}")

        rtsp_enabled = self._rtsp_cfg.get("enabled", False)
        # valve: drop=False, the data flow is passing normally; drop=True, the data flow is blocked.
        self._valve.set_property("drop", not rtsp_enabled)

        self._rtsp_bin = get_rtsp_client_bin(args=self._rtsp_cfg)
        self._rtsp_bin.set_property("name", f"rtsp-client-bin-{idx:03d}")

        self._elements = [
            self._valve,
            self._queue,
            self._nvvidconv,
            self._nvosd,
            self._rtsp_bin,
        ]
        for el in self._elements:
            p.add(el)

        self._valve.link(self._queue)
        self._queue.link(self._nvvidconv)
        self._nvvidconv.link(self._nvosd)
        self._nvosd.link(self._rtsp_bin)

        pad_name = f"src_{idx}"
        self._demux_src_pad = self._demux.request_pad_simple(pad_name)
        if not self._demux_src_pad:
            raise RuntimeError(f"Unable to get demux pad {pad_name}")
        valve_sink = self._valve.get_static_pad("sink")
        ret = self._demux_src_pad.link(valve_sink)
        if ret != Gst.PadLinkReturn.OK:
            raise RuntimeError(f"Failed to link demux {pad_name} → valve sink: {ret}")

        logger.info(f"StreamBranch {idx} built (rtsp_enabled={rtsp_enabled})")

    def get_queue_sink_pad(self) -> Gst.Pad | None:
        """Return the queue sink pad (demux 出口下游). 用于 DEBUG 步骤2 挂只读 probe。"""
        if self._queue is None:
            return None
        return self._queue.get_static_pad("sink")

    def enable_rtsp(self):
        if self._valve:
            self._valve.set_property("drop", False)
            logger.info(f"StreamBranch {self._pad_index}: RTSP enabled")

    def disable_rtsp(self):
        if self._valve:
            self._valve.set_property("drop", True)
            logger.info(f"StreamBranch {self._pad_index}: RTSP disabled")

    @property
    def is_rtsp_enabled(self) -> bool:
        if self._valve is None:
            return False
        return not self._valve.get_property("drop")

    @property
    def demux_src_pad(self) -> Gst.Pad | None:
        return self._demux_src_pad

    @property
    def nvosd(self) -> Gst.Element | None:
        return self._nvosd

    def teardown(self):
        """Unlink and remove all elements.  Must be called from an IDLE probe
        on the demux src pad (i.e. the pad is guaranteed idle)."""
        for el in reversed(self._elements):
            el.set_state(Gst.State.NULL)
            self._pipeline.remove(el)

        if self._demux_src_pad is not None:
            self._demux.release_request_pad(self._demux_src_pad)
            self._demux_src_pad = None

        self._elements.clear()
        logger.info("StreamBranch %d torn down", self._pad_index)
