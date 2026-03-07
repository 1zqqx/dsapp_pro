"""Optional message branch: tee → queue → nvmsgconv → nvmsgbroker."""

from __future__ import annotations

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # type: ignore

from eleproxy import acquire_nvmsgconv, acquire_nvmsgbroker, get_queue
from logger.get_logger import get_logger

logger = get_logger(__name__)


class MessageBranch:
    """Attach a ``tee → queue → msgconv → msgbroker`` branch to the inference
    chain tee.  Instantiate only when the *message* config section is present.
    """

    def __init__(self, pipeline: Gst.Pipeline, tee: Gst.Element, config: dict):
        self._pipeline = pipeline
        self._tee = tee
        self._cfg = config

    def build(self):
        """Create elements, add to pipeline, and link to a new tee src pad."""
        p = self._pipeline
        cfg = self._cfg

        queue = get_queue(index=30)
        queue.set_property("max-size-buffers", 60)
        queue.set_property("max-size-time", 2 * Gst.SECOND)

        msgconv = acquire_nvmsgconv(index=0, args=cfg.get("nvmsgconv"))
        msgbroker = acquire_nvmsgbroker(index=0, args=cfg.get("nvmsgbroker"))

        for el in (queue, msgconv, msgbroker):
            p.add(el)

        tee_pad = self._tee.request_pad_simple("src_%u")
        if not tee_pad:
            raise RuntimeError("Unable to get tee src pad for message branch")
        sink_pad = queue.get_static_pad("sink")
        ret = tee_pad.link(sink_pad)
        if ret != Gst.PadLinkReturn.OK:
            raise RuntimeError(f"Failed to link tee → message queue: {ret}")

        queue.link(msgconv)
        msgconv.link(msgbroker)

        logger.info("MessageBranch built")
