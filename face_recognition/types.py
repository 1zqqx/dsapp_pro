from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # type: ignore

if TYPE_CHECKING:
    from .stream_branch import StreamBranch


@dataclass
class SourceConfig:
    """Single RTSP/file source configuration."""

    uri: str
    source_id: str
    # TODO ?
    latency: int = 300
    rtsp_output: dict | None = None


@dataclass
class SourceRecord:
    """Runtime bookkeeping for one active source in the pipeline."""

    pad_index: int
    config: SourceConfig
    src_bin: Gst.Bin
    mux_sink_pad: Gst.Pad
    branch: StreamBranch | None = None
