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
    # 固定 mux 槽位：指定时该路始终使用 sink_<mux_slot>，与 preprocess roi-params-src-<N> 的 N 一致；
    # 断线重连后仍用同一槽位，preprocess 配置无需随添加顺序变化。
    mux_slot: int | None = None


@dataclass
class SourceRecord:
    """Runtime bookkeeping for one active source in the pipeline."""

    pad_index: int
    config: SourceConfig
    src_bin: Gst.Bin
    mux_sink_pad: Gst.Pad
    branch: StreamBranch | None = None
