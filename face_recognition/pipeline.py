"""FaceRecognitionPipeline — the main orchestrator.

Assembles the shared inference chain, optional message branch, per-stream
demux branches, and the FAISS probe.  Exposes a **thread-safe** public API
for adding/removing sources and toggling RTSP output at runtime.
"""

from __future__ import annotations

import threading

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib  # type: ignore

from eleproxy import (
    acquire_pipeline,
    acquire_nvurisrcbin,
    get_queue,
    acquire_nvstreamdemux,
)
from common.bus_call import bus_call
from logger.get_logger import get_logger

from .types import SourceConfig, SourceRecord
from .inference_chain import InferenceChain
from .stream_branch import StreamBranch
from .message_branch import MessageBranch
from .probe import FaceProbe

logger = get_logger(__name__)


class FaceRecognitionPipeline:
    """Top-level pipeline that wires every module together.

    All topology mutations (add/remove source, enable/disable RTSP) are
    dispatched to the GLib main-loop thread via ``GLib.idle_add`` so they are
    safe to call from any thread (FastAPI route, MQTT handler, etc.).
    """

    def __init__(self, config: dict):
        self._config = config
        self._pipeline: Gst.Pipeline | None = None
        self._loop: GLib.MainLoop | None = None
        self._loop_thread: threading.Thread | None = None

        self._inference_chain: InferenceChain | None = None
        self._demux: Gst.Element | None = None
        self._probe: FaceProbe | None = None

        self._sources: dict[int, SourceRecord] = {}
        self._next_pad_index = 0
        self._lock = threading.Lock()

        self._save_pipeline_graph: bool = False
        self._target_dir: str = None
        self._graph_name: str = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self):
        """Construct the complete pipeline with initial sources."""
        Gst.init(None)

        self._pipeline = acquire_pipeline()

        self._inference_chain = InferenceChain(
            self._pipeline, self._config.get("inference", {})
        )
        infer_post_tee = self._inference_chain.build()

        msg_cfg = self._config.get("message", None)
        if msg_cfg:
            msg_branch = MessageBranch(self._pipeline, infer_post_tee, msg_cfg)
            msg_branch.build()

        demux_queue = get_queue(index=40)
        self._demux = acquire_nvstreamdemux(args={"index": 0})
        self._pipeline.add(demux_queue)
        self._pipeline.add(self._demux)

        tee_demux_pad = infer_post_tee.request_pad_simple("src_%u")
        if not tee_demux_pad:
            raise RuntimeError("Unable to get infer_post_tee src pad for demux branch")
        sink_pad = demux_queue.get_static_pad("sink")
        ret = tee_demux_pad.link(sink_pad)
        if ret != Gst.PadLinkReturn.OK:
            raise RuntimeError(f"Failed to link infer_post_tee → demux queue: {ret}")
        demux_queue.link(self._demux)

        stream_ids: list[str] = []
        for src_dict in self._config.get("sources", []):
            src_cfg = SourceConfig(
                uri=src_dict["uri"],
                source_id=src_dict.get("source_id", ""),
                latency=src_dict.get("latency", 300),
                rtsp_output=src_dict.get("rtsp_output"),
                mux_slot=src_dict.get("mux_slot"),
            )
            self._add_source_internal(src_cfg)

        # stream_ids[i] = label for source_id i (sink_i)，与 preprocess roi-params-src-N 的 N 对应
        max_pad = max(self._sources.keys()) if self._sources else -1
        stream_ids = [""] * (max_pad + 1)
        for idx, rec in self._sources.items():
            stream_ids[idx] = rec.config.source_id or rec.config.uri

        # n_sources = len(self._sources)
        # HACK nvinfer 的batch-size 不小于 nvdspreprocess 的输出 batch
        n_sources = 4
        if n_sources > 0:
            self._inference_chain.update_batch_size(n_sources)

        # XXX
        self._probe = FaceProbe(
            self._config.get("faiss"),
            stream_ids=stream_ids,
        )
        self._probe.attach(infer_post_tee)

        logger.info(
            "FaceRecognitionPipeline built with %d initial source(s)", n_sources
        )

    # ------------------------------------------------------------------
    # Thread-safe public API
    # ------------------------------------------------------------------

    def add_source(self, source_cfg: dict) -> int:
        """Add a source at runtime.  Returns the assigned ``pad_index``.

        If ``source_cfg`` includes ``mux_slot`` (e.g. 0 or 1), that stream is
        pinned to sink_<mux_slot> so preprocess ``roi-params-src-<mux_slot>``
        always applies; useful when a stream reconnects and should keep the
        same ROI config."""
        result: dict = {}
        done = threading.Event()
        GLib.idle_add(self._do_add_source, source_cfg, result, done)
        done.wait(timeout=10)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result["pad_index"]

    def remove_source(self, pad_index: int):
        """Remove a source at runtime."""
        done = threading.Event()
        GLib.idle_add(self._do_remove_source, pad_index, done)
        done.wait(timeout=10)

    def enable_rtsp(self, pad_index: int):
        """Enable RTSP output for a specific stream."""
        GLib.idle_add(self._do_enable_rtsp, pad_index)

    def disable_rtsp(self, pad_index: int):
        """Disable RTSP output for a specific stream."""
        GLib.idle_add(self._do_disable_rtsp, pad_index)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Set pipeline to PLAYING and run the GLib main loop in a thread."""
        if self._pipeline is None:
            raise RuntimeError("Pipeline not built; call build() first")

        self._loop = GLib.MainLoop()
        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", bus_call, self._loop)

        if self._save_pipeline_graph:
            Gst.debug_bin_to_dot_file(
                self._pipeline, Gst.DebugGraphDetails.ALL, self._graph_name
            )
            logger.info(
                f"Saved pipeline graph to [{self._target_dir}/{self._graph_name}]"
            )

        logger.info("Starting pipeline")
        self._pipeline.set_state(Gst.State.PLAYING)

        self._loop_thread = threading.Thread(
            target=self._loop.run, daemon=True, name="glib-mainloop"
        )
        self._loop_thread.start()

    def stop(self):
        """Stop the pipeline and release resources."""
        if self._probe is not None:
            self._probe.stop()

        if self._pipeline is not None:
            self._pipeline.set_state(Gst.State.NULL)

        if self._loop is not None and self._loop.is_running():
            self._loop.quit()

        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5)
            self._loop_thread = None

        logger.info("Pipeline stopped")

    # ------------------------------------------------------------------
    # Internal helpers (executed on the GLib main-loop thread)
    # ------------------------------------------------------------------

    def _add_source_internal(self, src_cfg: SourceConfig) -> SourceRecord:
        """Add a source synchronously — used both at build time and from
        ``_do_add_source``. If ``src_cfg.mux_slot`` is set, that stream is
        pinned to sink_<mux_slot> (preprocess roi-params-src-<N> N = mux_slot)."""
        if src_cfg.mux_slot is not None:
            pad_index = src_cfg.mux_slot
            if pad_index in self._sources:
                raise RuntimeError(
                    f"mux_slot {pad_index} already in use; remove that source first"
                )
        else:
            pad_index = self._next_pad_index
            self._next_pad_index += 1

        src_bin = acquire_nvurisrcbin(
            index=pad_index,
            args={
                "uri": src_cfg.uri,
                "latency": src_cfg.latency,
            },
        )
        self._pipeline.add(src_bin)

        mux = self._inference_chain.streammux
        mux_sink_pad = mux.request_pad_simple(f"sink_{pad_index}")
        if not mux_sink_pad:
            raise RuntimeError(f"Unable to get mux sink pad sink_{pad_index}")
        src_pad = src_bin.get_static_pad("src")
        if not src_pad:
            raise RuntimeError("Unable to get src_bin src pad")
        ret = src_pad.link(mux_sink_pad)
        if ret != Gst.PadLinkReturn.OK:
            raise RuntimeError(f"Failed to link source → mux: {ret}")

        branch: StreamBranch | None = None
        rtsp_cfg = src_cfg.rtsp_output
        if rtsp_cfg is not None:
            branch = StreamBranch(self._pipeline, self._demux, pad_index, rtsp_cfg)
            branch.build()

        record = SourceRecord(
            pad_index=pad_index,
            config=src_cfg,
            src_bin=src_bin,
            mux_sink_pad=mux_sink_pad,
            branch=branch,
        )
        self._sources[pad_index] = record
        logger.info("Source added: pad_index=%d uri=%s", pad_index, src_cfg.uri)
        return record

    def _do_add_source(self, source_cfg_dict, result, done):
        """GLib.idle_add callback for adding a source at runtime."""
        try:
            src_cfg = SourceConfig(
                uri=source_cfg_dict["uri"],
                source_id=source_cfg_dict.get("source_id", ""),
                latency=source_cfg_dict.get("latency", 300),
                rtsp_output=source_cfg_dict.get("rtsp_output"),
                mux_slot=source_cfg_dict.get("mux_slot"),
            )
            record = self._add_source_internal(src_cfg)

            record.src_bin.sync_state_with_parent()
            if record.branch is not None:
                for el in record.branch._elements:
                    el.sync_state_with_parent()

            # HACK
            # n = len(self._sources)
            n = 4
            self._inference_chain.update_batch_size(n)

            if self._probe is not None:
                max_pad = max(self._sources.keys())
                ids = [""] * (max_pad + 1)
                for idx, r in self._sources.items():
                    ids[idx] = r.config.source_id or r.config.uri
                self._probe.update_stream_ids(ids)

            result["pad_index"] = record.pad_index
        except Exception as e:
            logger.error("Failed to add source: %s", e)
            result["error"] = str(e)
        finally:
            done.set()
        return GLib.SOURCE_REMOVE

    def _do_remove_source(self, pad_index, done):
        """GLib.idle_add callback — schedule removal via IDLE probe."""
        record = self._sources.get(pad_index)
        if record is None:
            logger.warning("remove_source: pad_index %d not found", pad_index)
            done.set()
            return GLib.SOURCE_REMOVE

        if record.branch is not None and record.branch.demux_src_pad is not None:
            record.branch.demux_src_pad.add_probe(
                Gst.PadProbeType.IDLE,
                self._on_idle_teardown,
                (pad_index, done),
            )
        else:
            self._teardown_source(pad_index)
            done.set()
        return GLib.SOURCE_REMOVE

    def _on_idle_teardown(self, pad, info, user_data):
        pad_index, done = user_data
        try:
            self._teardown_source(pad_index)
        finally:
            done.set()
        return Gst.PadProbeReturn.REMOVE

    def _teardown_source(self, pad_index: int):
        record = self._sources.pop(pad_index, None)
        if record is None:
            return

        if record.branch is not None:
            record.branch.teardown()

        src_pad = record.src_bin.get_static_pad("src")
        if src_pad is not None:
            src_pad.unlink(record.mux_sink_pad)
        self._inference_chain.streammux.release_request_pad(record.mux_sink_pad)

        record.src_bin.set_state(Gst.State.NULL)
        self._pipeline.remove(record.src_bin)

        # HACK
        # n = len(self._sources)
        n = 4
        if n > 0:
            self._inference_chain.update_batch_size(n)

        if self._probe is not None:
            ids = [r.config.source_id or r.config.uri for r in self._sources.values()]
            self._probe.update_stream_ids(ids)

        logger.info("Source removed: pad_index=%d", pad_index)

    def _do_enable_rtsp(self, pad_index):
        record = self._sources.get(pad_index)
        if record and record.branch:
            record.branch.enable_rtsp()
        return GLib.SOURCE_REMOVE

    def _do_disable_rtsp(self, pad_index):
        record = self._sources.get(pad_index)
        if record and record.branch:
            record.branch.disable_rtsp()
        return GLib.SOURCE_REMOVE

    # ------------------------------------------------------------------
    # save pipeline graph
    # ------------------------------------------------------------------
    def set_save_pipeline_graph(
        self, target_dir: str = "/tmp", graph_name: str = "graph"
    ):
        self._save_pipeline_graph = True
        self._target_dir = target_dir
        self._graph_name = graph_name
