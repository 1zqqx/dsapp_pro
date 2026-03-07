"""FAISS face-matching buffer probe.

Extracted from the original ``rtsp_io.pp.scrfd.faiss.rtsp.py`` pipeline.
The probe is attached to the **tee sink pad** so it processes every frame in
batch granularity across all streams.
"""

from __future__ import annotations

import json
import time
import ctypes

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # type: ignore

import pyds
import numpy as np

from ifaiss import build_faiss_index, AsyncFaissMatcher, FaissTask
from logger.get_logger import get_logger

logger = get_logger(__name__)

PGIE_CLASS_ID_FACE = 0
SGIE_ARCFACE_UNIQUE_ID = 2
ARCFACE_EMBEDDING_DIM = 512


def get_arcface_embedding_from_obj(
    obj_meta, unique_id_filter: int | None = None
) -> np.ndarray | None:
    """Extract the 512-d ArcFace embedding from *obj_meta* tensor user-meta.

    Returns ``None`` when no suitable tensor layer is found.
    """
    l_user = obj_meta.obj_user_meta_list
    while l_user is not None:
        try:
            user_meta = pyds.NvDsUserMeta.cast(l_user.data)
        except StopIteration:
            break

        if (
            user_meta.base_meta.meta_type
            != pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META
        ):
            try:
                l_user = l_user.next
            except StopIteration:
                break
            continue

        try:
            tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
        except StopIteration:
            break

        if unique_id_filter is not None and tensor_meta.unique_id != unique_id_filter:
            try:
                l_user = l_user.next
            except StopIteration:
                break
            continue

        if tensor_meta.num_output_layers < 1:
            try:
                l_user = l_user.next
            except StopIteration:
                break
            continue

        try:
            layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
            ptr = ctypes.cast(
                pyds.get_ptr(layer.buffer),
                ctypes.POINTER(ctypes.c_float),
            )
            arr = np.ctypeslib.as_array(ptr, shape=(ARCFACE_EMBEDDING_DIM,))
            return np.array(arr, dtype=np.float32, copy=True)
        except Exception as e:
            logger.debug("get_arcface_embedding_from_obj: %s", e)
            return None

        try:
            l_user = l_user.next
        except StopIteration:
            break
    return None


# DEBUG_NO_BBOX 步骤2: 只读 probe 每 N 个 buffer 打一次 log，避免刷屏
_DEBUG_STEP2_LOG_INTERVAL = 30


def _demux_sink_probe_readonly(pad, info, u_data):
    """DEBUG 步骤2: nvstreamdemux sink 只读 probe，确认进入 demux 的 buffer 是否有 batch_meta 和 obj_meta。"""
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if batch_meta is None:
        u_data["counter"] = u_data.get("counter", 0) + 1
        if u_data["counter"] % _DEBUG_STEP2_LOG_INTERVAL == 1:
            logger.warning("[DEBUG step2] demux sink: batch_meta is None")
        return Gst.PadProbeReturn.OK

    u_data["counter"] = u_data.get("counter", 0) + 1
    if u_data["counter"] % _DEBUG_STEP2_LOG_INTERVAL != 1:
        return Gst.PadProbeReturn.OK

    n_frames = 0
    per_frame_objs = []
    pyds.nvds_acquire_meta_lock(batch_meta)
    try:
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break
            n_frames += 1
            n_obj = 0
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    pyds.NvDsObjectMeta.cast(l_obj.data)
                    n_obj += 1
                except StopIteration:
                    break
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
            per_frame_objs.append(n_obj)
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
    finally:
        pyds.nvds_release_meta_lock(batch_meta)
    logger.info(
        "[DEBUG step2] demux sink: n_frames=%d per_frame_objs=%s",
        n_frames,
        per_frame_objs,
    )
    return Gst.PadProbeReturn.OK


def _branch_queue_sink_probe_readonly(pad, info, u_data):
    """DEBUG 步骤2: StreamBranch queue sink 只读 probe，确认 demux 出口 buffer 是否仍有 batch_meta/obj_meta。"""
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if batch_meta is None:
        u_data["counter"] = u_data.get("counter", 0) + 1
        if u_data["counter"] % _DEBUG_STEP2_LOG_INTERVAL == 1:
            logger.warning(
                "[DEBUG step2] branch%d queue sink: batch_meta is None",
                u_data.get("branch_idx", -1),
            )
        return Gst.PadProbeReturn.OK

    u_data["counter"] = u_data.get("counter", 0) + 1
    if u_data["counter"] % _DEBUG_STEP2_LOG_INTERVAL != 1:
        return Gst.PadProbeReturn.OK

    n_frames = 0
    per_frame_objs = []
    pyds.nvds_acquire_meta_lock(batch_meta)
    try:
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break
            n_frames += 1
            n_obj = 0
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    pyds.NvDsObjectMeta.cast(l_obj.data)
                    n_obj += 1
                except StopIteration:
                    break
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
            per_frame_objs.append(n_obj)
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
    finally:
        pyds.nvds_release_meta_lock(batch_meta)
    logger.info(
        "[DEBUG step2] branch%d queue sink: n_frames=%d per_frame_objs=%s",
        u_data.get("branch_idx", -1),
        n_frames,
        per_frame_objs,
    )
    return Gst.PadProbeReturn.OK


def _infer_src_probe_readonly(pad, info, u_data):
    """DEBUG 步骤2 延伸: pgie/sgie src 只读 probe，确认 nvinfer 是否输出 obj_meta。"""
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    name = u_data.get("name", "infer")
    if batch_meta is None:
        u_data["counter"] = u_data.get("counter", 0) + 1
        if u_data["counter"] % _DEBUG_STEP2_LOG_INTERVAL == 1:
            logger.warning("[DEBUG step2] %s src: batch_meta is None", name)
        return Gst.PadProbeReturn.OK

    u_data["counter"] = u_data.get("counter", 0) + 1
    if u_data["counter"] % _DEBUG_STEP2_LOG_INTERVAL != 1:
        return Gst.PadProbeReturn.OK

    n_frames = 0
    per_frame_objs = []
    pyds.nvds_acquire_meta_lock(batch_meta)
    try:
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break
            n_frames += 1
            n_obj = 0
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    pyds.NvDsObjectMeta.cast(l_obj.data)
                    n_obj += 1
                except StopIteration:
                    break
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
            per_frame_objs.append(n_obj)
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
    finally:
        pyds.nvds_release_meta_lock(batch_meta)
    logger.info(
        "[DEBUG step2] %s src: n_frames=%d per_frame_objs=%s",
        name,
        n_frames,
        per_frame_objs,
    )
    return Gst.PadProbeReturn.OK


def _pgie_sink_probe_readonly(pad, info, u_data):
    """DEBUG: PGIE sink 只读 probe，确认进入 PGIE 的输入是否有 batch_meta、帧数与尺寸。"""
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if batch_meta is None:
        u_data["counter"] = u_data.get("counter", 0) + 1
        if u_data["counter"] % _DEBUG_STEP2_LOG_INTERVAL == 1:
            logger.warning("[DEBUG step2] pgie sink: batch_meta is None")
        return Gst.PadProbeReturn.OK

    u_data["counter"] = u_data.get("counter", 0) + 1
    if u_data["counter"] % _DEBUG_STEP2_LOG_INTERVAL != 1:
        return Gst.PadProbeReturn.OK

    n_frames = 0
    frame_info = []
    pyds.nvds_acquire_meta_lock(batch_meta)
    try:
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break
            n_frames += 1
            frame_info.append(
                (frame_meta.source_frame_width, frame_meta.source_frame_height)
            )
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
    finally:
        pyds.nvds_release_meta_lock(batch_meta)
    logger.info(
        "[DEBUG step2] pgie sink: n_frames=%d frame_wh=%s",
        n_frames,
        frame_info,
    )
    return Gst.PadProbeReturn.OK


def _batch_buffer_probe(pad, info, u_data):
    """Tee sink-pad buffer probe — async FAISS face matching.

    Runs on the GStreamer streaming thread.  Only lightweight, non-blocking
    work is performed here:

    1. Iterate face ``obj_meta`` per frame; look up the tracker *object_id* in
       the async matcher cache.
    2. Cache hit → set OSD display text immediately.
    3. Cache miss / stale → extract embedding and submit to the background
       FAISS worker (non-blocking).
    4. Attach a ``NVDS_CUSTOM_MSG_BLOB`` JSON payload per frame for
       downstream ``nvmsgconv``.
    5. Periodically clean up stale cache entries for departed faces.
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        logger.warning("Unable to get GstBuffer")
        return Gst.PadProbeReturn.OK

    # async_matcher: AsyncFaissMatcher | None = u_data.get("async_matcher")
    # stream_ids: list[str] = u_data.get("stream_ids", [])

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if batch_meta is None:
        logger.warning("Unable to get batch meta from GstBuffer")
        return Gst.PadProbeReturn.OK

    async_matcher: AsyncFaissMatcher | None = u_data.get("async_matcher")
    stream_ids: list[str] = u_data.get("stream_ids", [])

    def _stream_id_for_source(sid: int) -> str:
        if sid < len(stream_ids) and stream_ids[sid]:
            return str(stream_ids[sid])
        return f"source_{sid}"

    pyds.nvds_acquire_meta_lock(batch_meta)
    try:
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            source_id = frame_meta.source_id
            frame_number = frame_meta.frame_num
            face_results: list[dict] = []

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                if obj_meta.class_id != PGIE_CLASS_ID_FACE:
                    try:
                        l_obj = l_obj.next
                    except StopIteration:
                        break
                    continue

                object_id = obj_meta.object_id
                cached = None

                if async_matcher is not None:
                    cached = async_matcher.get_result(object_id)
                    if cached is not None:
                        txt = obj_meta.text_params
                        txt.display_text = f"{cached.name}-{cached.score:.2f}"
                        txt.font_params.font_name = "Serif"
                        txt.font_params.font_size = 14
                        txt.font_params.font_color.set(0.0, 1.0, 0.0, 1.0)
                        txt.set_bg_clr = 0

                    if async_matcher.needs_submit(object_id, frame_number):
                        emb = get_arcface_embedding_from_obj(
                            obj_meta, SGIE_ARCFACE_UNIQUE_ID
                        )
                        if emb is not None:
                            async_matcher.submit(
                                FaissTask(
                                    object_id=object_id,
                                    source_id=source_id,
                                    frame_number=frame_number,
                                    embedding=emb,
                                )
                            )

                face_results.append(
                    {
                        "object_id": int(object_id),
                        "name": cached.name if cached else "Unknown",
                        "score": round(cached.score, 3) if cached else 0.0,
                    }
                )

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            if face_results:
                payload = {
                    "source_id": source_id,
                    "source_uri": _stream_id_for_source(source_id),
                    "frame_num": frame_number,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
                    "num_faces": len(face_results),
                    "faces": face_results,
                }
                custom_json = json.dumps(payload, ensure_ascii=False)
                pyds.nvds_add_custom_msg_blob_to_frame(
                    frame_meta, batch_meta, custom_json
                )

            if async_matcher is not None and frame_number % 300 == 0:
                async_matcher.cleanup(frame_number)

            if frame_number % 90 == 0:
                logger.debug(
                    f"frame={frame_number} src={_stream_id_for_source(source_id)} num_obj={len(face_results)}",
                )

            try:
                l_frame = l_frame.next
            except StopIteration:
                break
    finally:
        pyds.nvds_release_meta_lock(batch_meta)

    return Gst.PadProbeReturn.OK


class FaceProbe:
    """Manages the FAISS index, the :class:`AsyncFaissMatcher`, and the tee
    sink-pad buffer probe lifecycle."""

    def __init__(self, faiss_cfg: dict | None, stream_ids: list[str] | None = None):
        self._faiss_cfg = faiss_cfg
        self._stream_ids = stream_ids or []
        self._async_matcher: AsyncFaissMatcher | None = None

    def attach(self, tee: Gst.Element):
        """Build the FAISS index and attach the probe to *tee*'s sink pad."""
        faiss_index = build_faiss_index(self._faiss_cfg)
        if faiss_index is not None:
            self._async_matcher = AsyncFaissMatcher(
                faiss_index,
                max_queue_size=8,
                stale_frames=300,
                refresh_interval=90,
            )

        sink_pad = tee.get_static_pad("sink")
        if not sink_pad:
            raise RuntimeError("Unable to get tee sink pad for probe")
        probe_user_data = {
            "async_matcher": self._async_matcher,
            "stream_ids": self._stream_ids,
        }
        sink_pad.add_probe(
            Gst.PadProbeType.BUFFER, _batch_buffer_probe, probe_user_data
        )
        logger.info("FaceProbe attached (matcher=%s)", self._async_matcher is not None)

    def update_stream_ids(self, stream_ids: list[str]):
        self._stream_ids[:] = stream_ids

    def stop(self):
        if self._async_matcher is not None:
            self._async_matcher.stop()
            self._async_matcher = None
