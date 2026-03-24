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
    # batch_meta → frame_meta_list → frame_meta.obj_meta_list → obj_meta
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
                    cached = async_matcher.get_result(source_id, object_id)
                    txt = obj_meta.text_params
                    if cached is not None:
                        txt.display_text = f"{cached.name}-{cached.score:.2f}"
                    else:
                        txt.display_text = "Unknown"
                    txt.font_params.font_name = "Serif"
                    txt.font_params.font_size = 14
                    txt.font_params.font_color.set(0.0, 1.0, 0.0, 1.0)
                    txt.set_bg_clr = 0

                    if async_matcher.needs_submit(source_id, object_id, frame_number):
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
                # end of l_obj

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
            # end of l_frame

        l_user_meta = batch_meta.batch_user_meta_list
        while l_user_meta is not None:
            try:
                user_meta = pyds.NvDsUserMeta.cast(l_user_meta.data)
            except StopIteration:
                break
            if user_meta.base_meta.meta_type == pyds.NVDS_PREPROCESS_BATCH_META:
                # preprocess 的元数据
                try:
                    preprocess_batchmeta = pyds.GstNvDsPreProcessBatchMeta.cast(
                        user_meta.user_meta_data
                    )
                except StopIteration:
                    break
                roi_cnt = 0

                for roi_meta in preprocess_batchmeta.roi_vector:
                    # Label ROI in display
                    display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                    display_meta.num_labels = 1

                    txt_params = display_meta.text_params[0]
                    txt_params.display_text = f"Roi:{roi_cnt}"

                    txt_params.x_offset = int(roi_meta.roi.left)
                    txt_params.y_offset = int(roi_meta.roi.top)

                    txt_params.font_params.font_name = "Serif"
                    txt_params.font_params.font_size = 14
                    txt_params.font_params.font_color.set(0, 1.0, 0, 1.0)  # RGBA
                    txt_params.set_bg_clr = 0
                    # txt_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.5)
                    pyds.nvds_add_display_meta_to_frame(
                        roi_meta.frame_meta, display_meta
                    )
                    # logger.debug(
                    #     f"frame {roi_meta.frame_meta.frame_num} src {roi_meta.frame_meta.source_id} roi {roi_cnt}"
                    # )
                    roi_cnt += 1
            try:
                l_user_meta = l_user_meta.next
            except StopIteration:
                break
            # end of batch_user_meta_list
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
        logger.info(f"FaceProbe attached (matcher={self._async_matcher is not None})")

    def update_stream_ids(self, stream_ids: list[str]):
        self._stream_ids[:] = stream_ids

    def stop(self):
        if self._async_matcher is not None:
            self._async_matcher.stop()
            self._async_matcher = None
