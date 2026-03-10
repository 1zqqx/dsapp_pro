"""FAISS face-matching buffer probe.

Extracted from the original ``rtsp_io.pp.scrfd.faiss.rtsp.py`` pipeline.
The probe is attached to the **tee sink pad** so it processes every frame in
batch granularity across all streams.
"""

from __future__ import annotations


import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # type: ignore

import pyds

from logger.get_logger import get_logger

logger = get_logger(__name__)

PGIE_CLASS_ID_FACE = 0
SGIE_ARCFACE_UNIQUE_ID = 2
ARCFACE_EMBEDDING_DIM = 512


def _format_rect(rect_params):
    """Format rect_params as (left, top, width, height)."""
    if rect_params is None:
        return "N/A"
    return (
        f"({rect_params.left:.0f},{rect_params.top:.0f},"
        f"{rect_params.width:.0f}x{rect_params.height:.0f})"
    )


def _debug_pgie_sink_probe(pad, info, u_data):
    """nvinfer sink pad: 推理前,打印当前 batch 的帧与已有 obj 信息(如来自 preprocess 的 ROI)."""
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if batch_meta is None:
        logger.debug("[PGIE SINK] no batch_meta")
        return Gst.PadProbeReturn.OK

    pyds.nvds_acquire_meta_lock(batch_meta)
    try:
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except (StopIteration, TypeError):
                break

            source_id = frame_meta.source_id
            frame_num = frame_meta.frame_num
            num_obj = frame_meta.num_obj_meta

            batch_id = getattr(frame_meta, "batch_id", -1)
            pad_index = getattr(frame_meta, "pad_index", -1)

            logger.info(
                f"[PGIE SINK] frame={frame_num} src={source_id} num_obj={num_obj} "
                f"pad={pad_index} batch={batch_id} "
                "Sink"
            )

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except (StopIteration, TypeError):
                    break
                rect_str = _format_rect(getattr(obj_meta, "rect_params", None))
                conf = getattr(obj_meta, "confidence", 0.0)
                logger.info(
                    f"frame_src={source_id} frame_number={frame_num} "
                    f"obj id={obj_meta.object_id} class={obj_meta.class_id} "
                    f"rect={rect_str} conf={conf:.3f} "
                    "Sink"
                )
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
    except Exception as e:
        logger.info(f"[PGIE SINK] e: {e}")
    finally:
        pyds.nvds_release_meta_lock(batch_meta)
    return Gst.PadProbeReturn.OK


def _debug_pgie_src_probe(pad, info, u_data):
    """nvinfer src pad: 推理后,打印检测到的目标(bbox、class、confidence)."""
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if batch_meta is None:
        logger.debug("[PGIE SRC] no batch_meta")
        return Gst.PadProbeReturn.OK

    pyds.nvds_acquire_meta_lock(batch_meta)
    try:
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except (StopIteration, TypeError):
                break

            batch_id = getattr(frame_meta, "batch_id", -1)
            pad_index = getattr(frame_meta, "pad_index", -1)

            source_id = frame_meta.source_id
            frame_num = frame_meta.frame_num
            num_obj = frame_meta.num_obj_meta
            logger.info(
                f"[PGIE SRC] frame={frame_num} src={source_id} num_obj={num_obj} "
                f"pad={pad_index} batch={batch_id} "
                "After"
            )

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except (StopIteration, TypeError):
                    break
                rect_str = _format_rect(getattr(obj_meta, "rect_params", None))
                conf = getattr(obj_meta, "confidence", 0.0)
                logger.info(
                    f"frame_src={source_id} frame_number={frame_num} "
                    f"obj id={obj_meta.object_id} class={obj_meta.class_id} "
                    f"rect={rect_str} conf={conf:.3f} "
                    "Arter"
                )
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
    except Exception as e:
        logger.info(f"[PGIE SRC] e: {e}")
    finally:
        pyds.nvds_release_meta_lock(batch_meta)
    return Gst.PadProbeReturn.OK


class DeProbe:

    def __init__(self):
        pass

    def attach(self, ele: Gst.Element):
        """Build the FAISS index and attach the probe to *tee*'s sink pad."""

        sink_pad = ele.get_static_pad("sink")
        src_pad = ele.get_static_pad("src")
        if not sink_pad or not src_pad:
            raise RuntimeError("Unable to get tee sink pad for probe")

        sink_pad.add_probe(Gst.PadProbeType.BUFFER, _debug_pgie_sink_probe, None)
        src_pad.add_probe(Gst.PadProbeType.BUFFER, _debug_pgie_src_probe, None)
        logger.info(f" DeProbe attached ")

    def stop(self):
        pass
