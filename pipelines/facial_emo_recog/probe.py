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
# 默认与 nvconfigs 中 gie-unique-id 一致; 可在 faiss 配置里覆盖
SGIE_ARCFACE_DEFAULT_UNIQUE_ID = 2
SGIE_EMOTION_DEFAULT_UNIQUE_ID = 3
ARCFACE_EMBEDDING_DIM = 512

NUM_EMOTION_CLASSES = 8
# 与 dsapp_emotion_labels.txt / NvDsInferParseCustomEmotion 顺序一致
EMOTION_LABELS = (
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
)

# nvdsinfer_db.h: NvDsInferDataType FLOAT=0, HALF=1(FP16 engine 输出多为 HALF)
_NVDSINFER_FLOAT = 0
_NVDSINFER_HALF = 1


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

        if unique_id_filter is not None and int(tensor_meta.unique_id) != int(
            unique_id_filter
        ):
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
            buf_ptr = pyds.get_ptr(layer.buffer)
            if not buf_ptr:
                try:
                    l_user = l_user.next
                except StopIteration:
                    break
                continue
            dtype = _NVDSINFER_FLOAT
            try:
                dtype = int(layer.dataType)
            except Exception:
                pass
            if dtype == _NVDSINFER_HALF:
                hptr = ctypes.cast(buf_ptr, ctypes.POINTER(ctypes.c_uint16))
                u16 = np.ctypeslib.as_array(hptr, shape=(ARCFACE_EMBEDDING_DIM,))
                arr = u16.view(np.float16).astype(np.float32, copy=True)
            else:
                fptr = ctypes.cast(buf_ptr, ctypes.POINTER(ctypes.c_float))
                arr = np.ctypeslib.as_array(fptr, shape=(ARCFACE_EMBEDDING_DIM,))
                arr = np.array(arr, dtype=np.float32, copy=True)
            return arr
        except Exception as e:
            logger.debug("get_arcface_embedding_from_obj skip one tensor: %s", e)
            try:
                l_user = l_user.next
            except StopIteration:
                break
            continue

    return None


def _softmax_8(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64).reshape(-1)[:NUM_EMOTION_CLASSES]
    if x.size < NUM_EMOTION_CLASSES:
        return np.zeros(NUM_EMOTION_CLASSES, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    s = np.sum(e)
    if s <= 0:
        return np.ones(NUM_EMOTION_CLASSES, dtype=np.float64) / NUM_EMOTION_CLASSES
    return e / s


def _read_emotion_logits8(layer) -> np.ndarray | None:
    """只读输出层前 8 个标量;按 dataType 区分 FLOAT / HALF,避免 FP16 engine 按 float 误读导致异常."""
    try:
        buf_ptr = pyds.get_ptr(layer.buffer)
        if not buf_ptr:
            return None
        dtype = _NVDSINFER_FLOAT
        try:
            dtype = int(layer.dataType)
        except Exception:
            pass
        if dtype == _NVDSINFER_HALF:
            hptr = ctypes.cast(buf_ptr, ctypes.POINTER(ctypes.c_uint16))
            u16 = np.ctypeslib.as_array(hptr, shape=(NUM_EMOTION_CLASSES,))
            return u16.view(np.float16).astype(np.float32, copy=True)
        fptr = ctypes.cast(buf_ptr, ctypes.POINTER(ctypes.c_float))
        arr = np.ctypeslib.as_array(fptr, shape=(NUM_EMOTION_CLASSES,))
        return np.array(arr, dtype=np.float32, copy=True)
    except Exception as e:
        logger.error("_read_emotion_logits8: %s", e)
        return None


def _emotion_from_classifier_meta(
    obj_meta, unique_id_filter: int | None
) -> tuple[str, float] | None:
    """tensor 不可用时,从 nvinfer 写入的 NvDsClassifierMeta 取情绪(需解析器已过阈值写入)."""
    if unique_id_filter is None:
        return None
    uid = int(unique_id_filter)
    l_clf = obj_meta.classifier_meta_list
    while l_clf is not None:
        try:
            cm = pyds.NvDsClassifierMeta.cast(l_clf.data)
        except StopIteration:
            break
        if int(cm.unique_component_id) != uid:
            try:
                l_clf = l_clf.next
            except StopIteration:
                break
            continue
        if int(cm.num_labels) < 1:
            try:
                l_clf = l_clf.next
            except StopIteration:
                break
            continue
        try:
            li0 = pyds.NvDsLabelInfo.cast(cm.label_info_list.data)
        except Exception:
            try:
                l_clf = l_clf.next
            except StopIteration:
                break
            continue
        try:
            prob = float(li0.result_prob)
        except Exception:
            prob = 0.0
        label = None
        try:
            if getattr(li0, "pResult_label", None):
                label = str(li0.pResult_label)
            elif li0.result_label:
                label = pyds.get_string(li0.result_label)
        except Exception:
            label = None
        if not label:
            cid = int(getattr(li0, "result_class_id", -1))
            if 0 <= cid < len(EMOTION_LABELS):
                label = EMOTION_LABELS[cid]
            else:
                label = "unknown"
        return (label, prob)

    return None


def get_emotion_ferplus_from_obj(
    obj_meta, unique_id_filter: int | None = None
) -> tuple[str, float] | None:
    """FER+ 情绪:优先 tensor output meta(支持 FP16);失败则读 classifier_meta."""
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

        if unique_id_filter is not None and int(tensor_meta.unique_id) != int(
            unique_id_filter
        ):
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
            logits = _read_emotion_logits8(layer)
            if logits is None or logits.size < NUM_EMOTION_CLASSES:
                try:
                    l_user = l_user.next
                except StopIteration:
                    break
                continue
            probs = _softmax_8(logits)
            best = int(np.argmax(probs))
            conf = float(probs[best])
            label = EMOTION_LABELS[best] if 0 <= best < len(EMOTION_LABELS) else "ub"
            return (label, conf)
        except Exception as e:
            logger.error("get_emotion_ferplus_from_obj tensor pass: %s", e)
            try:
                l_user = l_user.next
            except StopIteration:
                break
            continue

    return _emotion_from_classifier_meta(obj_meta, unique_id_filter)


def _format_face_label(username: str, uconf: float, emotion: str, econf: float) -> str:
    """OSD 与 payload['label'] : {username:confidence}-{emotion:confidence}"""
    return f"{username}:{uconf:.2f}-{emotion}:{econf:.2f}"


def _arcface_src_probe(pad, info, u_data):
    """Extract ArcFace embeddings from tensor_meta BEFORE the next SGIE clears them.

    DeepStream nvinfer clears obj_user_meta_list when a downstream SGIE
    processes the same object, so tensor_meta must be captured here — on the
    arcface SGIE's src pad — before it reaches the emotion SGIE.
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if batch_meta is None:
        return Gst.PadProbeReturn.OK

    async_matcher: AsyncFaissMatcher | None = u_data.get("async_matcher")
    arcface_uid: int = int(
        u_data.get("arcface_gie_unique_id", SGIE_ARCFACE_DEFAULT_UNIQUE_ID)
    )
    if async_matcher is None:
        return Gst.PadProbeReturn.OK

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
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break
                if obj_meta.class_id == PGIE_CLASS_ID_FACE:
                    object_id = obj_meta.object_id
                    if async_matcher.needs_submit(source_id, object_id, frame_number):
                        emb = get_arcface_embedding_from_obj(obj_meta, arcface_uid)
                        if emb is not None:
                            async_matcher.submit(
                                FaissTask(
                                    object_id=object_id,
                                    source_id=source_id,
                                    frame_number=frame_number,
                                    embedding=emb,
                                )
                            )
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
    finally:
        pyds.nvds_release_meta_lock(batch_meta)

    return Gst.PadProbeReturn.OK


def _batch_buffer_probe(pad, info, u_data):
    """Tee sink-pad buffer probe:FAISS 身份 + FER+ 情绪,写 OSD 与 custom msg JSON.

    情绪优先从 ``output-tensor-meta`` 解析(含 FP16);失败时读 ``classifier_meta``.
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
    emotion_uid: int = int(
        u_data.get("emotion_gie_unique_id", SGIE_EMOTION_DEFAULT_UNIQUE_ID)
    )

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

                emo = get_emotion_ferplus_from_obj(obj_meta, emotion_uid)
                if emo is not None:
                    emotion_label, emotion_conf = emo
                else:
                    emotion_label, emotion_conf = "unknown", 0.0

                if cached is not None:
                    username = cached.name
                    username_conf = float(cached.score)
                else:
                    username = "unknown"
                    username_conf = 0.0

                txt = obj_meta.text_params
                txt.display_text = _format_face_label(
                    username, username_conf, emotion_label, emotion_conf
                )
                txt.font_params.font_name = "Serif"
                txt.font_params.font_size = 14
                txt.font_params.font_color.set(0.0, 1.0, 0.0, 1.0)
                txt.set_bg_clr = 0

                # ArcFace embedding extraction + FAISS submit now happens in
                # _arcface_src_probe (on the arcface SGIE src pad), because the
                # downstream emotion SGIE clears obj_user_meta_list (tensor_meta).

                face_results.append(
                    {
                        "object_id": int(object_id),
                        "username": username,
                        "username_confidence": round(username_conf, 3),
                        "emotion_label": emotion_label,
                        "emotion_confidence": round(emotion_conf, 3),
                        "label": _format_face_label(
                            username,
                            username_conf,
                            emotion_label,
                            emotion_conf,
                        ),
                        "name": username,
                        "score": round(username_conf, 3),
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

            if frame_number % 100 == 0:
                logger.debug(
                    f"frame={frame_number} src={_stream_id_for_source(source_id)} num_obj={len(face_results)}",
                )

            try:
                l_frame = l_frame.next
            except StopIteration:
                break
            # end of l_frame
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
        fc = faiss_cfg or {}
        self._arcface_gie_unique_id = int(
            fc.get("arcface_gie_unique_id", SGIE_ARCFACE_DEFAULT_UNIQUE_ID)
        )
        self._emotion_gie_unique_id = int(
            fc.get("emotion_gie_unique_id", SGIE_EMOTION_DEFAULT_UNIQUE_ID)
        )

    def attach(self, tee: Gst.Element, sgie_elements: list[Gst.Element] | None = None):
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
            "emotion_gie_unique_id": self._emotion_gie_unique_id,
        }
        sink_pad.add_probe(
            Gst.PadProbeType.BUFFER, _batch_buffer_probe, probe_user_data
        )
        if sgie_elements:
            _arcface_src = sgie_elements[0].get_static_pad("src")
            if _arcface_src:
                _arcface_src.add_probe(
                    Gst.PadProbeType.BUFFER,
                    _arcface_src_probe,
                    {
                        "async_matcher": self._async_matcher,
                        "arcface_gie_unique_id": self._arcface_gie_unique_id,
                    },
                )
                logger.info("arcface src-pad probe attached for embedding extraction")
        logger.info(f"FaceProbe attached (matcher={self._async_matcher is not None})")

    def update_stream_ids(self, stream_ids: list[str]):
        self._stream_ids[:] = stream_ids

    def stop(self):
        if self._async_matcher is not None:
            self._async_matcher.stop()
            self._async_matcher = None
