import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import time
import logging
import ctypes

import gi

gi.require_version("Gst", "1.0")

from gi.repository import Gst, GLib  # type: ignore

import pyds

# 若当前安装的 pyds 未包含 CT 自定义 binding，给出明确提示
if not hasattr(pyds, "alloc_ct_face_obj_struct"):
    _pyds_file = getattr(pyds, "__file__", "unknown")
    raise ImportError(
        "当前安装的 pyds 未包含 alloc_ct_face_obj_struct（CT 自定义 binding）。"
        "请在本仓库 bindings 目录下执行：clean 后重新 cmake/make，再 pip install 生成的 whl，并重启本脚本。"
        f"当前 pyds 来自: {_pyds_file}"
    )

import numpy as np
from common.bus_call import bus_call

from eleproxy import (
    acquire_pipeline,
    acquire_v4l2_source_bin,
    acquire_nvstreammux,
    acquire_nvinfer,
    get_queue,
    acquire_nvdsosd,
    acquire_nvvideoconvert,
    acquire_nveglglessink,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-6s | %(threadName)s | %(filename)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename=f"./logs/{os.path.basename(__file__).split('.')[0]}-{time.strftime('%Y%m%d:%H%M')}.log",
    filemode="a",
)

logger = logging.getLogger(__name__)

# SGIE ArcFace unique-id, 与 nvosd_arcface_cbs 一致
PGIE_CLASS_ID_FACE = 0
SGIE_ARCFACE_UNIQUE_ID = 2
MAX_TIME_STAMP_LEN = 32
ARCFACE_EMBEDDING_DIM = 512


def get_arcface_embedding_from_obj(obj_meta, unique_id_filter=None):
    """
    从 NvDsObjectMeta 上挂的 user_meta 里取出 ArcFace embedding(512d & SGIE output-tensor-meta=1)

    :param obj_meta: pyds.NvDsObjectMeta
    :param unique_id_filter: 若指定，只取 tensor_meta.unique_id == unique_id_filter 的层
    :return: np.ndarray shape (512,) float32，若无则 None
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


def get_embedding():
    """
    从 face_db 文件加载所有 name -> 512 维向量的映射
    """
    from pathlib import Path

    FACE_DB_DIR = "/home/good/wkspace/deepstream-sdk/deepstream_python_apps/apps/dsapp/scripts/face_db.temp.txt"
    path = Path(FACE_DB_DIR)
    if not path.exists():
        raise FileNotFoundError(f"face_db not found: {path}")

    db: dict[str, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            first_space = line.index(" ")
            name = line[:first_space].strip()
            vec_str = line[first_space + 1 :].strip()
            values = [float(x.strip()) for x in vec_str.split(",")]
            db[name] = np.array(values, dtype=np.float32)
    logger.info(f"load face numver {len(db)}")
    return db


def nvsink_sink_pad_buffer_probe(osd_sink_pad, info, u_data):
    """
    nvdsosd sink pad 的 buffer probe: 对每个带 ArcFace embedding 的人脸,
    分配并填充 CTFaceObjectMeta, 挂到当前帧的 frame_user_meta_list
    u_data: faiss_index(IIndexFlatIP), 用于 name 匹配；可为 None
    """

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        logger.warning("Unable to get GstBuffer")
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if batch_meta is None:
        logger.warning("Unable to get batch meta from GstBuffer")
        return Gst.PadProbeReturn.OK

    pyds.nvds_acquire_meta_lock(batch_meta)
    try:
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            frame_number = frame_meta.frame_num
            frame_width = frame_meta.source_frame_width
            frame_height = frame_meta.source_frame_height
            source_id = frame_meta.source_id

            l_obj = frame_meta.obj_meta_list
            face_obj_metas = []
            face_embeddings = []
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
                emb = get_arcface_embedding_from_obj(obj_meta, SGIE_ARCFACE_UNIQUE_ID)
                if emb is not None:
                    face_obj_metas.append(obj_meta)
                    face_embeddings.append(emb)
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            # FAISS 匹配
            matched_names = []
            matched_scores = []
            if face_embeddings and u_data is not None:
                try:
                    matched_names, matched_scores = u_data.search_with_scores(
                        face_embeddings
                    )
                except Exception as e:
                    logger.warning(f"faiss search failed: {e}")

            # 为每个人脸分配 CTFaceObjectMeta, 填满并挂到 frame_user_meta
            for i, obj_meta in enumerate(face_obj_metas):
                user_meta = pyds.nvds_acquire_user_meta_from_pool(batch_meta)
                if user_meta is None:
                    continue
                ct_meta = pyds.alloc_ct_face_obj_struct(user_meta)
                if ct_meta is None:
                    continue

                # 从 obj_meta / frame 填满 CTFaceObjectMeta
                ct_meta.confidence = float(obj_meta.confidence)
                ct_meta.frameId = frame_number
                ct_meta.sensorid = source_id

                ct_meta.bbox.top = obj_meta.rect_params.top
                ct_meta.bbox.left = obj_meta.rect_params.left
                ct_meta.bbox.width = obj_meta.rect_params.width
                ct_meta.bbox.height = obj_meta.rect_params.height

                if i < len(matched_names):
                    ct_meta.name = matched_names[i]
                    ct_meta.id = str(i)
                else:
                    ct_meta.name = "Unknown"
                    ct_meta.id = str(i)
                if i < len(matched_scores):
                    ct_meta.confidence = float(matched_scores[i])

                ct_meta.ts = pyds.alloc_buffer(MAX_TIME_STAMP_LEN + 1)
                pyds.generate_ts_rfc3339(ct_meta.ts, MAX_TIME_STAMP_LEN)
                ct_meta.objectId = str(obj_meta.object_id)
                ct_meta.sensorStr = f"source_{source_id}"

                user_meta.user_meta_data = ct_meta
                user_meta.base_meta.meta_type = pyds.NvDsMetaType.NVDS_USER_META
                pyds.nvds_add_user_meta_to_frame(frame_meta, user_meta)

                # 日志输出(字符串字段在绑定层为指针, 需用 pyds.get_string 读取)
                if frame_number % 10 == 0:
                    logger.info(
                        f"CTFaceObjectMeta frame={frame_number} src={source_id} "
                        f"id={pyds.get_string(ct_meta.id) if ct_meta.id else ''} "
                        f"name={pyds.get_string(ct_meta.name) if ct_meta.name else ''} "
                        f"confidence={ct_meta.confidence:.3f} "
                        f"bbox=({ct_meta.bbox.left:.0f},{ct_meta.bbox.top:.0f},{ct_meta.bbox.width:.0f}x{ct_meta.bbox.height:.0f}) "
                        f"objectId={pyds.get_string(ct_meta.objectId) if ct_meta.objectId else ''} "
                        f"sensorStr={pyds.get_string(ct_meta.sensorStr) if ct_meta.sensorStr else ''}"
                    )

            try:
                l_frame = l_frame.next
            except StopIteration:
                break
    finally:
        pyds.nvds_release_meta_lock(batch_meta)

    return Gst.PadProbeReturn.OK


def main(uri):
    Gst.init(None)

    try:
        pipeline = acquire_pipeline()

        src_bin = acquire_v4l2_source_bin(
            index=0,
            args={
                "gpu_id": 0,
                "device": uri,
                "caps_v4l2": "video/x-raw,format=YUY2,framerate=30/1",
                "caps_sinksrc": "video/x-raw(memory:NVMM),format=NV12",
            },
        )

        streammux = acquire_nvstreammux(
            index=0,
            args={
                "gpu_id": 0,
                "batch_size": 1,
                "batched_push_timeout": 25000,
                "live_source": 1,  # 摄像头/RTSP 为实时源, 必须 1, 否则时间戳按文件处理导致 sink 大量丢帧
                "width": 1280,
                "height": 720,
            },
        )

        pgie = acquire_nvinfer(
            index=0,
            args={
                "gpu_id": 0,
                "batch_size": 1,
                "config_file_path": "/home/good/wkspace/deepstream-sdk/deepstream_python_apps/apps/dsapp/nvconfigs/dsapp_scrfd_pgie_config.txt",
            },
        )

        # XXX pgie and sgie should use different index
        sgie = acquire_nvinfer(
            index=1,
            args={
                "gpu_id": 0,
                "batch_size": 1,
                "config_file_path": "/home/good/wkspace/deepstream-sdk/deepstream_python_apps/apps/dsapp/nvconfigs/dsapp_sgie_arcface_config.txt",
            },
        )

        queue_1 = get_queue(index=0)

        nvvidconvert = acquire_nvvideoconvert(index=0)

        nvosd = acquire_nvdsosd(index=0)

        sink = acquire_nveglglessink(
            index=0,
            args={
                "sync": False
            },  # 推理较慢时 sync=True 会因“帧迟到”大量丢帧；False 则尽快显示, 画面更流畅
        )

        # add elements
        pipeline.add(src_bin)
        pipeline.add(streammux)
        pipeline.add(pgie)
        pipeline.add(sgie)
        pipeline.add(queue_1)
        pipeline.add(nvvidconvert)
        pipeline.add(nvosd)
        pipeline.add(sink)

        # link elements(v4l2 链末端 pad 连 streammux, 与 deepstream-test1-usbcam 一致)
        streammux_sink_pad = streammux.request_pad_simple("sink_0")
        if not streammux_sink_pad:
            raise RuntimeError("Unable to get the sink pad of streammux")
        src_bin_src_pad = src_bin.get_static_pad("src")
        if not src_bin_src_pad:
            raise RuntimeError("Unable to get the src pad of src_bin")
        src_bin_src_pad.link(streammux_sink_pad)

        streammux.link(pgie)
        pgie.link(sgie)
        sgie.link(queue_1)
        queue_1.link(nvvidconvert)
        nvvidconvert.link(nvosd)
        nvosd.link(sink)

        # FAISS 索引: 在创建 pipeline 的线程中建一次, 通过 user_data 传给回调, 避免在回调里重复创建
        try:
            from ifaiss import IIndexFlatIP as IIP

            name_to_emb = get_embedding()
            names = list(name_to_emb.keys())
            vectors = list(name_to_emb.values())
            # 阈值 0.6 对“现场摄像头 vs face_db 照片”往往过严, 同人相似度常见 0.35–0.55, 可先降到 0.4 试
            faiss_index = IIP(dim=512, threshold=0.4)
            faiss_index.build_index(names, vectors)
        except Exception as e:
            logger.warning(
                "FAISS index build skipped (%s), callback will get u_data=None", e
            )
            faiss_index = None

        # sink of nvosd 回调: 第三参为 user_data, 在回调中即 u_data
        osdsinkpad = nvosd.get_static_pad("sink")
        if not osdsinkpad:
            sys.stderr.write(" Unable to get sink pad of nvosd \n")
        osdsinkpad.add_probe(
            Gst.PadProbeType.BUFFER,
            nvsink_sink_pad_buffer_probe,
            faiss_index,
        )

    except BaseException as e:
        logger.error(f"Failed to create pipeline: {e}")
        sys.exit(1)

    main_loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, main_loop)

    # export GST_DEBUG_DUMP_DOT_DIR=/home/good/wkspace/deepstream-sdk/deepstream_python_apps/apps/dsapp/graphs
    g_file_name = f"graph.{int(time.time())}"
    Gst.debug_bin_to_dot_file(pipeline, Gst.DebugGraphDetails.ALL, g_file_name)

    logger.info("===> Starting pipeline")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        main_loop.run()
    except Exception:
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    uri: str = "/dev/video0"
    main(uri)
