import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import time
import logging

import gi

gi.require_version("Gst", "1.0")

from gi.repository import Gst, GLib  # type: ignore
import numpy as np

from common.bus_call import bus_call
from eleproxy import *
from callbacks import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-6s | %(threadName)s | %(filename)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename=f"./logs/{os.path.basename(__file__).split('.')[0]}-{time.strftime('%Y%m%d:%H%M')}.log",
    filemode="a",
)

logger = logging.getLogger(__name__)


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
                "live_source": 1,  # 摄像头/RTSP 为实时源，必须 1，否则时间戳按文件处理导致 sink 大量丢帧
                "width": 1280,
                "height": 720,
            },
        )

        pgie = acquire_nvinfer(
            index=0,
            args={
                "gpu_id": 0,
                "batch_size": 1,
                # TODO 临时的
                # "config_file_path": "/home/good/wkspace/deepstream-sdk/deepstream_python_apps/apps/dsapp/nvconfigs/dsapp_pgie_config.txt",
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
            },  # 推理较慢时 sync=True 会因“帧迟到”大量丢帧；False 则尽快显示，画面更流畅
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

        # link elements（v4l2 链末端 pad 连 streammux，与 deepstream-test1-usbcam 一致）
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

        # FAISS 索引：在创建 pipeline 的线程中建一次，通过 user_data 传给回调，避免在回调里重复创建
        try:
            from ifaiss import IIndexFlatIP as IIP

            name_to_emb = get_embedding()
            names = list(name_to_emb.keys())
            vectors = list(name_to_emb.values())
            # 阈值 0.6 对“现场摄像头 vs face_db 照片”往往过严，同人相似度常见 0.35–0.55，可先降到 0.4 试
            faiss_index = IIP(dim=512, threshold=0.4)
            faiss_index.build_index(names, vectors)
        except Exception as e:
            logger.warning(
                "FAISS index build skipped (%s), callback will get u_data=None", e
            )
            faiss_index = None

        # sink of nvosd 回调：第三参为 user_data，在回调中即 u_data
        osdsinkpad = nvosd.get_static_pad("sink")
        if not osdsinkpad:
            sys.stderr.write(" Unable to get sink pad of nvosd \n")
        osdsinkpad.add_probe(
            Gst.PadProbeType.BUFFER,
            arcface_nvdsosd_sink_pad_buffer_probe,
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


def get_embedding():
    """
    从 face_db 文件加载所有 name -> 512 维向量的映射。
    """
    from pathlib import Path

    FACE_DB_DIR = (
        "/home/good/wkspace/deepstream-sdk/deepstream_python_apps/faiss/face_db.txt"
    )
    path = Path(FACE_DB_DIR)
    if not path.exists():
        raise FileNotFoundError(f"face_db not found: {path}")

    db: dict[str, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 格式: "name float1, float2, ..." ，第一个空格分隔名字与向量
            first_space = line.index(" ")
            name = line[:first_space].strip()
            vec_str = line[first_space + 1 :].strip()
            values = [float(x.strip()) for x in vec_str.split(",")]
            db[name] = np.array(values, dtype=np.float32)
    logger.info(f"load face numver {len(db)}")
    return db


if __name__ == "__main__":
    uri: str = (
        # "file:///home/good/wkspace/deepstream-sdk/ds8samples/streams/sample_1080p_h264.mp4"
        "/dev/video0"
        # "rtsp://127.0.0.1:8554/stream/1"
        # 注意 元组 与 字符串的区别 单元素元组需要逗号
    )
    main(uri)
