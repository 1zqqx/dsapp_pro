import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import time
import logging

import gi

gi.require_version("Gst", "1.0")
# gi.require_version("GstRtspServer", "1.0")

from gi.repository import Gst, GLib  # type: ignore

from common.bus_call import bus_call
from eleproxy import *
from callbacks import scrfd_nvdsosd_sink_pad_buffer_probe

logging.basicConfig(
    level=logging.INFO,
    # | %(name)s |
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

        # create elements
        src_bin = acquire_nvurisrcbin(
            index=0,
            args={
                "uri": uri,
                "file-loop": 1,
                "latency": 300,
            },
        )

        streammux = acquire_nvstreammux(
            index=0,
            args={
                "gpu_id": 0,
                "batch_size": 1,
                "batched_push_timeout": 25000,
                "live_source": 0,  # 摄像头/RTSP 为实时源，必须 1，否则时间戳按文件处理导致 sink 大量丢帧; 本地文件时则必须 0，否则会“尽力而为”导致速度不正常
                "width": 2560,
                "height": 1440,
            },
        )

        queue_0 = get_queue(index=0)

        preprocess = Gst.ElementFactory.make("nvdspreprocess", "preprocess-plugin")
        if not preprocess:
            sys.stderr.write(" Unable to create preprocess \n")
        preprocess.set_property(
            "config-file",
            "/home/good/wkspace/deepstream-sdk/deepstream_python_apps/apps/dsapp/nvconfigs/dsapp_preprocess_config.txt",
        )

        queue_1 = get_queue(index=1)

        pgie = acquire_nvinfer(
            index=0,
            args={
                "gpu_id": 0,
                "batch_size": 1,
                "input_tensor_meta": 1,
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

        queue_2 = get_queue(index=2)

        nvvidconvert = acquire_nvvideoconvert(index=0)

        nvosd = acquire_nvdsosd(index=0)

        sink = acquire_nveglglessink(
            index=0,
            args={
                "sync": True,  # True 则按帧率显示, 按照管道时钟, False 则尽快显示
            },  # 推理较慢时 sync=True 会因“帧迟到”大量丢帧; False 则尽快显示，画面更流畅; 但是 False 在播放本地视频时就会“尽力而为”导致速度不正常
        )

        # add elements
        pipeline.add(src_bin)
        pipeline.add(streammux)
        pipeline.add(queue_0)
        pipeline.add(preprocess)
        pipeline.add(queue_1)
        pipeline.add(pgie)
        pipeline.add(sgie)
        pipeline.add(queue_2)
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

        streammux.link(queue_0)
        queue_0.link(preprocess)
        preprocess.link(queue_1)
        queue_1.link(pgie)
        pgie.link(sgie)
        sgie.link(queue_2)
        queue_2.link(nvvidconvert)
        nvvidconvert.link(nvosd)
        nvosd.link(sink)

        # sink of nvosd 回调 函数
        osdsinkpad = nvosd.get_static_pad("sink")
        if not osdsinkpad:
            sys.stderr.write(" Unable to get sink pad of nvosd \n")
        osdsinkpad.add_probe(
            Gst.PadProbeType.BUFFER, scrfd_nvdsosd_sink_pad_buffer_probe, 0
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
    uri: str = (
        # "file:///home/good/wkspace/deepstream-sdk/ds8samples/streams/sample_1080p_h264.mp4"
        # "file:///home/good/temp/video_src/output_test_30_ds2.mp4"
        "file:///home/good/temp/video_src/2025-12-06_12.36_1min.mp4"
    )
    main(uri)
