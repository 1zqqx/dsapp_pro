#!/usr/bin/env python3
"""
FaceRecognitionPipeline 使用示例.

演示：
  1. 使用配置构建管道并启动
  2. 可选：从另一线程调用线程安全 API(动态添加/移除源、开关 RTSP)

运行方式(在 dsapp 目录下):
  cd apps/dsapp && python pipelines/face_recognition/example_run.py
  cd apps/dsapp && python pipelines/face_recognition/example_run.py --dynamic

运行前请根据本机路径修改 get_example_config() 中的 nvconfigs、face_db、mediamtx_url 等.
"""
from __future__ import annotations

import os
import sys
import time
import signal

# 保证能导入 dsapp 包(pipelines 的父目录)
_DSAPP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _DSAPP_ROOT not in sys.path:
    sys.path.insert(0, _DSAPP_ROOT)

from logger.get_logger import get_logger, setup_logging

# 先初始化日志,再导入依赖 GStreamer 的模块
setup_logging(level="DEBUG", log_file=None)
logger = get_logger(__name__)


def get_example_config():
    """返回符合 FaceRecognitionPipeline 的配置结构."""

    return {
        "debug_batch_meta_trace": True,
        "sources": [
            {
                "uri": "rtsp://127.0.0.1:10010/stream/1",
                "source_id": "cam-1",
                "mux_slot": 0,  # 注意 与 ROI 的 id 一致
                "latency": 300,
                "rtsp_output": {
                    "enabled": True,
                    "mediamtx_url": "rtsp://127.0.0.1:8554/stream/1",
                    "bitrate": 8000000,
                    "rtsp_protocols": 4,
                },
            },
            {
                "uri": "rtsp://127.0.0.1:10010/stream/2",
                "source_id": "cam-2",
                "mux_slot": 1,
                "latency": 300,
                "rtsp_output": {
                    "enabled": True,
                    "bitrate": 8000000,
                    "rtsp_protocols": 4,
                    "mediamtx_url": "rtsp://127.0.0.1:8554/stream/2",
                },
            },
        ],
        "inference": {
            "nvstreammux": {
                "gpu_id": 0,
                "batch_size": 2,
                "batched_push_timeout": 35000,
                "live_source": 1,
                "width": 2560,
                "height": 1440,
            },
            "nvdspreprocess": {
                "config_file": "/home/good/wkspace/deepstream-sdk/deepstream_python_apps/apps/dsapp/nvconfigs/dsapp_preprocess_pgie_config.txt",
            },
            "pgie": {
                "gpu_id": 0,
                "batch_size": 1,  # FIXME modify, 2;4 & [model_engine_file] -> bug
                "input_tensor_meta": True,
                "config_file_path": "/home/good/wkspace/deepstream-sdk/deepstream_python_apps/apps/dsapp/nvconfigs/dsapp_pgie_scrfd_config.txt",
                # "model_engine_file": "/home/good/wkspace/pubdata/models/scrfd/scrfd_2.5g_bnkps_640x640.onnx_b4_gpu0_fp16.engine",
            },
            # "nvdspreprocess_sgie": {
            #     "config_file": "/home/good/wkspace/deepstream-sdk/deepstream_python_apps/apps/dsapp/nvconfigs/dsapp_preprocess_sgie_1_config.txt",
            # },
            "sgie": {
                "gpu_id": 0,
                "batch_size": 8,
                "config_file_path": "/home/good/wkspace/deepstream-sdk/deepstream_python_apps/apps/dsapp/nvconfigs/dsapp_sgie_arcface_config.txt",
                # "model_engine_file": "/home/good/wkspace/pubdata/models/arcface/arcface_r50_webface.onnx_b8_gpu0_fp16.engine",
            },
            "nvtracker": {
                "config_file": "/home/good/wkspace/deepstream-sdk/deepstream_python_apps/apps/dsapp/nvconfigs/dsapp_tracker_config.txt",
            },
        },
        "faiss": {
            "face_db_dir": "/home/good/wkspace/deepstream-sdk/deepstream_python_apps/apps/dsapp/scripts/face_db.temp.txt",
        },
        "message": {
            # 可选：启用消息分支(Redis 等)
            "nvmsgconv": {
                "config": "/home/good/wkspace/deepstream-sdk/deepstream_python_apps/apps/dsapp/nvconfigs/dstest4_msgconv_config.txt",
                "payload_type": 1,
                "msg2p_newapi": True,
                "dummy_payload": True,
                "frame_interval": 25,
            },
            "nvmsgbroker": {
                "proto_lib": "/opt/nvidia/deepstream/deepstream/lib/libnvds_redis_proto.so",
                "conn_str": "127.0.0.1;6399",
                "config": "/home/good/wkspace/deepstream-sdk/deepstream_python_apps/apps/dsapp/nvconfigs/dsapp_msgbroker_cfg_redis.txt",
                "streamsize": 1000,
                "topic": "FR-topic-batchsize-fixed",
                "sync": False,
            },
        },
    }


def run_basic():
    """仅构建并启动管道, Ctrl+C 退出."""
    from face_recognition import FaceRecognitionPipeline

    config = get_example_config()

    pipeline = FaceRecognitionPipeline(config)
    pipeline.build()
    # export GST_DEBUG_DUMP_DOT_DIR=/home/good/wkspace/deepstream-sdk/deepstream_python_apps/apps/dsapp/graphs.temp
    pipeline.set_save_pipeline_graph(
        target_dir="/home/good/wkspace/deepstream-sdk/deepstream_python_apps/apps/dsapp/graphs.temp",
        graph_name="graph.basic.4",
    )
    pipeline.start()

    def on_sig(sig, frame):
        logger.info(f" Received {sig}, stopping...")
        pipeline.stop()

    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
    logger.info("Done.")


def run_with_dynamic_api():
    """构建并启动后,在另一线程中演示动态 add_source / remove_source / enable_rtsp / disable_rtsp."""
    import threading

    from face_recognition import FaceRecognitionPipeline

    config = get_example_config()
    # 先只用一个源启动
    config["sources"] = config["sources"][:1]
    for k in ("nvstreammux", "pgie", "sgie"):
        config["inference"][k]["batch_size"] = 1

    pipeline = FaceRecognitionPipeline(config)
    pipeline.build()
    pipeline.start()

    added_pad_index = [None]  # 用 list 以便闭包修改

    def demo_api():
        time.sleep(20)
        # 动态添加第二路源
        pad = pipeline.add_source(
            {
                "uri": "rtsp://127.0.0.1:10010/stream/2",
                "source_id": "cam-2",
                "mux_slot": 1,
                "latency": 300,
                "rtsp_output": {
                    "enabled": False,
                    "rtsp_protocols": 4,
                    "mediamtx_url": "rtsp://127.0.0.1:8554/stream/2",
                },
            }
        )
        added_pad_index[0] = pad
        logger.info(f"Dynamic add_source -> pad_index={pad}")
        time.sleep(5)
        pipeline.enable_rtsp(pad)
        logger.info(f"enable_rtsp({pad})")
        time.sleep(60)
        pipeline.disable_rtsp(pad)
        logger.info(f"disable_rtsp({pad})")
        time.sleep(5)
        pipeline.remove_source(pad)
        logger.info(f"remove_source({pad})")

    t = threading.Thread(target=demo_api, daemon=True)
    logger.info(" Start Change ")
    t.start()

    def on_sig(sig, frame):
        pipeline.stop()

    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)
    try:
        while t.is_alive():
            t.join(timeout=1)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
    logger.info("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FaceRecognitionPipeline 示例")
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="演示动态 add/remove source 与 enable/disable RTSP",
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    if args.dynamic:
        run_with_dynamic_api()
    else:
        run_basic()
