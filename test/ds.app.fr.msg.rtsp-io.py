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


def get_example_config(pgie_roi_batch: int = 4):
    """返回符合 FaceRecognitionPipeline 的配置结构."""
    if pgie_roi_batch not in (1, 2, 4):
        raise ValueError(f"pgie_roi_batch must be 2 or 4, got {pgie_roi_batch}")

    # 当前示例固定两路源:
    # - b4: 每路 2 个 ROI => 总 ROI=4 => pgie batch-size=4
    # - b2: 每路 1 个 ROI => 总 ROI=2 => pgie batch-size=2
    preprocess_cfg = (
        "/home/good/wkspace/deepstream-sdk/deepstream_python_apps/apps/dsapp/nvconfigs/"
        f"dsapp_preprocess_pgie_config_b{pgie_roi_batch}.txt"
    )
    engine_path = (
        f"/home/good/wkspace/pubdata/models/scrfd/scrfd_2.5g_bnkps_640x640."
        f"onnx_b{pgie_roi_batch}_gpu0_fp16.engine"
    )
    # 若对应 batch 的 engine 不存在，让 nvinfer 使用 onnx-file 构建（更慢但能避免直接失败）
    if not os.path.exists(engine_path):
        logger.warning(
            "Engine not found for b=%s: %s (will build from onnx)",
            pgie_roi_batch,
            engine_path,
        )
        engine_path = None

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
                "config_file": preprocess_cfg,
            },
            "pgie": {
                "gpu_id": 0,
                "batch_size": pgie_roi_batch,
                "input_tensor_meta": True,
                "config_file_path": "/home/good/wkspace/deepstream-sdk/deepstream_python_apps/apps/dsapp/nvconfigs/dsapp_pgie_scrfd_config.txt",
                "model_engine_file": engine_path,
            },
            # "nvdspreprocess_sgie": {
            #     "config_file": "/home/good/wkspace/deepstream-sdk/deepstream_python_apps/apps/dsapp/nvconfigs/dsapp_preprocess_sgie_1_config.txt",
            # },
            "sgie": {
                "gpu_id": 0,
                "batch_size": 8,
                "config_file_path": "/home/good/wkspace/deepstream-sdk/deepstream_python_apps/apps/dsapp/nvconfigs/dsapp_sgie_arcface_config.txt",
                "model_engine_file": "/home/good/wkspace/pubdata/models/arcface/arcface_r50_webface.onnx_b8_gpu0_fp16.engine",
            },
            "nvtracker": {
                "config_file": "/home/good/wkspace/deepstream-sdk/deepstream_python_apps/apps/dsapp/nvconfigs/dsapp_tracker_config.txt",
            },
        },
        "faiss": {
            "face_db_dir": "/home/good/wkspace/deepstream-sdk/deepstream_python_apps/apps/dsapp/scripts/face_db.temp.txt",
        },
    }


def run_basic(pgie_roi_batch: int = 4):
    """仅构建并启动管道, Ctrl+C 退出."""
    from face_recognition import FaceRecognitionPipeline

    config = get_example_config(pgie_roi_batch=pgie_roi_batch)

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


def run_with_dynamic_api(pgie_roi_batch: int = 4):
    """构建并启动后,在另一线程中演示动态 add_source / remove_source / enable_rtsp / disable_rtsp."""
    import threading

    from face_recognition import FaceRecognitionPipeline

    config = get_example_config(pgie_roi_batch=pgie_roi_batch)
    # 先只用一个源启动
    config["sources"] = config["sources"][:1]
    # for k in ("nvstreammux", "pgie", "sgie"):
    #     config["inference"][k]["batch_size"] = 1

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

        time.sleep(60)

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
    parser.add_argument(
        "--pgie-batch",
        type=int,
        default=4,
        choices=[1, 2, 4],
        help="SCRFD 的 preprocess tensor batch + pgie batch-size(2 或 4)",
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    if args.dynamic:
        # dynamic 模式当前未完整实现 batch/engine 的运行时切换.
        # 为了降低变量，固定使用 batch=4.
        run_with_dynamic_api(pgie_roi_batch=4)
    else:
        run_basic(pgie_roi_batch=args.pgie_batch)
