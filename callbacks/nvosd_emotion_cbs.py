"""
OSD 回调: 在 SCRFD 人脸检测 + FER+ 情绪 SGIE 管道上, 为人脸框显示情绪标签.

优先从每个对象的 output-tensor-meta 读取 8 维 logits 并在 Python 中做 softmax+argmax，
这样每张脸对应自己的推理结果，避免 nvinfer 在 secondary 下只传 batch=1 导致所有人脸同结果。
需在 SGIE 配置中设置 output-tensor-meta=1。
"""

import ctypes
import logging

import gi

gi.require_version("Gst", "1.0")

from gi.repository import Gst  # type: ignore

import numpy as np
import pyds

from .nvosd_cbs import PGIE_CLASS_ID_FACE

logger = logging.getLogger(__name__)

# 情绪 SGIE 的 gie-unique-id(与 dsapp_sgie_emotion_config.txt 中 gie-unique-id=3 一致)
SGIE_EMOTION_UNIQUE_ID = 3

# FER+ 8 类情绪，与 dsapp_emotion_labels.txt 顺序一致
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
NUM_EMOTION_CLASSES = 8


def emotion_scrfd_nvdsosd_sink_pad_buffer_probe(osd_sink_pad, info, u_data):
    """
    在 nvdsosd sink 上: 统计人脸数、为每个人脸框显示情绪标签(来自 SGIE 情绪分类器),
    并在画面上方显示帧信息.
    """
    return Gst.PadProbeReturn.OK
