#!/usr/bin/env python3
"""
用人脸照片目录构建 face_db 文件：每人一个文件夹，多张照片编码后取均值再归一化，
输出为「名字 + 512 维向量」每行，与 DeepStream FAISS 使用的格式一致。

支持两种编码方式（二选一，推荐用 ONNX 与 DeepStream 完全一致）：
  1) --onnx_arcface PATH：使用指定 ArcFace ONNX（与 DeepStream SGIE 同模型，如 r50 或 2.5G）
  2) 不传 --onnx_arcface：使用 InsightFace FaceAnalysis 内置识别模型

依赖：opencv-python, numpy；若用 ONNX 需 onnxruntime；若用内置需 insightface
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2  # type: ignore
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 与 DeepStream / face_db 一致
EMBED_DIM = 512
# 常见图片后缀
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
USE_GPU: bool = False


def _align_face_112(img: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """根据 5 点关键点把人脸对齐到 112x112（ArcFace 标准输入），用左眼、右眼、鼻尖 3 点仿射。"""
    # ArcFace/InsightFace 标准 112x112 下 5 点目标坐标，取前 3 点做仿射
    dst_5 = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )
    src = np.asarray(kps, dtype=np.float32)
    if src.shape[0] < 3 or src.shape[1] != 2:
        raise ValueError("kps must be at least 3x2")
    dst = dst_5[:3]
    src3 = src[:3]
    M = cv2.getAffineTransform(src3, dst)
    out = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
    return out


def _preprocess_arcface_onnx(crop_112: np.ndarray) -> np.ndarray:
    """BGR 112x112 -> NCHW float32，与 DeepStream net-scale-factor=1/255 一致。"""
    x = cv2.cvtColor(crop_112, cv2.COLOR_BGR2RGB)
    x = x.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0)
    return x.astype(np.float32)


def load_arcface_onnx(onnx_path: str):
    """加载 ArcFace ONNX，返回 run(input_nchw) -> (512,) np.ndarray。"""
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError:
        raise ImportError("使用 --onnx_arcface 需要安装 onnxruntime")

    sess = ort.InferenceSession(
        onnx_path,
        providers=(
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if USE_GPU
            else ["CPUExecutionProvider"]
        ),
    )
    iname = sess.get_inputs()[0].name

    def run(img_nchw: np.ndarray) -> np.ndarray:
        out = sess.run(None, {iname: img_nchw})
        emb = out[0]
        if emb.ndim == 2:
            emb = emb[0]
        emb = np.asarray(emb, dtype=np.float32)
        return emb

    return run


def collect_embeddings_onnx(
    image_paths: list[Path],
    detector,
    encode_fn,
) -> list[np.ndarray]:
    """用检测器 + ONNX 编码器，对每张图取最大人脸 embedding。"""
    embeddings = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            logger.warning("skip (read fail): %s", path)
            continue
        faces = detector.get(img)
        if not faces:
            logger.warning("skip (no face): %s", path)
            continue
        # 取面积最大的人脸
        face = max(
            faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )
        if not hasattr(face, "kps") or face.kps is None:
            logger.warning("skip (no kps): %s", path)
            continue
        try:
            crop = _align_face_112(img, face.kps)
        except Exception as e:
            logger.warning("skip (align fail %s): %s", e, path)
            continue
        x = _preprocess_arcface_onnx(crop)
        emb = encode_fn(x)
        if emb.size != EMBED_DIM:
            logger.warning("skip (dim=%s): %s", emb.size, path)
            continue
        embeddings.append(emb)
    return embeddings


def collect_embeddings_insightface(image_paths: list[Path], app) -> list[np.ndarray]:
    """用 InsightFace FaceAnalysis 对每张图取最大人脸的 normed_embedding。"""
    embeddings = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            logger.warning("skip (read fail): %s", path)
            continue
        faces = app.get(img)
        if not faces:
            logger.warning("skip (no face): %s", path)
            continue
        face = max(
            faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )
        emb = getattr(face, "normed_embedding", None) or getattr(
            face, "embedding", None
        )
        if emb is None:
            logger.warning("skip (no embedding): %s", path)
            continue
        emb = np.asarray(emb, dtype=np.float32)
        if emb.size != EMBED_DIM:
            logger.warning("skip (dim=%s): %s", emb.size, path)
            continue
        embeddings.append(emb)
    return embeddings


def mean_then_normalize(embeddings: list[np.ndarray]) -> np.ndarray:
    """先求均值再 L2 归一化，返回 (512,) float32。"""
    if not embeddings:
        raise ValueError("embeddings is empty")
    mean = np.mean(embeddings, axis=0).astype(np.float32)
    norm = np.linalg.norm(mean)
    if norm <= 1e-12:
        return mean
    return (mean / norm).astype(np.float32)


def format_vector(vec: np.ndarray) -> str:
    """与现有 face_db 一致：逗号分隔的浮点数。"""
    return ", ".join(f"{x:.8f}" for x in vec.tolist())


def main():
    parser = argparse.ArgumentParser(
        description="从每人一个文件夹的人脸照片构建 face_db 文件（名字 + 512 维向量/行）"
    )
    parser.add_argument(
        "photo_dir",
        type=Path,
        help="照片根目录，其下每个子文件夹为一个人，文件夹名为该人名字",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("face_db.txt"),
        help="输出文件路径，默认 face_db.txt",
    )
    parser.add_argument(
        "--onnx_arcface",
        type=Path,
        default=None,
        help="ArcFace ONNX 路径（与 DeepStream SGIE 同模型时匹配最佳，如 r50 或 2.5G 等）",
    )
    parser.add_argument(
        "--det_size",
        type=int,
        nargs=2,
        default=[640, 640],
        metavar=("W", "H"),
        help="检测输入尺寸，默认 640 640",
    )
    args = parser.parse_args()

    photo_dir = args.photo_dir.resolve()
    if not photo_dir.is_dir():
        logger.error("photo_dir 不是目录: %s", photo_dir)
        sys.exit(1)

    # 每人一个子目录
    person_dirs = [d for d in photo_dir.iterdir() if d.is_dir()]
    if not person_dirs:
        logger.error("photo_dir 下没有子目录: %s", photo_dir)
        sys.exit(1)

    detector = None
    encode_fn = None
    use_onnx = args.onnx_arcface is not None and args.onnx_arcface.is_file()

    if use_onnx:
        logger.info("使用 ArcFace ONNX: %s", args.onnx_arcface)
        try:
            from insightface.app import FaceAnalysis  # type: ignore

            app = FaceAnalysis(
                providers=(
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if USE_GPU
                    else ["CPUExecutionProvider"]
                )
            )
            app.prepare(ctx_id=0, det_size=tuple(args.det_size))
            detector = app
            encode_fn = load_arcface_onnx(str(args.onnx_arcface))
        except Exception as e:
            logger.error("加载 ONNX/检测器失败: %s", e)
            sys.exit(1)
    else:
        logger.info("使用 InsightFace FaceAnalysis 内置识别模型")
        try:
            from insightface.app import FaceAnalysis  # type: ignore

            app = FaceAnalysis(
                providers=(
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if USE_GPU
                    else ["CPUExecutionProvider"]
                )
            )
            app.prepare(ctx_id=0, det_size=tuple(args.det_size))
            detector = app
        except Exception as e:
            logger.error("加载 InsightFace 失败: %s", e)
            sys.exit(1)

    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []

    for person_dir in sorted(person_dirs):
        name = person_dir.name
        image_paths = [
            p
            for p in person_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        if not image_paths:
            logger.warning("无图片，跳过: %s", name)
            continue

        if use_onnx:
            embeddings = collect_embeddings_onnx(image_paths, detector, encode_fn)
        else:
            embeddings = collect_embeddings_insightface(image_paths, detector)

        if not embeddings:
            logger.warning("无有效人脸编码，跳过: %s", name)
            continue

        vec = mean_then_normalize(embeddings)
        line = f"{name} {format_vector(vec)}\n"
        lines.append(line)
        logger.info(
            "%s: %d 张图 -> %d 个 embedding -> 1 条",
            name,
            len(image_paths),
            len(embeddings),
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    logger.info("已写入 %d 人 -> %s", len(lines), out_path)


if __name__ == "__main__":
    main()

"""
python build_face_db.py /home/good/wkspace/pyremodel/insightface/data/huaibei_face \
    --onnx_arcface /home/good/wkspace/pubdata/models/arcface/arcface_r50_webface.onnx
"""
