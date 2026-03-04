import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import logging
import numpy as np

logger = logging.getLogger(__name__)


def _get_embedding(face_db_dir: str = None):
    """从 face_db 文本文件加载所有 name → 512 维向量的映射."""
    from pathlib import Path

    path = Path(face_db_dir)
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
    logger.info(f"load face number {len(db)}")
    return db


def build_faiss_index(faiss_cfg: dict = None):
    """
    构建 FAISS 索引; 失败时返回 None 而不中断管道.

    args:
    + face_db_dir: face embeding file path
    """
    if faiss_cfg is None:
        logger.warning("FAISS config is None, skipping index build")
        return None
    try:
        from ifaiss import IIndexFlatIP as IIP

        name_to_emb = _get_embedding(faiss_cfg.get("face_db_dir"))
        names = list(name_to_emb.keys())
        vectors = list(name_to_emb.values())
        faiss_index = IIP(dim=512, threshold=0.4)
        faiss_index.build_index(names, vectors)
        return faiss_index
    except Exception as e:
        logger.warning("FAISS index build skipped (%s), probe will get None", e)
        return None
