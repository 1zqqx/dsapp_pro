import logging

import numpy as np
import faiss as F

logger = logging.getLogger(__name__)


class IIndexFlatIP:
    """
    使用向量内积 需要做 向量 归一化 计算余弦相似度 值 越接近 1 表示越 相似
    """

    def __init__(self, dim: int = 512, threshold: float = 0.6):
        """
        :param dim: 构造 index 的维度
        :param threshold: 向量匹配的临界值
        """
        self._DIM = dim
        self._th = threshold

        self._INDEX = F.IndexFlatIP(dim)

    @property
    def get_dim(self):
        return self._DIM

    def build_index(self, names: list[str], vectors: list[np.ndarray]):
        """
        构建 faiss.IndexFlatIP 使用 的 向量 与之 一一对应的名字 \n
        vectors 将在 该函数内 归一化

        :param names: 名字
        :param vectors: 人脸编码向量
        """
        if len(names) != len(vectors):
            raise ValueError(" names and vectors must have the same length ")

        self._names = list(names)
        x = np.asarray(vectors, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        F.normalize_L2(x)
        self._INDEX.add(x)

    def search(self, vectors: list[np.ndarray]) -> list[str]:
        """
        :param vectors: 待匹配/search 的向量

        :return:
        返回 vectors 中 每个向量 与已知的向量 最近似的向量 在 names 中对应的值 如若没有对应的阈值以内的 返回 `Unknown`
        """
        x = np.asarray(vectors, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        F.normalize_L2(x)
        D, I = self._INDEX.search(x, 1)
        result = []
        for i in range(len(x)):
            idx = int(I[i, 0])
            sim = float(D[i, 0])
            if idx >= 0 and sim >= self._th:
                result.append(self._names[idx])
            else:
                if idx >= 0:
                    logger.info(
                        f"FAISS match below threshold: sim={sim:.4f} < th={self._th:.2f}"
                        f" -> Unknown (nearest was {self._names[idx]})",
                    )
                else:
                    logger.info(f"FAISS no neighbor: idx={idx} -> Unknown")
                result.append("Unknown")
        return result

    def search_with_scores(
        self, vectors: list[np.ndarray]
    ) -> tuple[list[str], list[float]]:
        """
        与 search 相同, 但同时返回每个查询的相似度(余弦, [0,1]).

        :return: (names, scores) 两个列表, 与输入向量一一对应
        """
        x = np.asarray(vectors, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        F.normalize_L2(x)
        D, I = self._INDEX.search(x, 1)
        names_out = []
        scores_out = []
        for i in range(len(x)):
            idx = int(I[i, 0])
            sim = float(D[i, 0])
            if idx >= 0 and sim >= self._th:
                names_out.append(self._names[idx])
                scores_out.append(sim)
                continue
            names_out.append("Unknown")
            if idx >= 0:
                logger.debug(
                    f"FAISS match below threshold: sim={sim:.4f} < th={self._th:.2f} -> Unknown"
                )
                scores_out.append(sim)
            else:
                scores_out.append(0.0)
        return names_out, scores_out
