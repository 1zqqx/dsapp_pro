"""Shared inference chain: mux → … → pgie → tracker → sgies… → tee.

``inference.sgies`` is an ordered list of nvinfer configs (each dict like the
legacy single ``sgie``). Legacy ``inference.sgie`` alone is still accepted as a
one-element chain.
"""

from __future__ import annotations

import re

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # type: ignore

from eleproxy import (
    acquire_nvstreammux,
    acquire_nvdspreprocess,
    acquire_nvinfer,
    acquire_nvtracker,
    get_queue,
)
from logger.get_logger import get_logger

logger = get_logger(__name__)

# nvinfer 生成的 engine 文件名通常含 _b1_ / _b2_ 表示 batch;路径需与当前 batch 一致才会复用
_ENGINE_BATCH_PATTERN = re.compile(r"_b\d+_", re.IGNORECASE)


def _engine_path_for_batch(config_file_path: str | None, batch: int) -> str | None:
    """从 nvinfer 配置文件中读取 model-engine-file,并把其中的 _bN_ 换成当前 batch 的路径
    若文件或 key 不存在则返回 None,调用方不设置 model-engine-file(用配置文件原值)
    """
    if not config_file_path or batch < 1:
        return None
    try:
        with open(config_file_path, "r", encoding="utf-8") as f:
            in_property = False
            for line in f:
                line = line.strip()
                if line == "[property]":
                    in_property = True
                    continue
                if in_property and line.startswith("["):
                    break
                if in_property and line.startswith("model-engine-file="):
                    path = line.split("=", 1)[1].strip()
                    if _ENGINE_BATCH_PATTERN.search(path):
                        new_path = _ENGINE_BATCH_PATTERN.sub(f"_b{batch}_", path)
                        return new_path
                    return path
    except Exception as e:
        logger.debug(
            "Could not read model-engine-file from %s: %s", config_file_path, e
        )
    return None


def _sgie_configs_from_inference(cfg: dict) -> list[dict]:
    """Resolve SGIE configs: prefer non-empty ``sgies``, else legacy ``sgie``."""
    sgies = cfg.get("sgies")
    if isinstance(sgies, list) and len(sgies) > 0:
        return list(sgies)
    legacy = cfg.get("sgie")
    if legacy:
        return [legacy]
    raise ValueError(
        "inference config must set non-empty 'sgies' (list of dicts) or legacy 'sgie'"
    )


class InferenceChain:
    """Build and own the shared inference sub-pipeline.

    The chain is: mux → queue → preprocess → queue → pgie → nvtracker
    → nvinfer (sgies[0]) → … → nvinfer (sgies[-1]) → tee.

    Callers use :pyattr:`streammux` to request sink pads for each source and
    :pyattr:`tee` to attach downstream branches (message, demux, etc.).
    """

    def __init__(self, pipeline: Gst.Pipeline, config: dict = None):
        self._pipeline = pipeline
        self._cfg = config or {}
        # inner ele
        self._streammux: Gst.Element | None = None
        self._preprocess: Gst.Element | None = None
        self._pgie: Gst.Element | None = None
        self._nvtracker: Gst.Element | None = None
        self._sgie_cfgs: list[dict] = []
        self._sgies: list[Gst.Element] = []
        self._tee: Gst.Element | None = None

    def build(self) -> Gst.Element:
        """Create all elements, add to *pipeline*, link, and return the tee."""
        cfg = self._cfg
        p = self._pipeline

        self._streammux = acquire_nvstreammux(index=0, args=cfg.get("nvstreammux"))
        q_pre = get_queue(index=10)
        self._preprocess = acquire_nvdspreprocess(args=cfg.get("nvdspreprocess"))
        q_post = get_queue(index=11)
        self._pgie = acquire_nvinfer(index=0, args=cfg.get("pgie"))
        self._nvtracker = acquire_nvtracker(index=0, args=cfg.get("nvtracker"))
        self._sgie_cfgs = _sgie_configs_from_inference(cfg)
        self._sgies = [
            acquire_nvinfer(index=1 + i, args=sgie_args)
            for i, sgie_args in enumerate(self._sgie_cfgs)
        ]
        self._tee = Gst.ElementFactory.make("tee", "inference-tee")
        if not self._tee:
            raise RuntimeError("Unable to create tee")

        elements = [
            self._streammux,
            q_pre,
            self._preprocess,
            q_post,
            self._pgie,
            self._nvtracker,
            *self._sgies,
            self._tee,
        ]
        for el in elements:
            p.add(el)

        self._streammux.link(q_pre)
        q_pre.link(self._preprocess)
        self._preprocess.link(q_post)
        q_post.link(self._pgie)
        self._pgie.link(self._nvtracker)
        prev = self._nvtracker
        for sgie_el in self._sgies:
            prev.link(sgie_el)
            prev = sgie_el
        prev.link(self._tee)
        logger.info(f"InferenceChain built ({len(self._sgies)} sgie(s))")
        return self._tee

    def update_batch_size(self, n: int):
        """Update batch-size on mux / pgie / all sgies at runtime.
        同时把 pgie 与各 sgie 的 model-engine-file 设为与当前 batch 一致的路径(如 ..._b2_.engine),
        这样 nvinfer 会复用已存在的 engine,而不是每次从 ONNX 重建
        """
        for el in (self._streammux, self._pgie, *self._sgies):
            if el is not None:
                el.set_property("batch-size", n)
        pgie_cfg = self._cfg.get("pgie") or {}
        if self._pgie is not None:
            engine_path = _engine_path_for_batch(pgie_cfg.get("config_file_path"), n)
            if engine_path:
                self._pgie.set_property("model-engine-file", engine_path)
        for sgie_el, sgie_cfg in zip(self._sgies, self._sgie_cfgs):
            engine_path = _engine_path_for_batch(sgie_cfg.get("config_file_path"), n)
            if engine_path:
                sgie_el.set_property("model-engine-file", engine_path)
        logger.info(f"batch-size updated to {n}")

    @property
    def streammux(self) -> Gst.Element:
        assert self._streammux is not None, "build() not called yet"
        return self._streammux

    @property
    def preprocess(self) -> Gst.Element:
        assert self._preprocess is not None, "build() not called yet"
        return self._preprocess

    @property
    def pgie(self) -> Gst.Element:
        assert self._pgie is not None, "build() not called yet"
        return self._pgie

    @property
    def sgies(self) -> list[Gst.Element]:
        assert self._sgies, "build() not called yet"
        return self._sgies

    @property
    def sgie(self) -> Gst.Element:
        """First secondary nvinfer (backward compatible when only one SGIE)."""
        assert self._sgies, "build() not called yet"
        return self._sgies[0]

    @property
    def tee(self) -> Gst.Element:
        assert self._tee is not None, "build() not called yet"
        return self._tee
