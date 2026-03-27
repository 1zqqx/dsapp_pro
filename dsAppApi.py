from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

from .logger.get_logger import get_logger, setup_logging

logger = get_logger(__name__)


def _create_face_recognition_pipeline(config: dict[str, Any]) -> Any:
    from .pipelines.face_recognition import FaceRecognitionPipeline

    return FaceRecognitionPipeline(config)


PIPELINE_REGISTRY: dict[str, Callable[[dict[str, Any]], Any]] = {
    "FACE_RECOGNITION": _create_face_recognition_pipeline,
}


def _ensure_dsapp_root_on_syspath() -> Path:
    """Ensure dsapp root directory is importable as top-level modules."""
    dsapp_root = Path(__file__).resolve().parent
    dsapp_root_str = str(dsapp_root)
    if dsapp_root_str not in sys.path:
        sys.path.insert(0, dsapp_root_str)
    return dsapp_root


def load_config(config_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load pipeline JSON config."""
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Config path is not a file: {path}")

    try:
        content = path.read_text(encoding="utf-8")
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Config JSON parse failed: {path} ({exc})") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Config root must be JSON object: {path}")

    return data


def build_pipeline(pipeline_name: str, config: dict[str, Any]) -> Any:
    factory = PIPELINE_REGISTRY.get(pipeline_name)
    if factory is None:
        supported = ", ".join(sorted(PIPELINE_REGISTRY.keys())) or "<empty>"
        raise ValueError(
            f"Unsupported pipeline '{pipeline_name}'. Supported pipelines: {supported}"
        )
    return factory(config)


def run_from_config(
    config_path: str | os.PathLike[str],
    pipeline_name: str,
):
    """
    Build and start a registered pipeline from a config file.
    Returns the running pipeline handle.
    """
    config = load_config(config_path)
    _ensure_dsapp_root_on_syspath()
    setup_logging(level="INFO", log_file=None)
    pipeline = build_pipeline(pipeline_name=pipeline_name, config=config)
    pipeline.build()
    pipeline.start()
    logger.info(
        "Pipeline started (pipeline=%s, config=%s)",
        pipeline_name,
        Path(config_path),
    )
    return pipeline


def stop_pipeline(pipeline: Any) -> None:
    """Best-effort stop helper with error tolerance."""
    if pipeline is None:
        return
    try:
        pipeline.stop()
    except Exception:
        logger.exception("Failed to stop pipeline cleanly")
