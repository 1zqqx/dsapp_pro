from __future__ import annotations

import argparse
import signal
import threading
import time
from pathlib import Path

from .dsAppApi import run_from_config, stop_pipeline
from .logger.get_logger import get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run dsapp pipeline from JSON config")
    parser.add_argument(
        "--pipeline",
        required=True,
        help="Pipeline name to run, e.g. face_recognition",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to instance config JSON file",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    pipeline = None
    stop_event = threading.Event()

    def _request_stop(signum: int, _frame) -> None:
        logger.info("Received signal %s, stopping pipeline...", signum)
        stop_event.set()
        stop_pipeline(pipeline)

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    try:
        pipeline = run_from_config(
            config_path=Path(args.config),
            pipeline_name=args.pipeline,
        )
        while not stop_event.is_set():
            time.sleep(1)
        return 0
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, stopping pipeline...")
        return 0
    except Exception as exc:
        logger.exception(
            "Failed to run dsapp (pipeline=%s, config=%s)",
            args.pipeline,
            args.config,
        )
        print(f"ERROR: {exc}")
        return 1
    finally:
        stop_pipeline(pipeline)
