"""
项目集中式日志配置.

库模块用法:
    from dsapp.logger import get_logger
    logger = get_logger(__name__)

入口脚本用法:
    from dsapp.logger import get_logger, setup_logging
    setup_logging(level="DEBUG", log_file="./logs/my_pipeline.log")
    logger = get_logger(__name__)
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

_DEFAULT_FMT = (
    "%(asctime)s | %(levelname)-6s | %(threadName)s "
    "| %(filename)s:%(lineno)d | %(message)s"
)
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

_setup_done = False


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    fmt: Optional[str] = None,
    datefmt: Optional[str] = None,
) -> None:
    """
    配置 root logger(幂等,仅首次调用生效).

    Parameters
    ----------
    level : str | None
        日志级别名称,如 "DEBUG"、"INFO".
        优先使用传入值,其次读取环境变量 DSAPP_LOG_LEVEL,默认 "INFO".
    log_file : str | None
        日志文件路径. 若为 None 则自动生成 ./logs/dsapp-<日期时间>.log.
        ;传入空字符串 "" 表示不写文件.
    fmt : str | None
        日志格式字符串,默认沿用项目统一格式.
    datefmt : str | None
        时间格式字符串,默认 "%Y-%m-%d %H:%M:%S".
    """
    global _setup_done
    if _setup_done:
        return
    _setup_done = True

    level_name = level or os.environ.get("DSAPP_LOG_LEVEL", "INFO")
    log_level = getattr(logging, level_name.upper(), logging.INFO)
    fmt = fmt or _DEFAULT_FMT
    datefmt = datefmt or _DEFAULT_DATEFMT

    root = logging.getLogger()
    root.setLevel(log_level)

    formatter = logging.Formatter(fmt, datefmt=datefmt)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file != "":
        if log_file is None:
            log_dir = Path("./logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = str(log_dir / f"dsapp-{time.strftime('%Y%m%d-%H%M%S')}.log")
        else:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(formatter)
        root.addHandler(fh)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取 logger 实例.

    首次调用时若 root logger 尚未配置,会自动执行 setup_logging() 以确保
    即使库模块单独导入也有合理的默认输出.

    Parameters
    ----------
    name : str | None
        logger 名称,通常传 __name__.
    """
    if not _setup_done:
        setup_logging()
    return logging.getLogger(name)
