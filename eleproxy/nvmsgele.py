import re
import tempfile
import gi
import logging

gi.require_version("Gst", "1.0")

from gi.repository import Gst, GLib  # type: ignore

logger = logging.getLogger(__name__)


def _msgbroker_config_with_streamsize(config_path: str, streamsize: int) -> str:
    """
    从已有配置文件读取内容，覆盖 [message-broker] 中的 streamsize，写入临时文件并返回路径。
    """
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()
    # 在 [message-broker] 段内替换 streamsize=数字 为 streamsize=streamsize；若无则在该段末尾添加
    new_line = f"streamsize={streamsize}"
    if re.search(r"streamsize\s*=\s*\d+", content, re.IGNORECASE):
        content = re.sub(
            r"streamsize\s*=\s*\d+", new_line, content, flags=re.IGNORECASE
        )
    else:
        # 在 [message-broker] 后第一个空行或下一段前插入
        content = re.sub(
            r"(\[message-broker\]\s*\n)",
            r"\1" + new_line + "\n",
            content,
            count=1,
        )
    fd, path = tempfile.mkstemp(suffix=".txt", prefix="dsapp_msgbroker_")
    with open(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def acquire_nvmsgbroker(index: int, args: dict = None):
    """
    Create nvmsgbroker instance to send NvDsPayload to remote (Redis/Kafka/MQTT etc.).

    args:
    + proto_lib: str, protocol adaptor library path (e.g. libnvds_redis_proto.so).
    + conn_str: str, connection string (e.g. "127.0.0.1;6379").
    + config: str, optional config file path (Redis: hostname, port, streamsize, payloadkey, etc.).
    + streamsize: int, optional Redis stream max length; 若同时提供 config，会生成临时配置覆盖原文件中的 streamsize.
    + topic: str, optional topic/channel name.
    + sync: bool, default False.

    Returns:
    + Gst.Element: nvmsgbroker instance.
    """
    args = args or {}
    el = Gst.ElementFactory.make("nvmsgbroker", f"nvmsgbroker-{index:02d}")
    if not el:
        raise RuntimeError(" Unable to create nvmsgbroker")

    if args.get("proto_lib") is not None:
        el.set_property("proto-lib", args["proto_lib"])
    if args.get("conn_str") is not None:
        el.set_property("conn-str", args["conn_str"])

    config_path = args.get("config")
    streamsize = args.get("streamsize")
    if streamsize is not None and config_path:
        try:
            config_path = _msgbroker_config_with_streamsize(
                config_path, int(streamsize)
            )
        except Exception as e:
            logger.warning(
                "Override streamsize in config failed (%s), using original config", e
            )
    if config_path is not None:
        el.set_property("config", config_path)

    if args.get("topic") is not None:
        el.set_property("topic", args["topic"])
    if args.get("sync") is not None:
        el.set_property("sync", bool(args["sync"]))
    return el


def acquire_nvmsgconv(index: int, args: dict = None):
    """
    Create nvmsgconv instance to convert NvDsEventMsgMeta etc. to schema payload (e.g. JSON).
    + For property payload_type
        - 1=custom;
        - 0=DeepStream schema
        - ?
        - schema = 0: Full message schema with separate payload per object (Default);
        - schema = 1; Minimal message with multiple objects in single payload.

    args:
    + config: str, path to msgconv config file.
    + payload_type: int, schema type (0=DeepStream, 257=PAYLOAD_CUSTOM for custom msg2p-lib).
    + msg2p_lib: str, absolute path to custom payload generator .so (when payload_type=257).

    Returns:
    + Gst.Element: nvmsgconv instance.
    """
    args = args or {}
    el = Gst.ElementFactory.make("nvmsgconv", f"nvmsgconv-{index:02d}")
    if not el:
        raise RuntimeError("Unable to create nvmsgconv")

    if args.get("config") is not None:
        el.set_property("config", args["config"])
    if args.get("payload_type") is not None:
        el.set_property("payload-type", int(args["payload_type"]))
    if args.get("msg2p_lib") is not None:
        el.set_property("msg2p-lib", args["msg2p_lib"])
    return el
