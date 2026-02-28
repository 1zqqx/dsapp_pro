import gi
import logging

gi.require_version("Gst", "1.0")

from gi.repository import Gst, GLib  # type: ignore

logger = logging.getLogger(__name__)


def acquire_nvdspreprocess(args: dict = None):
    """
    Create nvmsgbroker instance to send NvDsPayload to remote (Redis/Kafka/MQTT etc.).

    args:
    + config_file: str, configuraton config path.

    Returns:
    + Gst.Element: nvdspreprocess instance.
    """

    args = args or {}
    preprocess = Gst.ElementFactory.make("nvdspreprocess", "preprocess-plugin")
    if not preprocess:
        raise RuntimeError(" Unable to create preprocess \n")

    config_file = args.get("config_file", None)
    if config_file is not None:
        preprocess.set_property("config-file", config_file)

    return preprocess
