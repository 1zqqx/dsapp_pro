import gi
import pyds

gi.require_version("Gst", "1.0")
# gi.require_version("GstRtspServer", "1.0")

from gi.repository import Gst, GLib  # type: ignore


def acquire_pipeline():
    pipeline = Gst.Pipeline()

    if not pipeline:
        raise RuntimeError(" Unable to create Pipeline \n")

    return pipeline
