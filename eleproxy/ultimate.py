import gi
import pyds

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")

from gi.repository import Gst, GLib  # type: ignore


def acquire_autovideosink(index: int = 0, args=None):
    """
    Create an autovideosink element with appropriate properties set.

    Returns:
    + An autovideosink element.
    """

    video_sink = Gst.ElementFactory.make("autovideosink", f"video-sink-{index:02}")
    if not video_sink:
        raise RuntimeError(" Unable to create autovideosink ")

    # sync=True：按管道时钟/帧率播放，保持正常速度；sync=False 会尽快显示帧导致快放
    video_sink.set_property("sync", True)

    return video_sink


def acquire_nveglglessink(index: int = 0, args=None):
    """
    Create an nveglglessink element with appropriate properties set.


    args:
    + sync: bool, default True: 是否按管道时钟/帧率播放，保持正常速度；sync=False 会尽快显示帧导致快放

    Returns:
    + An nveglglessink element.
    """

    args = args or {}
    _sync: bool = args.get("sync", True)

    nveglglessink = Gst.ElementFactory.make("nveglglessink", f"nvegl-sink-{index:02}")
    if not nveglglessink:
        raise RuntimeError(" Unable to create nveglglessink ")

    nveglglessink.set_property("sync", _sync)

    return nveglglessink


def acquire_nv3dsink(index: int = 0, args=None):
    """Use the integrated gpu or aarch64 card"""
    pass
