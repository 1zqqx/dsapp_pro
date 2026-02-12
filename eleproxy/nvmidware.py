import gi
import logging

gi.require_version("Gst", "1.0")

from gi.repository import Gst, GLib  # type: ignore

logger = logging.getLogger(__name__)


def acquire_nvstreammux(index: int = 0, args: dict = None):
    """
    Create nvstreammux instance to form batches from one or more sources.
    All properties will use profile properties if they are not set.

    out:
    + Availability: Always
        - Capabilities:
            video/x-raw(memory:NVMM)

    args:
    + gpu_id: Set GPU Device ID.

    + batch_size: Number of frames to batch before pushing to the next element.

    + batched_push_timeout:Timeout in microseconds to wait after the first buffer is available
                    to push the batch even if the complete batch is not formed.

    + live_source: 1=live (camera/RTSP), 0=non-live (file). Must be 1 for live sources else sink drops frames as "too late".

    + width: nvsWidth of each frame in output batched buffer. This property MUST be set.

    + height: Height of each frame in output batched buffer. This property MUST be set.
    """

    args = args or {}
    streammux = Gst.ElementFactory.make("nvstreammux", f"nvstreammux-{index:02d}")
    if not streammux:
        raise RuntimeError(" Unable to create nvstreammux ")

    gpu_id = args.get("gpu_id", None)
    if gpu_id:
        streammux.set_property("gpu_id", gpu_id)

    batch_size = args.get("batch_size", None)
    if batch_size:
        streammux.set_property("batch-size", batch_size)

    batched_push_timeout = args.get("batched_push_timeout", None)
    if batched_push_timeout:
        streammux.set_property("batched-push-timeout", batched_push_timeout)

    live_source = args.get("live_source", None)
    if live_source is not None:
        streammux.set_property("live-source", bool(live_source))

    width = args.get("width", None)
    if width:
        streammux.set_property("width", width)

    height = args.get("height", None)
    if height:
        streammux.set_property("height", height)

    return streammux


def acquire_primary_nvinfer(index: int = 0, args: dict = None):
    """
    Create primary nvinfer instance to perform inference.
    All properties will use profile properties if they are not set.

    Pad Templates:
    + SINK template: 'sink'
        - Availability: Always
        - Capabilities:
            video/x-raw(memory:NVMM)
                    format: { (string)NV12, (string)RGBA }
                        width: [ 1, 2147483647 ]
                    height: [ 1, 2147483647 ]
                    framerate: [ 0/1, 2147483647/1 ]

    + SRC template: 'src'
        - Availability: Always
        - Capabilities:
            video/x-raw(memory:NVMM)
                    format: { (string)NV12, (string)RGBA }
                        width: [ 1, 2147483647 ]
                    height: [ 1, 2147483647 ]
                    framerate: [ 0/1, 2147483647/1 ]

    args:
    + gpu_id: Set GPU Device ID.
    + batch_size: Maximum batch size for inference.
    + config_file_path: Path to the TensorRT engine configuration file.
    + interval: Specifies number of consecutive batches to be skipped for inference. Default: 0-infer on erery frame
    + model_engine_file: Absolute path to the pre-generated serialized engine file for the model. default None

    Config priority (NVIDIA doc: Gst-nvinfer):
        GObject/set_property() OVERRIDE config file. Same key in both → code wins.
        So: config file is applied first (when pipeline goes READY or when config-file-path set),
        then any property set via set_property() overrides the file value.

    Return:
    + Gst.Element: primary nvinfer instance
    """
    args = args or {}
    pgie = Gst.ElementFactory.make("nvinfer", f"primary-inference-{index:02d}")
    if not pgie:
        raise RuntimeError(" Unable to create primary nvinfer ")

    gpu_id = args.get("gpu_id", None)
    if gpu_id:
        pgie.set_property("gpu_id", gpu_id)

    batch_size = args.get("batch_size", None)
    if batch_size:
        pgie.set_property("batch-size", batch_size)

    config_file_path = args.get("config_file_path", None)
    if config_file_path:
        pgie.set_property("config-file-path", config_file_path)

    model_engine_file = args.get("model_engine_file", None)
    if model_engine_file:
        pgie.set_property("model-engine-file", model_engine_file)

    interval = args.get("interval", None)
    if interval is not None:
        # interval may be 0, so use (interval is not None) to check
        pgie.set_property("interval", int(interval))

    return pgie


def acquire_nvvideoconvert(index: int = 0, args: dict = None):
    """
    Create nvvideoconvert instance to convert video frame from one colorspace(NV12) to another(RGBA).

    Args:
    + None

    Return:
    + None
    """
    args = args or {}
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", f"nvvideoconvert-{index:02d}")
    if not nvvidconv:
        raise RuntimeError(" Unable to create nvvideoconvert ")
    return nvvidconv


def acquire_nvdsosd(index: int = 0, args: dict = None):
    """
    Create OSD to draw on the converted RGBA buffer

    Args:
    + None

    Return:
    + None
    """
    args = args or {}
    nvosd = Gst.ElementFactory.make("nvdsosd", f"nvdsosd-{index:02d}")
    if not nvosd:
        raise RuntimeError(" Unable to create nvdsosd ")
    return nvosd
