import os
import gi
import pyds
import logging

gi.require_version("Gst", "1.0")
# gi.require_version("GstRtspServer", "1.0")

from gi.repository import Gst, GLib  # type: ignore

logger = logging.getLogger(__name__)


def acquire_v4l2_source_bin(index: int = 0, device: str = "0", args=None):
    """
    创建基于 v4l2src 的 raw 格式 USB 摄像头 source bin。
    v4l2src -> [capsfilter] -> videoconvert -> nvvideoconvert -> capsfilter(NVMM)

    out:
    + video/x-raw(memory:NVMM)，可与 nvstreammux 直接连接。

    args :
    + device: cover device
    + gpu_id: nvvideoconvert 的 gpu-id
    + caps_v4l2: v4l2 端 capsfilter 的 caps, default "video/x-raw,format=YUY2,framerate=30/1"
    + caps_sinksrc: nvvideoconvert 后 capsfilter 的 caps, default "video/x-raw(memory:NVMM),format=NV12"
    + framerate: caps_v4l2 prop "30/1" 超过摄像头支持最大速率报错
    """
    args = args or {}
    bin_name = f"v4l2-src-bin-{index:02}"

    # 1. create
    cre_bin = Gst.Bin.new(bin_name)
    if not cre_bin:
        raise RuntimeError(f" Unable to create bin {bin_name} ")

    # 2. source
    source = Gst.ElementFactory.make("v4l2src", "v4l2-source")
    if not source:
        raise RuntimeError(" Unable to create source bin v4l2src ")
    _device = args.get("device", device)
    source.set_property("device", _device)

    # 3. capsfilter 约束 v4l2src 下游协商结果，与 deepstream-test1-usbcam 完全一致
    caps_v4l2 = Gst.ElementFactory.make("capsfilter", "v4l2src_caps")
    if not caps_v4l2:
        raise RuntimeError(" Unable to create source bin v4l2src capsfilter ")
    _fr = args.get("framerate", "30/1")
    _c = Gst.Caps.from_string(args.get("caps_v4l2", f"video/x-raw, framerate={_fr}"))
    if _c is None:
        raise RuntimeError(" Gst.Caps.from_string returned NULL ")
    caps_v4l2.set_property("caps", _c)

    # 4.
    vidconvsrc = Gst.ElementFactory.make("videoconvert", "convertor_src1")
    if not vidconvsrc:
        raise RuntimeError(" Unable to create source bin videoconvert ")

    # 5. cpu mem -> gpu mem
    nvvidconvsrc = Gst.ElementFactory.make("nvvideoconvert", "convertor_src2")
    if not nvvidconvsrc:
        raise RuntimeError(" Unable to create source bin nvvideoconvert ")
    if "gpu_id" in args and args["gpu_id"] is not None:
        nvvidconvsrc.set_property("gpu-id", int(args["gpu_id"]))

    # 6. NVMM capsfilter, 与 deepstream-test1-usbcam 完全一致
    caps_sinksrc = Gst.ElementFactory.make("capsfilter", "nvmm_caps")
    if not caps_sinksrc:
        raise RuntimeError(" Unable to create source bin caps_sinksrc ")
    _c_nvmm = Gst.Caps.from_string(args.get("caps_sinksrc", "video/x-raw(memory:NVMM)"))
    if _c_nvmm is None:
        raise RuntimeError(" Gst.Caps.from_string(NVMM) returned NULL ")
    caps_sinksrc.set_property("caps", _c_nvmm)

    cre_bin.add(source)
    cre_bin.add(caps_v4l2)
    cre_bin.add(vidconvsrc)
    cre_bin.add(nvvidconvsrc)
    cre_bin.add(caps_sinksrc)

    source.link(caps_v4l2)
    caps_v4l2.link(vidconvsrc)
    vidconvsrc.link(nvvidconvsrc)
    nvvidconvsrc.link(caps_sinksrc)

    ghost_pad = Gst.GhostPad.new("src", caps_sinksrc.get_static_pad("src"))
    if not cre_bin.add_pad(ghost_pad):
        raise RuntimeError(" Failed to add ghost pad in v4l2 source bin ")

    return cre_bin


# XXX handler func args (发出 pad-added 的 Element, 新增的 pad, user_data)
def _cb_newpad(decodebin, n_pad, data):
    # decode-bin ==> nvurisrcbin.n_pad
    caps = n_pad.get_current_caps()
    if not caps:
        caps = n_pad.query_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    if gstname.find("video") != -1:
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(n_pad):
                raise RuntimeError(
                    "Failed to link decoder src pad to source bin ghost pad\n"
                )
        else:
            raise RuntimeError(
                " Error: Decodebin did not pick nvidia decoder plugin.\n"
            )


def _decodebin_child_added(child_proxy, Object, name, user_data):
    if name.find("decodebin") != -1:
        Object.connect("child-added", _decodebin_child_added, user_data)

    if "source" in name:
        source_element = child_proxy.get_by_name("source")
        if source_element.find_property("drop-on-latency") != None:
            # 当解码器延迟时, 是否丢弃帧
            Object.set_property("drop-on-latency", True)


def acquire_nvurisrcbin(index: int = 0, uri: str = "0", args=None):
    """
    使用 nvurisrcbin 创建 source bin。适用于 file:// 和 rtsp://
    nvurisrcbin 内部使用 NVIDIA 解码链,可避免 uridecodebin + qtdemux 的 not-negotiated 问题

    out type:
    + video/x-raw(memory:NVMM)

    args:
    + uri: str = None, source uri
    + file-loop: bool, default False, Loop file sources after EOS.
    + cudadec-memtype: int = [0/1/2], default 0, Set to specify memory type for cuda decoder buffers, more info -> nvurisrcbin
    + latency: int, default 100, Latency in milliseconds to buffer data from source before pushing downstream. This is used to smooth out jitter in live sources such as RTSP streams. Higher latency will result in smoother playback, but increased time to display the first frame.

    Returns:
    + Gst.Bin: 带 ghost pad "src" 的 source bin
    """
    args = args or {}

    bin_name = f"nvuri-src-bin-{index:02}"
    cre_bin = Gst.Bin.new(bin_name)
    if not cre_bin:
        raise RuntimeError(f" Unable to create source bin {bin_name} ")

    source = Gst.ElementFactory.make("nvurisrcbin", f"nvurisrcbin-{index:02}")
    if not source:
        raise RuntimeError(f" Unable to create nvurisrcbin-{index:02} ")

    _fl = int(args.get("file-loop", 0))
    source.set_property("file-loop", _fl)
    _cmt = int(args.get("cudadec-memtype", 0))
    source.set_property("cudadec-memtype", _cmt)
    _uri = args.get("uri", uri)
    source.set_property("uri", _uri)
    _lat = int(args.get("latency", 100))
    source.set_property("latency", _lat)

    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    source.connect("pad-added", _cb_newpad, cre_bin)
    source.connect("child-added", _decodebin_child_added, cre_bin)

    # _identity = Gst.ElementFactory.make("identity", f"identity-{index:02}")
    # # True 则按帧率显示, 按照管道时钟, False 则尽快显示
    # _identity.set_property("sync", True)

    Gst.Bin.add(cre_bin, source)
    # Gst.Bin.add(cre_bin, _identity)
    # source.link(_identity)
    ghost_pad = cre_bin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not ghost_pad:
        raise RuntimeError(f" Failed to add ghost pad in {bin_name} ")

    return cre_bin


def acquire_filesrc_h264_bin(index: int = 0, path: str = "", args=None):
    """
    使用显式 filesrc + qtdemux/h264parse + nvv4l2decoder 创建 source bin
    用于绕过 nvurisrcbin 对部分 MP4 的 segment 问题 或支持裸 H.264 流

    支持:
    + MP4/MOV/M4V: filesrc -> qtdemux -(pad-added)-> h264parse -> nvv4l2decoder
    + 裸 H.264 (.h264/.264): filesrc -> h264parse -> nvv4l2decoder

    args:
    + path: str, 文件路径 (非 file:// URI)

    out:
    + video/x-raw(memory:NVMM)
    """
    args = args or {}
    _path = args.get("path", path)
    if not _path:
        raise ValueError("acquire_filesrc_h264_bin requires path")

    use_demux = _path.lower().endswith((".mp4", ".mov", ".m4v"))
    bin_name = f"filesrc-h264-bin-{index:02}"

    def _demux_pad_added_cb(demux_element, pad, h264parser):
        name = pad.get_name()
        if name.startswith("video/x-h264"):
            sinkpad = h264parser.get_static_pad("sink")
            if not sinkpad.is_linked():
                pad.link(sinkpad)

    cre_bin = Gst.Bin.new(bin_name)
    if not cre_bin:
        raise RuntimeError(f"Unable to create source bin {bin_name}")

    source = Gst.ElementFactory.make("filesrc", f"filesrc-{index:02}")
    if not source:
        raise RuntimeError("Unable to create filesrc")
    source.set_property("location", _path)

    h264parser = Gst.ElementFactory.make("h264parse", f"h264parse-{index:02}")
    if not h264parser:
        raise RuntimeError("Unable to create h264parse")

    decoder = Gst.ElementFactory.make("nvv4l2decoder", f"nvv4l2decoder-{index:02}")
    if not decoder:
        raise RuntimeError("Unable to create nvv4l2decoder")

    cre_bin.add(source)
    cre_bin.add(h264parser)
    cre_bin.add(decoder)

    if use_demux:
        demux = Gst.ElementFactory.make("qtdemux", f"qtdemux-{index:02}")
        if not demux:
            raise RuntimeError("Unable to create qtdemux")
        cre_bin.add(demux)
        source.link(demux)
        demux.connect("pad-added", _demux_pad_added_cb, h264parser)
    else:
        source.link(h264parser)

    h264parser.link(decoder)

    ghost_pad = Gst.GhostPad.new("src", decoder.get_static_pad("src"))
    if not cre_bin.add_pad(ghost_pad):
        raise RuntimeError(f"Failed to add ghost pad in {bin_name}")

    return cre_bin


def acquire_uridecodebin(index, uri, args=None):
    """
    创建 uridecodebin 的 source bin
    当 decodebin 选用 nvdec_* 时直接输出 NVMM; 选用软件解码器时,
    通过 videoconvert -> nvvideoconvert 转为 NVMM, 保证输出 video/x-raw(memory:NVMM)

    Returns:
    + Gst.Bin: 带 ghost pad "src" 的 source bin
    """
    args = args or {}
    _ts_from_rtsp: bool = None

    def _cb_newpad(decodebin, decoder_src_pad, data):
        caps = decoder_src_pad.get_current_caps()
        if caps is None:
            return
        gststruct = caps.get_structure(0)
        if gststruct is None:
            return
        gstname = gststruct.get_name()
        source_bin = data
        features = caps.get_features(0)

        if gstname.find("video") == -1:
            return

        bin_ghost_pad = source_bin.get_static_pad("src")
        if features.contains("memory:NVMM"):
            # 硬件解码(nvdec_*), 直接接到 ghost pad
            if not bin_ghost_pad.set_target(decoder_src_pad):
                raise RuntimeError(
                    "Failed to link decoder src pad to source bin ghost pad"
                )
        else:
            # 软件解码(avdec_* 等) 输出系统内存, 需转为 NVMM
            queue = Gst.ElementFactory.make("queue", "swdec-queue")
            vidconv = Gst.ElementFactory.make("videoconvert", "swdec-vidconv")
            nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "swdec-nvconv")
            caps_nvmm = Gst.ElementFactory.make("capsfilter", "swdec-nvmm-caps")
            if not all([queue, vidconv, nvvidconv, caps_nvmm]):
                raise RuntimeError(
                    "Unable to create sw-dec conversion chain (queue/videoconvert/nvvideoconvert/capsfilter)"
                )
            caps_nvmm.set_property(
                "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM)")
            )
            for el in (queue, vidconv, nvvidconv, caps_nvmm):
                source_bin.add(el)

            queue.link(vidconv)
            vidconv.link(nvvidconv)
            nvvidconv.link(caps_nvmm)

            if (
                decoder_src_pad.link(queue.get_static_pad("sink"))
                != Gst.PadLinkReturn.OK
            ):
                raise RuntimeError(
                    "Failed to link decoder src pad to sw-dec conversion chain"
                )
            if not bin_ghost_pad.set_target(caps_nvmm.get_static_pad("src")):
                raise RuntimeError(
                    "Failed to link sw-dec nvmm caps to source bin ghost pad"
                )

            for el in (queue, vidconv, nvvidconv, caps_nvmm):
                el.sync_state_with_parent()

    def _decodebin_child_added(child_proxy, Object, name, user_data):
        # print("Decodebin child added:", name, "\n")
        if name.find("decodebin") != -1:
            Object.connect("child-added", _decodebin_child_added, user_data)

        if _ts_from_rtsp:
            if name.find("source") != -1:
                pyds.configure_source_for_ntp_sync(hash(Object))

    _ts_from_rtsp = uri.startswith("rtsp://")

    bin_name = f"urisrc-bin-{index:02}"
    cre_bin = Gst.Bin.new(bin_name)
    if not cre_bin:
        raise RuntimeError(f" Unable to create source bin {bin_name} ")

    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        raise RuntimeError(" Unable to create uri decode bin \n")

    uri_decode_bin.set_property("uri", uri)
    if args:
        if "buffer_size" in args and args["buffer_size"] is not None:
            uri_decode_bin.set_property("buffer-size", args["buffer_size"])
        if "buffer_duration" in args and args["buffer_duration"] is not None:
            uri_decode_bin.set_property("buffer-duration", args["buffer_duration"])
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added", _cb_newpad, cre_bin)
    uri_decode_bin.connect("child-added", _decodebin_child_added, cre_bin)

    Gst.Bin.add(cre_bin, uri_decode_bin)
    bin_pad = cre_bin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        raise RuntimeError(" Failed to add ghost pad in source bin \n")

    return cre_bin
