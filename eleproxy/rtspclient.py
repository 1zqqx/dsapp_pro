"""
将 video/x-raw 编码为 H264/H265 并通过 UDP 推送, 可配合 GstRtspServer 提供 RTSP 服务.
抽离自 apps/deepstream-test1-rtsp-out/deepstream_test1_rtsp_out.py 的编码与 RTSP 推流部分.
"""

import sys
from typing import Optional, Tuple, Any

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GstRtspServer  # type: ignore

try:
    from common.platform_info import PlatformInfo
except Exception:
    PlatformInfo = None


def get_rtsp_push_bin(
    args: Optional[dict] = None,
) -> Tuple[Gst.Bin, Optional[Any]]:
    """
    创建编码并推送 RTSP 视频流的 Gst.Bin, 以及可选的 RTSP 服务端.

    管道逻辑: sink(外部连接)-> capsfilter -> encoder -> enc_parse -> rtppay -> udpsink.
    返回的 bin 带有一个 ghost sink pad, 用于连接上游的 video/x-raw 输出(如 nvvidconv 的 src).

    args :
    + codec: str, "H264" | "H265", 默认 "H264"
    + bitrate: int, 编码码率, 默认 4000000
    + enc_type: int, 0=硬件编码(nvv4l2), 1=软件编码(x264/x265), 默认 0
    + iframeinterval: int, I 帧间隔(帧数), 默认 15, 用于丢包后尽快恢复、减轻花屏
    + idrinterval: int, IDR 帧间隔(帧数), 默认 15
    + udp_host: str, 组播/单播地址, 默认 "224.224.255.255"
    + udp_port: int, UDP 端口, 默认 5400
    + jitterbuffer_latency_ms: int, RTSP 服务端 rtpjitterbuffer 的 latency(ms), 默认 1000, 减小 "max delay reached" 丢包与花屏
    + create_rtsp_server: bool, 是否创建并返回 RTSP 服务端, 默认 True
    + rtsp_port: int, RTSP 服务端口, 默认 8554
    + mount_path: str, RTSP 挂载路径, 默认 "/deepstream/1"
    + media_configure_cb: callable (factory, media) -> None, 可选；在媒体创建后调用，用于给 media 管道加 bus 监听等

    Returns:
      (bin, server): bin 为 Gst.Bin, 需加入 pipeline 并将上游元素连到 bin 的 sink;
                     server 为 GstRtspServer.RTSPServer 或 None.
                     若 create_rtsp_server 为 True, 调用方需在合适时机调用 server.attach(None).
    """

    args = args or {}
    codec = args.get("codec", "H264")
    bitrate = int(args.get("bitrate", 4000000))
    enc_type = int(args.get("enc_type", 0))
    udp_host = args.get("udp_host", "224.224.255.255")
    udp_port = int(args.get("udp_port", 5400))
    create_rtsp_server = args.get("create_rtsp_server", True)
    rtsp_port = int(args.get("rtsp_port", 8554))
    mount_path = args.get("mount_path", "/ds-test")
    media_configure_cb = args.get("media_configure_cb")

    # 1. capsfilter
    caps = Gst.ElementFactory.make("capsfilter", "rtsp-push-filter")
    if not caps:
        raise RuntimeError("Unable to create capsfilter")
    if enc_type == 0:
        caps.set_property(
            "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420")
        )
    else:
        caps.set_property("caps", Gst.Caps.from_string("video/x-raw, format=I420"))

    # 2. encoder
    if codec == "H264":
        encoder = (
            Gst.ElementFactory.make("nvv4l2h264enc", "rtsp-push-encoder")
            if enc_type == 0
            else Gst.ElementFactory.make("x264enc", "rtsp-push-encoder")
        )
    elif codec == "H265":
        encoder = (
            Gst.ElementFactory.make("nvv4l2h265enc", "rtsp-push-encoder")
            if enc_type == 0
            else Gst.ElementFactory.make("x265enc", "rtsp-push-encoder")
        )
    else:
        raise ValueError("codec must be H264 or H265")
    if not encoder:
        raise RuntimeError("Unable to create encoder")
    if enc_type == 0:
        encoder.set_property("insert-sps-pps", 1)
        # 关键帧间隔：RTP 丢包后解码器需等下一个 IDR 才能恢复，间隔过大会长时间花屏/下半屏绿块
        iframe_interval = int(args.get("iframeinterval", 15))
        idr_interval = int(args.get("idrinterval", 15))
        if codec == "H264":
            encoder.set_property("iframeinterval", iframe_interval)
            encoder.set_property("idrinterval", idr_interval)
        elif codec == "H265":
            encoder.set_property("iframeinterval", iframe_interval)
            encoder.set_property("idrinterval", idr_interval)
    encoder.set_property("bitrate", bitrate)
    if PlatformInfo is not None:
        try:
            platform_info = PlatformInfo()
            if platform_info.is_integrated_gpu() and enc_type == 0:
                encoder.set_property("preset-level", 1)
        except Exception:
            pass

    # 3. enc_parse (encoder 与 rtppay 之间, 确保 SPS/PPS 在码流中)
    enc_parse = (
        Gst.ElementFactory.make("h264parse", "rtsp-push-h264parse")
        if codec == "H264"
        else Gst.ElementFactory.make("h265parse", "rtsp-push-h265parse")
    )
    if not enc_parse:
        raise RuntimeError("Unable to create encoder output parse")
    enc_parse.set_property("config-interval", -1)

    # 4. rtppay
    rtppay = (
        Gst.ElementFactory.make("rtph264pay", "rtsp-push-rtppay")
        if codec == "H264"
        else Gst.ElementFactory.make("rtph265pay", "rtsp-push-rtppay")
    )
    if not rtppay:
        raise RuntimeError("Unable to create rtppay")
    rtppay.set_property("config-interval", 1)

    # 5. udpsink
    sink = Gst.ElementFactory.make("udpsink", "rtsp-push-udpsink")
    if not sink:
        raise RuntimeError("Unable to create udpsink")
    sink.set_property("host", udp_host)
    sink.set_property("port", udp_port)
    sink.set_property("async", False)
    sink.set_property("sync", 1)

    # 6. bin 组装与链接
    bin_ = Gst.Bin.new("rtsp-push-bin")
    bin_.add(caps)
    bin_.add(encoder)
    bin_.add(enc_parse)
    bin_.add(rtppay)
    bin_.add(sink)
    caps.link(encoder)
    encoder.link(enc_parse)
    enc_parse.link(rtppay)
    rtppay.link(sink)

    # ghost pad: 将 capsfilter 的 sink 暴露为 bin 的 sink
    sink_pad = caps.get_static_pad("sink")
    if not sink_pad:
        raise RuntimeError("Unable to get sink pad of capsfilter")
    ghost_sink = Gst.GhostPad.new("sink", sink_pad)
    bin_.add_pad(ghost_sink)

    # 7. 可选: RTSP 服务端(从同一 UDP 端口读 RTP, 以 RTSP 提供给客户端)
    # GstRtspServer 会对名为 pay0 的元素设置 pt (payload type)，故 pay0 必须是 rtph264pay/rtph265pay，不能是 udpsrc
    # udpsrc 需 address=udp_host 与推流目标一致，且 timeout=0 无限等首包，否则 DESCRIBE 时管道未就绪会 503
    server = None
    if create_rtsp_server:
        server = GstRtspServer.RTSPServer.new()
        server.props.service = f"{rtsp_port}"
        factory = GstRtspServer.RTSPMediaFactory.new()
        # address: 与 udpsink 的 host 一致；timeout=0 无限等首包；reuse=True 允许端口复用，避免 bind 失败
        udp_bind_addr = udp_host if udp_host in ("127.0.0.1", "0.0.0.0") else "0.0.0.0"
        # rtpjitterbuffer latency(ms)：加大可减少“max delay reached”导致的丢包，避免客户端报 RTP missed / 解码错误（卡顿、下半屏花屏）
        jitter_latency_ms = int(args.get("jitterbuffer_latency_ms", 1000))
        jitter_str = f"rtpjitterbuffer latency={jitter_latency_ms}"
        if codec == "H264":
            launch = (
                '( udpsrc address=%s port=%d timeout=0 reuse=true buffer-size=524288 caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)H264" '
                "! %s ! rtph264depay ! rtph264pay name=pay0 pt=96 )"
            ) % (udp_bind_addr, udp_port, jitter_str)
        else:
            launch = (
                '( udpsrc address=%s port=%d timeout=0 reuse=true buffer-size=524288 caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)H265" '
                "! %s ! rtph265depay ! rtph265pay name=pay0 pt=96 )"
            ) % (udp_bind_addr, udp_port, jitter_str)
        factory.set_launch(launch)
        factory.set_shared(True)
        try:
            factory.set_latency(2000)
        except Exception:
            pass

        if callable(media_configure_cb):
            try:
                factory.connect("media-configure", media_configure_cb)
            except Exception:
                pass
        server.get_mount_points().add_factory(mount_path, factory)

    return bin_, server


def get_rtsp_client_bin(
    args: Optional[dict] = None,
) -> Gst.Bin:
    """
    创建通过 rtspclientsink 推流到 mediaMTX 的 Gst.Bin.

    管道逻辑: sink(ghost) -> nvvideoconvert -> capsfilter(I420) -> nvv4l2h265enc
              -> h265parse -> rtspclientsink

    args :
    + insert-sps-pps: int, default -1
    + bitrate: int, 编码码率, 默认 4000000
    + iframeinterval: int, I 帧间隔, 默认 15
    + idrinterval: int, IDR 帧间隔, 默认 15
    + num_B_frames: int, B 帧数量, 默认 0
    + profile: int, H265 编码 profile, 默认 0
    + mediamtx_url: str, 推流目标地址, 默认 "rtsp://127.0.0.1:8554/stream/10086"
    + rtsp_protocols: int | None, rtspclientsink 使用的传输协议, 默认 None (不设置) 4=TCP(可靠易卡顿) 1=UDP(实时易花屏)
    + caps: str, default: `video/x-raw(memory:NVMM), format=I420`
    + config-interval: int, default -1

    Returns:
      Gst.Bin, 带有一个 ghost sink pad, 用于连接上游输出.
    """
    args = args or {}

    # 1. nvvideoconvert
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "rtsp-client-convert")
    if not nvvidconv:
        raise RuntimeError("Unable to create nvvideoconvert for rtspclientsink bin")

    # 2. capsfilter
    capsfilter = Gst.ElementFactory.make("capsfilter", "rtsp-client-caps")
    if not capsfilter:
        raise RuntimeError("Unable to create capsfilter for rtspclientsink bin")
    _caps = args.get("caps", "video/x-raw(memory:NVMM), format=I420")
    capsfilter.set_property("caps", Gst.Caps.from_string(_caps))

    # 3. H265 encoder
    encoder = Gst.ElementFactory.make("nvv4l2h265enc", "rtsp-client-encoder")
    if not encoder:
        raise RuntimeError("Unable to create nvv4l2h265enc")

    _a = args.get("insert-sps-pps", -1)
    if _a is not None:
        encoder.set_property("insert-sps-pps", _a)
    _a = args.get("profile", 0)
    if _a is not None:
        encoder.set_property("profile", _a)
    _a = args.get("bitrate", 8000000)
    if _a is not None:
        encoder.set_property("bitrate", _a)
    _a = args.get("iframeinterval", 15)
    if _a is not None:
        encoder.set_property("iframeinterval", _a)
    _a = args.get("idrinterval", 15)
    if _a is not None:
        encoder.set_property("idrinterval", _a)
    _a = args.get("num-B-Frames", 0)
    if _a is not None:
        encoder.set_property("num-B-Frames", _a)

    # 4. h265parse
    h265parse = Gst.ElementFactory.make("h265parse", "rtsp-client-h265parse")
    if not h265parse:
        raise RuntimeError("Unable to create h265parse")
    _a = args.get("config-interval", -1)
    if _a is not None:
        h265parse.set_property("config-interval", _a)

    # 5. rtspclientsink
    clientsink = Gst.ElementFactory.make("rtspclientsink", "rtsp-client-sink")
    if not clientsink:
        raise RuntimeError(
            "Unable to create rtspclientsink (需安装 gst-rtsp-server 插件)"
        )
    mediamtx_url = args.get("mediamtx_url", "rtsp://127.0.0.1:8554/stream/10086")
    clientsink.set_property("location", mediamtx_url)
    _a = args.get("rtsp_protocols", None)
    if _a is not None:
        clientsink.set_property("protocols", _a)

    # 6. bin 组装与链接
    bin_ = Gst.Bin.new("rtsp-client-bin")
    for ele in (nvvidconv, capsfilter, encoder, h265parse, clientsink):
        bin_.add(ele)
    nvvidconv.link(capsfilter)
    capsfilter.link(encoder)
    encoder.link(h265parse)
    h265parse.link(clientsink)

    # ghost pad: 将 nvvideoconvert 的 sink 暴露为 bin 的 sink
    sink_pad = nvvidconv.get_static_pad("sink")
    if not sink_pad:
        raise RuntimeError("Unable to get sink pad of nvvideoconvert")
    ghost_sink = Gst.GhostPad.new("sink", sink_pad)
    bin_.add_pad(ghost_sink)

    return bin_
