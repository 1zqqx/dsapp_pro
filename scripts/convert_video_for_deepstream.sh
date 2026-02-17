#!/bin/bash
# 将视频转换为 DeepStream 兼容格式
# 解决 yuvj420p / full range / HDR / segment event 导致的错误
#
# 用法:
#   ./convert_video_for_deepstream.sh input.mp4 [output.mp4]   # 输出 MP4
#   ./convert_video_for_deepstream.sh input.mp4 output.h264    # 输出裸 H.264（绕过 MP4 容器，推荐）

INPUT="${1:?用法: $0 input.mp4 [output.mp4 或 output.h264]}"
OUTPUT="${2:-${INPUT%.*}_dscompat.mp4}"

echo "输入: $INPUT"
echo "输出: $OUTPUT"

# 1. format=yuv420p: 确保像素格式
# 2. h264_metadata: 强制写入 BT.709 limited range 到 H.264 流内（覆盖 bt2020/arib-std-b67）
# 3. avoid_negative_ts make_zero: 修复时间戳，避免 "Got data flow before segment event"
# 4. movflags +faststart: moov 放文件头（仅 MP4）

EXT="${OUTPUT##*.}"
# 编码时强制 BT.709 limited range，避免继承源文件的 HDR 元数据
ENC_OPTS=(-c:v libx264 -pix_fmt yuv420p -color_range tv -color_primaries bt709 -color_trc bt709 -colorspace bt709)
BSF="h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1:video_full_range_flag=0"

if [[ "${EXT,,}" == "h264" || "${EXT,,}" == "264" ]]; then
  # 裸 H.264：完全绕过 MP4 容器，避免 qtdemux segment 问题（推荐）
  ffmpeg -i "$INPUT" \
    -vf "format=yuv420p" \
    "${ENC_OPTS[@]}" -an \
    -bsf:v "$BSF" \
    -avoid_negative_ts make_zero \
    -y "$OUTPUT"
else
  ffmpeg -i "$INPUT" \
    -vf "format=yuv420p" \
    "${ENC_OPTS[@]}" \
    -bsf:v "$BSF" \
    -avoid_negative_ts make_zero \
    -movflags +faststart \
    -y "$OUTPUT"
fi

echo "转换完成: $OUTPUT"
echo "验证: ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,pix_fmt,color_space,color_range -of default=noprint_wrappers=1 $OUTPUT"
