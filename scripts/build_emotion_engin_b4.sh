#!/bin/bash
# 用 trtexec 预构建 emotion-ferplus-8 的 TensorRT engine（batch=4），
# 避免 nvinfer 自建时误用 batch=1 生成 _b1_.engine。
# 运行前请先执行 emotion_onnx_dynamic_batch.py 生成 dynamic_batch.onnx。
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$(cd "$SCRIPT_DIR/../models.temp" && pwd)"
ONNX="$MODELS_DIR/emotion-ferplus-8_dynamic_batch.onnx"
ENGINE="$MODELS_DIR/emotion-ferplus-8_dynamic_batch.onnx_b4_gpu0_fp16.engine"

if [[ ! -f "$ONNX" ]]; then
  echo "缺少 $ONNX，请先运行: python3 $SCRIPT_DIR/emotion_onnx_dynamic_batch.py"
  exit 1
fi

# 从 ONNX 取第一个输入名（常见为 Input3 / input 等）
INPUT_NAME=$(python3 -c "
import onnx
m = onnx.load('$ONNX')
print(m.graph.input[0].name)
")

echo "INPUT_NAME: ${INPUT_NAME}" # Input3

# 静态 batch=4：模型内部 Reshape 按 batch=1 写死，无法做真正的动态 batch，
# 故只建 batch=4 的 engine，与 nvinfer batch-size=4 一致。
/usr/src/tensorrt/bin/trtexec --onnx="$ONNX" \
  --saveEngine="$ENGINE" \
  --fp16 \
  --shapes="${INPUT_NAME}:4x1x64x64" \
  --memPoolSize=workspace:64

echo "已生成: $ENGINE"
echo "请确保 dsapp_sgie_emotion_config.txt 中 model-engine-file 指向此文件。"
