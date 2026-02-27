#!/usr/bin/env python3
"""
把 emotion-ferplus-8.onnx 的 batch 维改为动态，以便 nvinfer 能建 batch>1 的 engine，
多人脸时一次得到 (N,8) 输出，解析器可推送 N 个 attribute。
同时修复图中 Reshape 的固定 shape [1,4096] 为 [-1,4096]，否则 TensorRT 在 batch>1 时报错。
用法: python3 emotion_onnx_dynamic_batch.py [输入.onnx] [输出.onnx]
默认: models.temp/emotion-ferplus-8.onnx -> models.temp/emotion-ferplus-8_dynamic_batch.onnx
"""
import os
import sys

import numpy as np

try:
    import onnx
    from onnx import numpy_helper
except ImportError:
    print("需要安装 onnx: pip install onnx", file=sys.stderr)
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "models.temp"))
DEFAULT_INPUT = os.path.join(MODELS_DIR, "emotion-ferplus-8.onnx")
DEFAULT_OUTPUT = os.path.join(MODELS_DIR, "emotion-ferplus-8_dynamic_batch.onnx")


def fix_reshape_constant_batch(model):
    """
    将图中 Reshape 使用的固定 shape [1, 4096] 改为 [-1, 4096]，使第一维随 batch 推断，
    否则 TensorRT 在 batch>1 时会报 reshape volume 冲突。
    """
    # 收集所有 Reshape 的 shape 输入名
    shape_input_names = set()
    for node in model.graph.node:
        if node.op_type == "Reshape" and len(node.input) >= 2:
            shape_input_names.add(node.input[1])

    # 替换 initializer 中值为 [1, 4096] 的 shape
    for i, init in enumerate(model.graph.initializer):
        if init.name not in shape_input_names:
            continue
        try:
            arr = numpy_helper.to_array(init)
        except Exception:
            continue
        if arr.ndim == 1 and arr.size == 2 and int(arr[0]) == 1 and int(arr[1]) == 4096:
            new_init = numpy_helper.from_array(
                np.array([-1, 4096], dtype=np.int64), init.name
            )
            model.graph.initializer[i].CopyFrom(new_init)
            return True

    # 若 shape 来自 Constant 节点，在图中查找并替换
    const_name_to_value = {}
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t:
                    try:
                        const_name_to_value[node.output[0]] = numpy_helper.to_array(
                            attr.t
                        )
                    except Exception:
                        pass

    for name in shape_input_names:
        if name not in const_name_to_value:
            continue
        arr = const_name_to_value[name]
        if arr.ndim == 1 and arr.size == 2 and int(arr[0]) == 1 and int(arr[1]) == 4096:
            # 用新 initializer 替代 Constant，并让 Reshape 引用该 initializer
            new_init = numpy_helper.from_array(
                np.array([-1, 4096], dtype=np.int64), name + "_batch_fix"
            )
            model.graph.initializer.append(new_init)
            for node in model.graph.node:
                if (
                    node.op_type == "Reshape"
                    and len(node.input) >= 2
                    and node.input[1] == name
                ):
                    node.input[1] = new_init.name
                    return True
    return False


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    output_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT

    if not os.path.isfile(input_path):
        print(f"输入文件不存在: {input_path}", file=sys.stderr)
        sys.exit(1)

    model = onnx.load(input_path)
    # 输入/输出第一维改为动态 (batch)
    batch_param = "batch"
    for inp in model.graph.input:
        if inp.type.tensor_type.shape.dim:
            d0 = inp.type.tensor_type.shape.dim[0]
            d0.dim_param = batch_param
            d0.ClearField("dim_value")
    for out in model.graph.output:
        if out.type.tensor_type.shape.dim:
            d0 = out.type.tensor_type.shape.dim[0]
            d0.dim_param = batch_param
            d0.ClearField("dim_value")

    if fix_reshape_constant_batch(model):
        print("已修复图中 Reshape shape [1,4096] -> [-1,4096]")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    onnx.save(model, output_path)

    print(f"已写出动态 batch 模型: {output_path}")


if __name__ == "__main__":
    main()
