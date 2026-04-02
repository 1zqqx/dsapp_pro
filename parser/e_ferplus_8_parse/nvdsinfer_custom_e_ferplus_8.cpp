/**
 * FERPlus(emotion-ferplus-8.onnx)在 DeepStream Secondary nvinfer(SGIE)上的自定义分类解析库.
 *
 * 行为约定: 假定每次回调对应「一个 ROI / 一张脸」的分类结果,只读取输出 buffer 的前 8 个 logit,
 * 经 softmax 后选出置信度最高的情绪类别,可选地写入 attrList(低于阈值则不写入).
 *
 * nvinfer 中的调用时机(何时被调):
 * - TensorRT 完成该次 SGIE 前向推理/输出层数据已填入 NvDsInferLayerInfo 之后;
 * - nvinfer 插件在把原始 tensor 转成下游可用的分类属性之前,按配置里的 parse-classifier-func-name
 *   调用本符号(与 CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE 校验的签名一致).
 *
 * 执行过程中实际调用次数:
 * - 等于「该 SGIE 完成推理并需要解析分类输出的次数」,通常每完成一次 forward 调用解析函数 1 次.
 * - 若 pipeline 对每张人脸单独 enqueue(batch=1),则大致为: 人脸数 × 相关帧上的推理次数.
 * - 若一次推理输出包含多张脸展平在同一 buffer(N×8),本实现仍只解析前 8 个元素,仅产生一个属性;
 *   多脸 batch 需改为按 N 循环(可参考同仓库 parser/emotion_parse/nvdsinfer_custom_emotion.cpp).
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "nvdsinfer_custom_impl.h"

static const int NUM_EMOTION_CLASSES = 8;
static const char* EMOTION_LABELS[NUM_EMOTION_CLASSES] = {
    "neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt",
};

/** 从原始输出 buffer 中按线性下标 idx 读取一个标量,并按 dataType 转为 float(支持 FLOAT / HALF). */
static float getFloat(const void* buf, int idx, NvDsInferDataType dtype) {
    if (dtype == FLOAT) return ((const float*)buf)[idx];
    if (dtype == HALF) {
        const uint16_t* h = (const uint16_t*)buf;
        uint16_t v = h[idx];
        int sign = (v >> 15) & 1;
        int expo = (v >> 10) & 0x1f;
        int mant = v & 0x3ff;
        if (expo == 0) {
            return mant ? (sign ? -1.f : 1.f) * (mant / 1024.f) * (1.f / 16384.f)
                        : (sign ? -0.f : 0.f);
        }
        if (expo == 31) return mant ? 0.f : (sign ? -INFINITY : INFINITY);
        return (sign ? -1.f : 1.f) * (1.f + mant / 1024.f) * std::ldexp(1.f, expo - 15);
    }
    return 0.f;
}

/** 对 logits[0..num-1] 做数值稳定的 softmax,概率写入 out(长度至少 num).sum<=0 时退化为均匀分布. */
static void softmax(const float* logits, int num, float* out) {
    float max_logit = logits[0];
    for (int i = 1; i < num; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    float sum = 0.f;
    for (int i = 0; i < num; i++) {
        out[i] = std::exp(logits[i] - max_logit);
        sum += out[i];
    }
    if (sum <= 0.f) {
        const float uniform = 1.0f / static_cast<float>(num);
        for (int i = 0; i < num; i++) out[i] = uniform;
        return;
    }
    for (int i = 0; i < num; i++) out[i] /= sum;
}

/**
 * SGIE 分类输出的自定义解析入口(由 nvinfer 在每次需要解析分类结果时调用).
 *
 * 参数说明:
 * - @param outputLayersInfo: 本次推理各输出层的元数据与 device/host 上的 buffer 指针; 本实现只使用
 * [0], 从中读取 8 类情绪的 logit.
 * - @param networkInfo: 网络输入尺寸等信息; 本解析器未使用(保留为 nvinfer 统一签名).
 * - @param classifierThreshold: 分类置信度阈值; softmax 后最大概率低于该值则不向 attrList 追加属性.
 * - @param attrList: 输出参数.成功且过阈值时 push_back 一个 NvDsInferAttribute(attributeIndex /
 *   attributeValue 均为胜出类别下标,attributeConfidence 为 softmax 概率,attributeLabel 为
 *   EMOTION_LABELS 中对应 C 字符串,由 strdup 分配,由框架/调用约定负责释放).
 * - @param descString: 输出参数.过阈值时设为与胜出标签相同的简短描述; 否则清空.
 *
 * 返回值:
 * - bool: true 表示解析流程结束且不应让 nvinfer 视为"解析失败"; 即使无输出或数据不合法也返回
 * true, 与仓库内其它自定义 parser 一致,避免插件报错中断 pipeline.
 *
 * 最终 输出数据 形状: 最多 1 条分类属性(8 类之一 + 置信度 + 标签字符串),挂在 attrList 上供
 * nvdsanalytics / 应用层通过 NvDsClassifierMeta 等路径消费.
 */
extern "C" bool NvDsInferParseCustomEFerplus8(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, float classifierThreshold,
    std::vector<NvDsInferAttribute>& attrList, std::string& descString) {
    (void)networkInfo;
    descString.clear();

    if (outputLayersInfo.empty() || !outputLayersInfo[0].buffer) return true;

    const NvDsInferLayerInfo& layer = outputLayersInfo[0];
    const NvDsInferDims& dims = layer.inferDims;
    int numElements = 1;
    for (unsigned int k = 0; k < dims.numDims; k++) numElements *= (int)dims.d[k];

    if (numElements < NUM_EMOTION_CLASSES) return true;

    float logits[NUM_EMOTION_CLASSES];
    float probs[NUM_EMOTION_CLASSES];
    for (int i = 0; i < NUM_EMOTION_CLASSES; i++) {
        logits[i] = getFloat(layer.buffer, i, layer.dataType);
    }
    softmax(logits, NUM_EMOTION_CLASSES, probs);

    int best = 0;
    for (int i = 1; i < NUM_EMOTION_CLASSES; i++) {
        if (probs[i] > probs[best]) best = i;
    }
    const float confidence = probs[best];

    if (confidence < classifierThreshold) return true;

    NvDsInferAttribute attr;
    std::memset(&attr, 0, sizeof(attr));
    attr.attributeIndex = best;
    attr.attributeValue = best;
    attr.attributeConfidence = confidence;
    attr.attributeLabel = strdup(EMOTION_LABELS[best]);
    attrList.push_back(attr);
    descString = attr.attributeLabel;
    return true;
}

/* 编译期断言: 本函数签名与 nvdsinfer 要求的自定义分类解析函数原型一致,否则编译失败. */
CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomEFerplus8);
