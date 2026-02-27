/**
 * FER+ emotion-ferplus-8 情绪识别 secondary nvinfer 自定义分类解析.
 *
 * 模型单次推理输出 float32[1,8]。DeepStream 行为有两种可能：
 * 1) 每张脸推理一次并调用解析器一次，每次传入 [1,8] -> 只推送 1 个 attribute；
 * 2) 多张脸推理结果在一次调用里传入，buffer 为 N*8（即 [N,8] 展平）-> 按 N 个人分别读 8 个值，推送
 * N 个 attribute。 本解析器按 buffer 总元素数 numElements 判断：若为 8 的倍数则
 * batchSize=numElements/8，按 batch 读并推送 batchSize 个 attribute。
 *
 * 配置要点：
 * - is-classifier=1, process-mode=2, operate-on-gie-id=<PGIE 的 gie-unique-id>
 * - parse-classifier-func-name=NvDsInferParseCustomEmotion
 * - custom-lib-path=.../libnvdsinfer_custom_emotion.so
 * - labelfile-path 指向 dsapp_emotion_labels.txt（与下方顺序一致）
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include "nvdsinfer_custom_impl.h"

static const char* EMOTION_LABELS[] = {"neutral", "happiness", "surprise", "sadness",
                                       "anger",   "disgust",   "fear",     "contempt"};

static const int NUM_EMOTION_CLASSES = 8;

static float getFloat(const void* buf, int idx, NvDsInferDataType dtype) {
    if (dtype == FLOAT) return ((const float*)buf)[idx];
    if (dtype == HALF) {
        const uint16_t* h = (const uint16_t*)buf;
        uint16_t v = h[idx];
        int sign = (v >> 15) & 1;
        int expo = (v >> 10) & 0x1f;
        int mant = v & 0x3ff;
        if (expo == 0)
            return mant ? (sign ? -1.f : 1.f) * (mant / 1024.f) * (1.f / 16384.f)
                        : (sign ? -0.f : 0.f);
        if (expo == 31) return mant ? 0.f : (sign ? -INFINITY : INFINITY);
        return (sign ? -1.f : 1.f) * (1.f + mant / 1024.f) * std::ldexp(1.f, expo - 15);
    }
    return 0.f;
}

/** 对 logits 做 softmax, 结果写入 out (需至少 num 个 float) */
static void softmax(const float* logits, int num, float* out) {
    float max_logit = logits[0];
    for (int i = 1; i < num; i++)
        if (logits[i] > max_logit) max_logit = logits[i];
    float sum = 0.f;
    for (int i = 0; i < num; i++) {
        out[i] = std::exp(logits[i] - max_logit);
        sum += out[i];
    }
    for (int i = 0; i < num; i++) out[i] /= sum;
}

extern "C" bool NvDsInferParseCustomEmotion(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
                                            NvDsInferNetworkInfo const& networkInfo,
                                            float classifierThreshold,
                                            std::vector<NvDsInferAttribute>& attrList,
                                            std::string& descString) {
    (void)networkInfo;

    if (outputLayersInfo.empty() || !outputLayersInfo[0].buffer) return true;

    const NvDsInferLayerInfo& layer = outputLayersInfo[0];
    const NvDsInferDims& d = layer.inferDims;

    const int numClasses = NUM_EMOTION_CLASSES;
    int numElements = 1;
    for (unsigned int k = 0; k < d.numDims; k++) numElements *= (int)d.d[k];
    if (numElements < numClasses) return true;

    // 若 buffer 为 N*8（多张脸一次传入），则 batchSize=N；否则为单张脸 batchSize=1
    int batchSize = numElements / numClasses;
    if (batchSize <= 0 || batchSize > 1024) batchSize = 1;

    static bool debug_once = true;
    if (debug_once) {
        debug_once = false;
        std::fprintf(stderr, "\n[emotion_parse] numElements=%d batchSize=%d dims=(%d,%d)\n\n",
                     numElements, batchSize, d.numDims >= 1 ? (int)d.d[0] : 0,
                     d.numDims >= 2 ? (int)d.d[1] : 0);
    }

    float logits[8];
    float probs[8];
    descString.clear();

    for (int b = 0; b < batchSize; b++) {
        const int offset = b * numClasses;
        for (int i = 0; i < numClasses; i++)
            logits[i] = getFloat(layer.buffer, offset + i, layer.dataType);
        softmax(logits, numClasses, probs);

        int best = 0;
        for (int i = 1; i < numClasses; i++)
            if (probs[i] > probs[best]) best = i;
        float confidence = probs[best];
        if (confidence < classifierThreshold) confidence = classifierThreshold;

        NvDsInferAttribute attr;
        attr.attributeIndex = best;
        attr.attributeValue = 0;
        attr.attributeConfidence = confidence;
        attr.attributeLabel = strdup(best < NUM_EMOTION_CLASSES ? EMOTION_LABELS[best] : "unknown");
        attrList.push_back(attr);
        if (!descString.empty()) descString += "; ";
        descString += attr.attributeLabel;
    }

    return true;
}

CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomEmotion);
