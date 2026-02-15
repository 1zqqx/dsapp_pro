/**
 * ArcFace (e.g. r50/w600k) 人脸识别 secondary nvinfer 自定义分类解析.
 *
 * ArcFace 输出为 512 维 embedding，不是类别 softmax，因此本解析器：
 * - 仅做“通过”解析：保证 nvinfer 不报错，可选向 attrList 填一个占位属性；
 * - 实际识别（与底库比对）需在应用层通过 output-tensor-meta 取原始 embedding 做相似度计算.
 *
 * 配置要点：
 * - is-classifier=1, operate-on-roi=1, operate-on-gie-id=<PGIE的gie-unique-id>
 * - parse-classifier-func-name=NvDsInferParseCustomArcFace
 * - custom-lib-path=.../libnvdsinfer_custom_arcface_r50_wbface.so
 * - output-tensor-meta=1  （必须，否则应用层拿不到 embedding）
 */

#include "nvdsinfer_custom_impl.h"
#include <cmath>
#include <cstring>
#include <string>
#include <cstdint>

static float getFloat(const void *buf, int idx, NvDsInferDataType dtype)
{
    if (dtype == FLOAT)
        return ((const float *)buf)[idx];
    if (dtype == HALF)
    {
        const uint16_t *h = (const uint16_t *)buf;
        uint16_t v = h[idx];
        int sign = (v >> 15) & 1;
        int expo = (v >> 10) & 0x1f;
        int mant = v & 0x3ff;
        if (expo == 0)
            return mant ? (sign ? -1.f : 1.f) * (mant / 1024.f) * (1.f / 16384.f) : (sign ? -0.f : 0.f);
        if (expo == 31)
            return mant ? 0.f : (sign ? -INFINITY : INFINITY);
        return (sign ? -1.f : 1.f) * (1.f + mant / 1024.f) * std::ldexp(1.f, expo - 15);
    }
    return 0.f;
}

/**
 * 自定义分类解析入口.
 * ArcFace 单输出层形状 (1, 512)，此处只保证解析成功并可选写入一个占位属性；
 * 应用层通过 NvDsInferTensorMeta（output-tensor-meta=1）读取 512 维做底库匹配.
 */
extern "C" bool NvDsInferParseCustomArcFace(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    float classifierThreshold,
    std::vector<NvDsInferAttribute> &attrList,
    std::string &descString)
{
    if (outputLayersInfo.empty() || !outputLayersInfo[0].buffer)
        return true; /* 无输出也返回 true，避免 nvinfer 报错 */

    const NvDsInferLayerInfo &layer = outputLayersInfo[0];
    const NvDsInferDims &d = layer.inferDims;
    int numElements = (int)(d.numElements > 0 ? d.numElements : 1);
    for (unsigned int k = 0; k < d.numDims; k++)
        numElements = (int)(numElements * (int)d.d[k]);
    if (numElements <= 0)
        numElements = 512;

    /* 可选：写一个占位属性，便于 downstream 知道该对象已做过 ArcFace */
    NvDsInferAttribute attr;
    attr.attributeIndex = 0;
    attr.attributeValue = 0;
    attr.attributeConfidence = 1.0f;
    attr.attributeLabel = strdup("face_embedding");
    attrList.push_back(attr);
    descString = "face_embedding";

    // explicit neglect
    (void)classifierThreshold;
    (void)networkInfo;
    (void)numElements;
    (void)getFloat;
    return true;
}

CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomArcFace);
