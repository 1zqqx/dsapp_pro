/*
 * SCRFD (InsightFace) custom bbox parser for DeepStream nvinfer.
 * Parses 9 output layers:
 *   scores: (1,12800,1), (1,3200,1), (1,800,1)
 *   bbox:   (1,12800,4), (1,3200,4), (1,800,4)
 *   kps:    (1,12800,10), (1,3200,10), (1,800,10)  [ignored for detection]
 * Strides: 8, 16, 32. Bbox format: (l,t,r,b) from center, multiply by stride.
 */

#include "nvdsinfer_custom_impl.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLIP(a, min_val, max_val) (MAX(MIN(a, max_val), min_val))

/**
 * 从推理输出缓冲区中按线性下标读取一个浮点值.
 * nvinfer 可能使用 FLOAT 或 HALF(FP16)，需按 dataType 正确解析.
 * @param buf   层 buffer 指针(CPU 侧，由 nvinfer 在调用 parser 前拷贝)
 * @param idx   线性下标(第 idx 个元素)
 * @param dtype FLOAT 或 HALF
 * @return 对应的 float 值
 */
static float getFloat(const void *buf, int idx, NvDsInferDataType dtype)
{
    if (dtype == FLOAT)
    {
        return ((const float *)buf)[idx];
    }
    if (dtype == HALF)
    {
        /* FP16: 按 IEEE 754 half 格式解析 */
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
 * 按 "存在某两维等于 want_anchors / want_c" 匹配, 兼容 TensorRT 不同维度顺序 (N,L,C) vs (N,C,L)
 * 在 outputLayersInfo 中按 "形状" 查找某一层，不假定维度顺序.
 * TensorRT 输出的 inferDims 可能是 (N,L,C) 或 (N,C,L)，只要存在两维分别等于
 * want_anchors 和 want_c 即视为匹配(例如 score 层 12800×1，bbox 层 12800×4).
 * @param layers       所有输出层信息
 * @param want_anchors 期望的 "锚点数" 维大小(如 12800、3200、800)
 * @param want_c       期望的 "通道" 维大小(score=1，bbox=4)
 * @param out_idx      输出: 匹配到的层在 layers 中的下标
 * @return 是否找到
 */
static bool findLayerByShape(
    const std::vector<NvDsInferLayerInfo> &layers,
    int want_anchors,
    int want_c,
    int &out_idx)
{
    for (size_t i = 0; i < layers.size(); i++)
    {
        const NvDsInferDims &d = layers[i].inferDims;
        bool has_na = false, has_c = false;
        for (unsigned int k = 0; k < d.numDims; k++)
        {
            if ((int)d.d[k] == want_anchors)
                has_na = true;
            if ((int)d.d[k] == want_c)
                has_c = true;
        }
        if (has_na && has_c)
        {
            out_idx = (int)i;
            return true;
        }
    }
    return false;
}

/**
 * 计算 bbox 层中 (anchor_idx, channel_k) 在连续内存中的线性下标.
 * 布局 (1, 12800, 4) 时: channel 在最后一维，index = anchor_idx * 4 + k.
 * 布局 (1, 4, 12800) 时: channel 在中间，index = k * na + anchor_idx.
 * @param anchor_idx   锚点下标 [0, na)
 * @param k            bbox 分量 0..3 (l,t,r,b)
 * @param na           该 stride 的锚点总数
 * @param anchor_dim   inferDims 中大小为 na 的维度下标
 * @param channel_dim  inferDims 中大小为 4 的维度下标
 */
static inline __attribute__((always_inline)) int bboxLinearIndex(int anchor_idx, int k, int na, int anchor_dim, int channel_dim)
{
    if (channel_dim > anchor_dim)
        return anchor_idx * 4 + k;
    return k * na + anchor_idx;
}

/**
 * 自定义 bbox 解析入口: NvDsInferParseCustomSCRFD
 * 由 nvinfer 在每次推理完成后调用，将模型原始输出解析为 NvDsInferObjectDetectionInfo 列表.
 *
 * 参数说明:
 * @param outputLayersInfo  各输出层的 buffer、inferDims、dataType 等(只读)
 * @param networkInfo       网络输入宽高 (width/height)，用于将坐标裁剪到图像内
 * @param detectionParams   配置中的类别数、perClassPreclusterThreshold 等
 * @param objectList        输出: 解析得到的目标框列表，push_back 填入
 * @return true 表示解析成功; false 会导致 nvinfer 报错
 */
extern "C" bool NvDsInferParseCustomSCRFD(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    /* 至少需要 6 层: 3 个 stride 的 score + bbox */
    if (outputLayersInfo.size() < 6u)
    {
        std::cerr << "SCRFD parser: expected at least 6 output layers (score+bbox x3), got "
                  << outputLayersInfo.size() << std::endl;
        return false;
    }

    /* SCRFD 三尺度: stride 8/16/32 对应 80×80×2、40×40×2、20×20×2 个锚点 */
    const int strides[] = {8, 16, 32};
    const int anchors[] = {12800, 3200, 800}; /* 80*80*2, 40*40*2, 20*20*2 */
    const int heights[] = {80, 40, 20};
    const int widths[] = {80, 40, 20};

    /* 置信度阈值: 优先使用配置文件 [class-attrs-all] 的 pre-cluster-threshold */
    float threshold = 0.5f;
    if (detectionParams.numClassesConfigured > 0 &&
        !detectionParams.perClassPreclusterThreshold.empty())
    {
        threshold = detectionParams.perClassPreclusterThreshold[0];
    }

    const int net_w = (int)networkInfo.width;
    const int net_h = (int)networkInfo.height;

    for (int s = 0; s < 3; s++)
    {
        int stride = strides[s];
        int na = anchors[s];
        int grid_h = heights[s];
        int grid_w = widths[s];
        int num_cells = grid_h * grid_w;
        int num_anchors_per_cell = na / num_cells; /* 2 -> each cell has two anchor boxes.*/

        int score_idx = -1, bbox_idx = -1;
        if (!findLayerByShape(outputLayersInfo, na, 1, score_idx) ||
            !findLayerByShape(outputLayersInfo, na, 4, bbox_idx))
        {
            continue; /* skip this stride if layers not found */
        }

        const NvDsInferLayerInfo &score_layer = outputLayersInfo[score_idx];
        const NvDsInferLayerInfo &bbox_layer = outputLayersInfo[bbox_idx];

        if (!score_layer.buffer || !bbox_layer.buffer)
            continue;

        /* 确定 bbox 层维度顺序，以便 bboxLinearIndex 正确计算线性下标 */
        int bbox_anchor_dim = -1, bbox_channel_dim = -1;
        for (unsigned int k = 0; k < bbox_layer.inferDims.numDims; k++)
        {
            if ((int)bbox_layer.inferDims.d[k] == na)
                bbox_anchor_dim = (int)k;
            if ((int)bbox_layer.inferDims.d[k] == 4)
                bbox_channel_dim = (int)k;
        }
        if (bbox_anchor_dim < 0 || bbox_channel_dim < 0)
            continue;

        NvDsInferDataType score_dtype = score_layer.dataType;
        NvDsInferDataType bbox_dtype = bbox_layer.dataType;

        for (int idx = 0; idx < na; idx++)
        {
            float conf = getFloat(score_layer.buffer, idx, score_dtype);
            if (conf < threshold)
                continue;

            /* 锚点对应特征图上的 cell，中心在图像坐标 (cx, cy) */
            int cell_idx = idx / num_anchors_per_cell;
            int cy_grid = cell_idx / grid_w;
            int cx_grid = cell_idx % grid_w;
            float cx = (cx_grid + 0.5f) * stride;
            float cy = (cy_grid + 0.5f) * stride;

            /* SCRFD bbox 格式: 中心到四边的距离 (l,t,r,b)，网络输出需乘 stride 得像素距离 */
            float l = getFloat(bbox_layer.buffer, bboxLinearIndex(idx, 0, na, bbox_anchor_dim, bbox_channel_dim), bbox_dtype) * stride;
            float t = getFloat(bbox_layer.buffer, bboxLinearIndex(idx, 1, na, bbox_anchor_dim, bbox_channel_dim), bbox_dtype) * stride;
            float r = getFloat(bbox_layer.buffer, bboxLinearIndex(idx, 2, na, bbox_anchor_dim, bbox_channel_dim), bbox_dtype) * stride;
            float b = getFloat(bbox_layer.buffer, bboxLinearIndex(idx, 3, na, bbox_anchor_dim, bbox_channel_dim), bbox_dtype) * stride;

            /* distance2bbox: 左=中心x-左距，上=中心y-上距，右/下=中心+右/下距 */
            float x1 = cx - l;
            float y1 = cy - t;
            float x2 = cx + r;
            float y2 = cy + b;

            /*
             * SCRFD 做按锚点 的 墨迹密集预测 这里填的是「候选框」，不是最终每脸一个的框.
             * 每个 stride 对应一个特征图 每个网格点 就是 锚点
             * 每个锚点输出 1 个 分数(这个位置有人脸的概率) + 4 个 坐标(l,t,r,b)
             * 每个锚点 (12800+3200+800 个) 只要置信度(1 个分数) 大于等于 阈值就输出一个框，
             * 同一张脸会被多个锚点/多尺度重复检出，产生大量重叠框.
             * nvinfer 在 parser 返回后会对此 objectList 做 NMS，保留高置信度、抑制重叠框，
             * 得到 每脸一个 的最终 NvDsObjectMeta. 因此 parser 只负责：原始张量 → 候选框列表.
             */
            NvDsInferObjectDetectionInfo obj;
            obj.classId = 0; /* SCRFD 单类别: 0=face */
            obj.detectionConfidence = conf;
            obj.left = CLIP(x1, 0.f, (float)(net_w - 1));
            obj.top = CLIP(y1, 0.f, (float)(net_h - 1));
            obj.width = CLIP(x2 - x1 + 1.f, 1.f, (float)net_w);
            obj.height = CLIP(y2 - y1 + 1.f, 1.f, (float)net_h);

            objectList.push_back(obj);
        }
    }

    static bool debug_once = true;
    if (debug_once)
    {
        debug_once = false;
        std::cerr << "[=] DEBUG | SCRFD parser: " << outputLayersInfo.size() << " layers, "
                  << networkInfo.width << "x" << networkInfo.height << ", objects=" << objectList.size() << std::endl;
    }
    return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomSCRFD);

/**
 * cd apps/dsapp/parser/scrfd_parse
 * mkdir build && cd build
 * cmake -S .. -B .
 * or
 * cmake .. -DDS_INCLUDES=/opt/nvidia/deepstream/deepstream-8.0/sources/includes
 * make
 */
