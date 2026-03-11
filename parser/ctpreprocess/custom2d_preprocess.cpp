// Minimal custom 2D preprocess library for nvdspreprocess
// - Synchronous group transform (CustomTransformation): applies NvBufSurfTransform
//   to all ROIs/frames described by CustomTransformParams.
// - Tensor preparation (CustomTensorPreparation): converts converted_frame_ptr
//   from scaling pool into an NCHW (or NHWC) float tensor for all units in the batch.
//
// This implementation intentionally keeps the logic simple and deterministic:
// every NvDsPreProcessUnit in NvDsPreProcessBatch::units is processed in order,
// and tensor slots are filled sequentially, so each ROI is mapped to a fixed
// index within the batch tensor.

#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
// gstnvdsmeta 头文件 中使用了 string/vector 在此之前引入
#include <gstnvdsmeta.h>
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>
#include <nvdspreprocess_interface.h>
#include <nvdspreprocess_meta.h>

// 插件 gstnvdspreprocess.cpp:509 要求 initLib 必须返回非空，否则报 "Error while initializing Custom Library"。
// 插件只把 ctx 透传给 CustomTensorPreparation 和 deInitLib，从不解引用，故用静态缓冲区占位即可。
static char s_ctx_placeholder[64];

extern "C" CustomCtx* initLib(CustomInitParams initParams) {
    (void)initParams;
    return reinterpret_cast<CustomCtx*>(s_ctx_placeholder);
}

extern "C" void deInitLib(CustomCtx* ctx) { (void)ctx; }

extern "C" NvDsPreProcessStatus CustomTransformation(NvBufSurface* in_surf, NvBufSurface* out_surf,
                                                     CustomTransformParams& params) {
    if (!in_surf || !out_surf) {
        return NVDSPREPROCESS_INVALID_PARAMS;
    }

    // Configure transform session.
    NvBufSurfTransform_Error err =
        NvBufSurfTransformSetSessionParams(&params.transform_config_params);
    if (err != NvBufSurfTransformError_Success) {
        return NVDSPREPROCESS_CUDA_ERROR;
    }

    // Synchronous transform for the whole batch / all ROIs.
    err = NvBufSurfTransform(in_surf, out_surf, &params.transform_params);
    if (err != NvBufSurfTransformError_Success) {
        return NVDSPREPROCESS_CUDA_ERROR;
    }

    // If a sync object is set (for async flows), wait for completion here so
    // that from the custom lib point of view the transform is finished.
    if (params.sync_obj) {
        NvBufSurfTransformSyncObjWait(params.sync_obj, -1);
    }

    return NVDSPREPROCESS_SUCCESS;
}

extern "C" NvDsPreProcessStatus CustomAsyncTransformation(NvBufSurface* in_surf,
                                                          NvBufSurface* out_surf,
                                                          CustomTransformParams& params) {
    // For this application we do not rely on true async behavior. Implement
    // the async entry point as a thin wrapper around the synchronous
    // CustomTransformation so that every frame/ROI is processed in a single
    // pass without round‑robin scheduling.
    return CustomTransformation(in_surf, out_surf, params);
}

// Same key as nvdspreprocess config [user-configs]; not in public interface header
static const char* PIXEL_NORM_FACTOR_KEY = "pixel-normalization-factor";

static inline float get_norm_factor(
    const std::unordered_map<std::string, std::string>& user_configs) {
    auto it = user_configs.find(PIXEL_NORM_FACTOR_KEY);
    if (it == user_configs.end()) {
        return 1.0f / 255.0f;
    }
    try {
        return std::stof(it->second);
    } catch (...) {
        return 1.0f / 255.0f;
    }
}

extern "C" NvDsPreProcessStatus CustomTensorPreparation(CustomCtx* ctx, NvDsPreProcessBatch* batch,
                                                        NvDsPreProcessCustomBuf*& buf,
                                                        CustomTensorParams& tensorParam,
                                                        NvDsPreProcessAcquirer* acquirer) {
    (void)ctx;

    if (!batch || !acquirer) {
        return NVDSPREPROCESS_INVALID_PARAMS;
    }

    const auto& tp = tensorParam.params;

    if (tp.network_input_shape.size() < 4) {
        return NVDSPREPROCESS_INVALID_PARAMS;
    }

    const int n_in = tp.network_input_shape[0];
    const int c_in = tp.network_input_shape[1];
    const int h_in = tp.network_input_shape[2];
    const int w_in = tp.network_input_shape[3];

    const size_t nchw = static_cast<size_t>(c_in) * h_in * w_in;

    // Acquire buffer for the whole batch.
    buf = acquirer->acquire();
    if (!buf || !buf->memory_ptr) {
        return NVDSPREPROCESS_RESOURCE_ERROR;
    }

    // Only FP32 tensors are handled here. (NvDsDataType is C enum: NvDsDataType_FP32)
    if (tp.data_type != NvDsDataType_FP32) {
        return NVDSPREPROCESS_INVALID_PARAMS;
    }

    float* dst = static_cast<float*>(buf->memory_ptr);

    // Pixel normalization factor (defaults to 1/255).
    // In this minimal implementation we ignore mean offsets for simplicity.
    float norm = 1.0f / 255.0f;
    // user_configs are only available at init time; for a fully featured
    // implementation you would cache them in CustomCtx. Here we just rely on
    // default normalization to keep code simple and deterministic.

    const bool is_nchw = (tp.network_input_order == NvDsPreProcessNetworkInputOrder_kNCHW);

    const int max_units = std::min<int>(static_cast<int>(batch->units.size()), n_in);

    for (int unit_idx = 0; unit_idx < max_units; ++unit_idx) {
        const NvDsPreProcessUnit& unit = batch->units[unit_idx];

        if (!unit.converted_frame_ptr) {
            continue;
        }

        const uint8_t* src = static_cast<const uint8_t*>(unit.converted_frame_ptr);

        const size_t base = static_cast<size_t>(unit_idx) * nchw;

        // Assume RGB packed input (3 channels) from scaling pool.
        const int channels = std::min(c_in, 3);

        if (is_nchw) {
            for (int c = 0; c < channels; ++c) {
                const size_t c_off = base + static_cast<size_t>(c) * h_in * w_in;
                for (int y = 0; y < h_in; ++y) {
                    for (int x = 0; x < w_in; ++x) {
                        const size_t src_idx = (static_cast<size_t>(y) * w_in + x) * 3 + c;
                        const size_t dst_idx = c_off + static_cast<size_t>(y) * w_in + x;
                        dst[dst_idx] = static_cast<float>(src[src_idx]) * norm;
                    }
                }
            }
        } else {
            // NHWC layout.
            const size_t n_off = static_cast<size_t>(unit_idx) * h_in * w_in * channels;
            for (int y = 0; y < h_in; ++y) {
                for (int x = 0; x < w_in; ++x) {
                    const size_t src_idx = (static_cast<size_t>(y) * w_in + x) * 3;
                    const size_t dst_idx = n_off + (static_cast<size_t>(y) * w_in + x) * channels;
                    for (int c = 0; c < channels; ++c) {
                        dst[dst_idx + c] = static_cast<float>(src[src_idx + c]) * norm;
                    }
                }
            }
        }
    }

    return NVDSPREPROCESS_SUCCESS;
}
