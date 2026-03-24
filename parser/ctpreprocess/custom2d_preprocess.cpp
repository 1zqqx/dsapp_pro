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
#include <iostream>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

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

    static bool debug_units_once = true;
    if (debug_units_once) {
        std::cerr << "[=] preprocess tensor slots debug: n_in=" << n_in
                  << ", units.size=" << batch->units.size()
                  << ", max_units=" << max_units
                  << ", network_input_order=" << (int)tp.network_input_order
                  << std::endl;
        std::cerr << "[=] tensor shape NCHW-like: n=" << n_in
                  << " c=" << c_in << " h=" << h_in << " w=" << w_in
                  << ", tensor_data_type=" << (int)tp.data_type
                  << ", network_color_format=" << (int)tp.network_color_format
                  << std::endl;
        for (int unit_idx = 0; unit_idx < max_units; ++unit_idx) {
            const NvDsPreProcessUnit& unit = batch->units[unit_idx];
            const auto& roi = unit.roi_meta.roi;
            unsigned int src_id = 0;
            if (unit.frame_meta) {
                src_id = (unsigned int)unit.frame_meta->source_id;
            }
            const NvBufSurfaceParams* conv = unit.roi_meta.converted_buffer;
            uint32_t pitch_dbg = conv ? conv->pitch : 0;
            NvBufSurfaceColorFormat fmt_dbg = conv ? conv->colorFormat : NVBUF_COLOR_FORMAT_INVALID;
            uint32_t height_dbg = conv ? conv->height : 0;
            uint32_t dataSize_dbg = conv ? conv->dataSize : 0;
            std::cerr << "  unit[" << unit_idx << "]: src_id=" << src_id
                      << " frame_num=" << (unit.frame_meta ? (unsigned long)unit.frame_meta->frame_num : 0ul)
                      << " converted=" << (unit.converted_frame_ptr ? 1 : 0)
                      << " roi=(" << (int)roi.left << "," << (int)roi.top << ","
                      << (int)roi.width << "x" << (int)roi.height << ")"
                      << " conv.pitch=" << pitch_dbg
                      << " conv.h=" << height_dbg
                      << " conv.fmt=" << (int)fmt_dbg
                      << " conv.dataSize=" << dataSize_dbg
                      << std::endl;
        }
        debug_units_once = false;
    }

    for (int unit_idx = 0; unit_idx < max_units; ++unit_idx) {
        const NvDsPreProcessUnit& unit = batch->units[unit_idx];

        if (!unit.converted_frame_ptr) {
            continue;
        }

        const void* src_ptr = unit.converted_frame_ptr;
        const uint8_t* src_cpu = nullptr;
        std::vector<uint8_t> host_copy;

        const size_t base = static_cast<size_t>(unit_idx) * nchw;

        // nvdspreprocess 内部对 RGB/BGR 会把 scaling pool 的输出格式统一成 RGBA
        // 因此这里必须使用 pitch + bpp 来正确索引，而不能假设紧密 packed RGB=3bytes/px。
        const NvBufSurfaceParams* conv = unit.roi_meta.converted_buffer;
        uint32_t pitch = conv ? conv->pitch : 0;
        NvBufSurfaceColorFormat fmt = conv ? conv->colorFormat : NVBUF_COLOR_FORMAT_INVALID;
        uint32_t conv_h = conv ? conv->height : 0;
        uint32_t conv_data_size = conv ? conv->dataSize : 0;

        // bytes per pixel for single-plane formats used by nvdspreprocess scaling pool
        int bpp = 0;
        switch (fmt) {
            case NVBUF_COLOR_FORMAT_RGBA:
            case NVBUF_COLOR_FORMAT_BGRA:
            case NVBUF_COLOR_FORMAT_ARGB:
            case NVBUF_COLOR_FORMAT_ABGR:
            case NVBUF_COLOR_FORMAT_RGBx:
            case NVBUF_COLOR_FORMAT_BGRx:
            case NVBUF_COLOR_FORMAT_xRGB:
            case NVBUF_COLOR_FORMAT_xBGR:
                bpp = 4;
                break;
            case NVBUF_COLOR_FORMAT_RGB:
            case NVBUF_COLOR_FORMAT_BGR:
                bpp = 3;
                break;
            case NVBUF_COLOR_FORMAT_GRAY8:
                bpp = 1;
                break;
            default:
                // 兜底：保守按 4 处理，避免 bpp 太小导致越界读。
                bpp = 4;
                break;
        }

        if (pitch == 0) {
            pitch = static_cast<uint32_t>(w_in) * static_cast<uint32_t>(bpp);
        }

        // 防止 pitch/bpp 推断错误导致越界。
        const int max_y = (conv_h > 0) ? std::min(h_in, (int)conv_h) : h_in;
        const int max_x_by_pitch = (bpp > 0) ? std::min(w_in, (int)(pitch / (uint32_t)bpp)) : w_in;

        // unit.converted_frame_ptr 可能指向 device memory，直接 src[src_idx] 解引用会 core dump。
        // 这里先拷贝到 CPU 可读的临时 buffer，再从 buffer 填充 tensor。
        size_t copy_bytes = static_cast<size_t>(pitch) * static_cast<size_t>(max_y);
        if (conv_data_size > 0) {
            copy_bytes = std::min(copy_bytes, static_cast<size_t>(conv_data_size));
        }
        if (copy_bytes > 0 && src_ptr != nullptr) {
            host_copy.resize(copy_bytes);
            cudaError_t cerr = cudaMemcpy(host_copy.data(), src_ptr, copy_bytes, cudaMemcpyDefault);
            if (cerr == cudaSuccess) {
                src_cpu = host_copy.data();
            } else {
                std::cerr << "[!] preprocess tensor: cudaMemcpy failed: " << cudaGetErrorString(cerr)
                          << " copy_bytes=" << copy_bytes << std::endl;
                src_cpu = nullptr;
            }
        } else {
            src_cpu = nullptr;
        }

        // Assume RGB packed input (3 channels) from scaling pool.
        // tensor 要输出 c_in 个通道，但源像素布局可能只有 1/3/4。
        // 当源为灰度(bpp=1)时，后续复制时需要避免 src[src_idx + c] 越界。
        const int channels = std::min(c_in, 3);

        if (is_nchw) {
            for (int c = 0; c < channels; ++c) {
                const size_t c_off = base + static_cast<size_t>(c) * h_in * w_in;
                for (int y = 0; y < max_y; ++y) {
                    for (int x = 0; x < max_x_by_pitch; ++x) {
                        const size_t src_row_off = static_cast<size_t>(y) * pitch;
                        const size_t src_px_off = static_cast<size_t>(x) * (size_t)bpp;
                        size_t src_idx = src_row_off + src_px_off;
                        // 如果源为灰度/单通道(bpp==1)，所有 c 都读同一个字节。
                        if (bpp > 1) {
                            src_idx += (size_t)c;
                        }
                        const size_t dst_idx = c_off + static_cast<size_t>(y) * w_in + x;
                        // src 指针假设可访问；另外用 conv_data_size 做保守边界。
                        if (conv_data_size > 0 && src_idx >= conv_data_size) {
                            dst[dst_idx] = 0.f;
                        } else {
                            if (src_cpu) {
                                dst[dst_idx] = static_cast<float>(src_cpu[src_idx]) * norm;
                            } else {
                                dst[dst_idx] = 0.f;
                            }
                        }
                    }
                }
            }
        } else {
            // NHWC layout.
            const size_t n_off = static_cast<size_t>(unit_idx) * h_in * w_in * channels;
            for (int y = 0; y < max_y; ++y) {
                for (int x = 0; x < max_x_by_pitch; ++x) {
                    const size_t src_row_off = static_cast<size_t>(y) * pitch;
                    const size_t src_px_off = static_cast<size_t>(x) * (size_t)bpp;
                    size_t src_idx = src_row_off + src_px_off;
                    const size_t dst_idx = n_off + (static_cast<size_t>(y) * w_in + x) * channels;
                    for (int c = 0; c < channels; ++c) {
                        size_t idx = src_idx;
                        if (bpp > 1) {
                            idx += (size_t)c;
                        }
                        if (conv_data_size > 0 && idx >= conv_data_size) {
                            dst[dst_idx + c] = 0.f;
                        } else {
                            if (src_cpu) {
                                dst[dst_idx + c] = static_cast<float>(src_cpu[idx]) * norm;
                            } else {
                                dst[dst_idx + c] = 0.f;
                            }
                        }
                    }
                }
            }
        }
    }

    return NVDSPREPROCESS_SUCCESS;
}
