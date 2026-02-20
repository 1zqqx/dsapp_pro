import gi

gi.require_version("Gst", "1.0")

from gi.repository import Gst  # type: ignore

import logging

logger = logging.getLogger(__name__)

import numpy as np
import pyds

from .nvosd_cbs import PGIE_CLASS_ID_FACE
from .nvosd_cbs import get_arcface_embedding_from_obj

# SGIE ArcFace 的 gie-unique-id，用于过滤 tensor meta（可选，不设则取第一个 NVDSINFER_TENSOR_OUTPUT_META）
SGIE_ARCFACE_UNIQUE_ID = 2


def arcface_nvdsosd_sink_pad_buffer_probe(osd_sink_pad, info, u_data):
    """
    Docstring for arcface_nvdsosd_sink_pad_buffer_probe

    :param pad: 调用此回调函数的 pad 这里是 nvdsosd -> sink
    :param info: 指向 GstBuffer 中数据 的指针
    :param u_data: 用户在使用设置 此回调函数 时传进来的参数

    :return: Gst.PadProbeReturn.OK (int)
    """

    frame_number = 0
    # Intiallizing object counter with 0.
    obj_counter = {
        PGIE_CLASS_ID_FACE: 0,
    }
    num_rects = 0

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        logger.warning(" Unable to get GstBuffer ")
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if batch_meta is None:
        logger.warning(" Unable to get batch meta from GstBuffer ")
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj = frame_meta.obj_meta_list
        # 收集有 embedding 的人脸对应的 obj_meta 与向量，用于 FAISS 后写回框内文字
        face_embeddings = []
        face_obj_metas = []
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            cid = obj_meta.class_id
            obj_counter[cid] = obj_counter.get(cid, 0) + 1

            # SGIE ArcFace, get 512d embedding
            emb = get_arcface_embedding_from_obj(obj_meta, SGIE_ARCFACE_UNIQUE_ID)
            if emb is not None:
                face_obj_metas.append(obj_meta)
                face_embeddings.append(emb)
                logger.debug(
                    f"frame={frame_number} arcface_embedding dim={len(emb)} norm={np.linalg.norm(emb)}",
                    # obj_meta.rect_params.left,
                    # obj_meta.rect_params.top,
                    # obj_meta.rect_params.width,
                    # obj_meta.rect_params.height,
                )

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # FAISS 匹配，拿到名字与置信度（余弦相似度）
        matched_names = []
        matched_scores = []
        if face_embeddings and u_data is not None:
            try:
                matched_names, matched_scores = u_data.search_with_scores(
                    face_embeddings
                )
            except Exception as e:
                logger.warning(f" faiss search failed: {e}")

        # 每个检测到的人脸框左上角显示：名字 + 置信度(覆盖默认的 face/face_embedding)
        for i, obj_meta in enumerate(face_obj_metas):
            txt = obj_meta.text_params
            if i < len(matched_names) and i < len(matched_scores):
                name = matched_names[i]
                score = matched_scores[i]
                txt.display_text = f"{name} {score:.2f}"
            else:
                txt.display_text = "Unknown"
            txt.font_params.font_name = "Serif"
            txt.font_params.font_size = 15
            txt.font_params.font_color.set(0.0, 1.0, 0.0, 1.0)  # 绿色
            txt.set_bg_clr = 0
            # txt.text_bg_clr.set(0.15, 0.15, 0.15, 0.85)  # 深灰半透明，替代纯黑

        # Acquiring a display meta object. The memory ownership remains in
        # the C code so downstream plugins can still access it. Otherwise
        # the garbage collector will claim it when this probe function exits.
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1

        # & display_meta.text_params[0] 获取第一个 text_params 的地址，py_nvosd_text_params 是一个 pyds.NvOSD_TextParams 对象
        py_nvosd_text_params = display_meta.text_params[0]

        # Setting display text to be shown on screen
        # Note that the pyds module allocates a buffer for the string, and the
        # memory will not be claimed by the garbage collector.
        # Reading the display_text field here will return the C address of the
        # allocated string. Use pyds.get_string() to get the string content.
        face_match_str = (" | ".join(matched_names)) if matched_names else ""
        dt_str = (
            f"Frame Number={frame_number}; "
            f"Number of Objects={num_rects}; "
            f"Face_count={obj_counter[PGIE_CLASS_ID_FACE]}"
            f"; Match={face_match_str}"
            if face_match_str
            else ""
        )
        # 对 display_text 赋值时 会在 C 侧为这一段文字分配一块内存 并把 py 字符串拷贝进去 这块内存 由 C 管理不会被 GC 回收
        py_nvosd_text_params.display_text = dt_str

        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12

        # Font, text color, font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 14
        # set(red, green, blue, alpha); set to green

        py_nvosd_text_params.font_params.font_color.set(0, 1.0, 0, 1.0)
        # Text background color
        py_nvosd_text_params.set_bg_clr = 0
        # py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.0)

        # Using pyds.get_string() to get display_text as string
        if frame_number % 15 == 0:
            logger.info(dt_str)
            if matched_names:
                logger.info(f" face matches: {matched_names} ")

        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    # batch_meta.batch_user_meta_list
    l_user_meta = batch_meta.batch_user_meta_list
    while l_user_meta is not None:
        try:
            # Casting l_user_meta.data to pyds.NvDsUserMeta
            user_meta = pyds.NvDsUserMeta.cast(l_user_meta.data)
        except StopIteration:
            break
        if user_meta.base_meta.meta_type == pyds.NVDS_PREPROCESS_BATCH_META:
            try:
                # Casting user_meta.data to pyds.GstNvDsPreProcessBatchMeta
                preprocess_batchmeta = pyds.GstNvDsPreProcessBatchMeta.cast(
                    user_meta.user_meta_data
                )
            except StopIteration:
                break
            roi_cnt = 0

            for roi_meta in preprocess_batchmeta.roi_vector:
                # Label ROI in display
                display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                display_meta.num_labels = 1

                txt_params = display_meta.text_params[0]
                txt_params.display_text = f"Roi:{roi_cnt}"

                txt_params.x_offset = int(roi_meta.roi.left)
                txt_params.y_offset = int(roi_meta.roi.top)

                txt_params.font_params.font_name = "Serif"
                txt_params.font_params.font_size = 16
                txt_params.font_params.font_color.red = 0
                txt_params.font_params.font_color.green = 1.0
                txt_params.font_params.font_color.blue = 0
                txt_params.font_params.font_color.alpha = 1.0

                txt_params.set_bg_clr = 0
                # txt_params.text_bg_clr.red = 0.0
                # txt_params.text_bg_clr.green = 0.0
                # txt_params.text_bg_clr.blue = 0.0
                # txt_params.text_bg_clr.alpha = 0.5

                pyds.nvds_add_display_meta_to_frame(roi_meta.frame_meta, display_meta)

                logger.debug(
                    f"frame {roi_meta.frame_meta.frame_num} src {roi_meta.frame_meta.source_id} roi {roi_cnt}"
                )
                roi_cnt += 1
        try:
            l_user_meta = l_user_meta.next
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK
