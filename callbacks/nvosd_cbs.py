import gi

gi.require_version("Gst", "1.0")

from gi.repository import Gst  # type: ignore

import ctypes
import logging

import numpy as np
import pyds

logger = logging.getLogger(__name__)

# SGIE ArcFace 的 gie-unique-id，用于过滤 tensor meta（可选，不设则取第一个 NVDSINFER_TENSOR_OUTPUT_META）
SGIE_ARCFACE_UNIQUE_ID = 2
ARCFACE_EMBEDDING_DIM = 512


def get_arcface_embedding_from_obj(obj_meta, unique_id_filter=None):
    """
    从 NvDsObjectMeta 上挂的 user_meta 里取出 ArcFace embedding(512d & SGIE output-tensor-meta=1)

    :param obj_meta: pyds.NvDsObjectMeta
    :param unique_id_filter: 若指定，只取 tensor_meta.unique_id == unique_id_filter 的层
    :return: np.ndarray shape (512,) float32，若无则 None
    """
    l_user = obj_meta.obj_user_meta_list
    while l_user is not None:
        try:
            user_meta = pyds.NvDsUserMeta.cast(l_user.data)
        except StopIteration:
            break

        if (
            user_meta.base_meta.meta_type
            != pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META
        ):
            try:
                l_user = l_user.next
            except StopIteration:
                break
            continue

        try:
            tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
        except StopIteration:
            break

        if unique_id_filter is not None and tensor_meta.unique_id != unique_id_filter:
            try:
                l_user = l_user.next
            except StopIteration:
                break
            continue

        if tensor_meta.num_output_layers < 1:
            try:
                l_user = l_user.next
            except StopIteration:
                break
            continue

        try:
            layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
            ptr = ctypes.cast(
                pyds.get_ptr(layer.buffer),
                ctypes.POINTER(ctypes.c_float),
            )
            arr = np.ctypeslib.as_array(ptr, shape=(ARCFACE_EMBEDDING_DIM,))
            return np.array(arr, dtype=np.float32, copy=True)
        except Exception as e:
            logger.debug("get_arcface_embedding_from_obj: %s", e)
            return None

        try:
            l_user = l_user.next
        except StopIteration:
            break
    return None


PGIE_CLASS_ID_VEHICLE = 0  # 车辆
PGIE_CLASS_ID_BICYCLE = 1  # 自行车
PGIE_CLASS_ID_PERSON = 2  # 行人
PGIE_CLASS_ID_ROADSIGN = 3  # 交通标志


def nvdsosd_sink_pad_buffer_probe(osd_sink_pad, info, u_data):
    """
    Docstring for osd_sink_pad_buffer_probe

    :args:
    + pad: 调用此回调函数的 pad 这里是 nvdsosd -> sink
    + info: 指向 GstBuffer 中数据 的指针
    + u_data: 用户在使用设置 此回调函数 时传进来的参数

    :return:
    + Gst.PadProbeReturn.OK (int)
    """

    frame_number = 0
    # Intiallizing object counter with 0.
    obj_counter = {
        PGIE_CLASS_ID_VEHICLE: 0,
        PGIE_CLASS_ID_PERSON: 0,
        PGIE_CLASS_ID_BICYCLE: 0,
        PGIE_CLASS_ID_ROADSIGN: 0,
    }
    num_rects = 0

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        logger.error(" Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
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
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            obj_counter[obj_meta.class_id] += 1
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # Acquiring a display meta object. The memory ownership remains in
        # the C code so downstream plugins can still access it. Otherwise
        # the garbage collector will claim it when this probe function exits.
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        # Setting display text to be shown on screen
        # Note that the pyds module allocates a buffer for the string, and the
        # memory will not be claimed by the garbage collector.
        # Reading the display_text field here will return the C address of the
        # allocated string. Use pyds.get_string() to get the string content.
        py_nvosd_text_params.display_text = "Frame Number={} Number of Objects={} Vehicle_count={} Person_count={}".format(
            frame_number,
            num_rects,
            obj_counter[PGIE_CLASS_ID_VEHICLE],
            obj_counter[PGIE_CLASS_ID_PERSON],
        )

        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12

        # Font, text color, font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 10

        # set(red, green, blue, alpha); set to White
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # Text background color
        py_nvosd_text_params.set_bg_clr = 1
        # background set(red, green, blue, alpha); set to Black; alpha -> 透明度
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)

        # Using pyds.get_string() to get display_text as string
        logger.info(pyds.get_string(py_nvosd_text_params.display_text))

        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


# SCRFD 单类别，face 的 classId 为 0
PGIE_CLASS_ID_FACE = 0


def scrfd_nvdsosd_sink_pad_buffer_probe(osd_sink_pad, info, u_data):
    """
    Docstring for scrfd_nvdsosd_sink_pad_buffer_probe

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
        face_embedings = []  # list of np.ndarray shape (512,) float32 's L2 norm
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            cid = obj_meta.class_id
            # 支持任一 class_id 计数 避免 key error
            obj_counter[cid] = obj_counter.get(cid, 0) + 1

            # SGIE ArcFace, get 512d embedding
            emb = get_arcface_embedding_from_obj(obj_meta, SGIE_ARCFACE_UNIQUE_ID)
            if emb is not None:
                norm = float(np.linalg.norm(emb))  # L2 norm
                face_embedings.append(norm)
                logger.debug(
                    f"frame={frame_number} arcface_embedding dim={len(emb)} norm={norm} head3={emb[:3].tolist()}",
                    # obj_meta.rect_params.left,
                    # obj_meta.rect_params.top,
                    # obj_meta.rect_params.width,
                    # obj_meta.rect_params.height,
                )

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

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
        dt_str = (
            f"Frame Number={frame_number}; "
            f"Number of Objects={num_rects}; "
            f"Face_count={obj_counter[PGIE_CLASS_ID_FACE]}"
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
            # logger.info(pyds.get_string(py_nvosd_text_params.display_text))
            logger.info(dt_str)
            logger.info(f"face_embedings L2 norm: {face_embedings}")

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
