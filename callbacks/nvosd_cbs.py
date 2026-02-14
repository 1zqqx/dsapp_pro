import logging

import gi

gi.require_version("Gst", "1.0")

from gi.repository import Gst  # type: ignore

import pyds

logger = logging.getLogger(__name__)


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
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            cid = obj_meta.class_id
            # 支持任一 class_id 计数 避免 key error
            obj_counter[cid] = obj_counter.get(cid, 0) + 1
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
        py_nvosd_text_params.display_text = (
            f"Frame Number={frame_number}; "
            f"Number of Objects={num_rects}; "
            f"Face_count={obj_counter[PGIE_CLASS_ID_FACE]}"
        )

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
            logger.info(pyds.get_string(py_nvosd_text_params.display_text))

        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK
