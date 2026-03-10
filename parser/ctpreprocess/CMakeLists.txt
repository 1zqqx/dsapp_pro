cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(dsapp_libcustom2d_preprocess LANGUAGES CXX)

set(DS_INCLUDES "/opt/nvidia/deepstream/deepstream-8.0/sources/includes"
    CACHE PATH "DeepStream NvDsPreProcess includes (nvdspreprocess_interface.h, nvdspreprocess_meta.h)")

add_library(custom2d_preprocess SHARED custom2d_preprocess.cpp)

target_include_directories(custom2d_preprocess PRIVATE
    ${DS_INCLUDES}
    /usr/include
    /usr/include/gstreamer-1.0
    /usr/include/glib-2.0
    /usr/lib/x86_64-linux-gnu/glib-2.0/include
    /usr/local/cuda/include
)


target_compile_options(custom2d_preprocess PRIVATE -fPIC -Wall -O2)

# link_directories(/opt/nvidia/deepstream/deepstream/lib)
# target_link_libraries(custom2d_preprocess PRIVATE
#     nvbufsurface
#     nvbufsurftransform
#     nvdsgst_meta
#     nvdspreprocess_meta
#     nvdspreprocess
#     gstreamer-1.0
#     glib-2.0
#     gobject-2.0
# )

set_target_properties(custom2d_preprocess PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "custom2d_preprocess"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../libs"
)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "LIBRARY_OUTPUT_DIRECTORY: ${CMAKE_CURRENT_SOURCE_DIR}/../../libs")
