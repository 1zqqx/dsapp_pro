cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(arcface LANGUAGES CXX)

set(DS_INCLUDES "/opt/nvidia/deepstream/deepstream-8.0/sources/includes"
    CACHE PATH "DeepStream nvds infer includes (nvdsinfer.h, nvdsinfer_custom_impl.h)")

add_library(nvdsinfer_custom_arcface_r50_wbface SHARED arcface_r50_wbface.cpp)

target_include_directories(nvdsinfer_custom_arcface_r50_wbface PRIVATE
    ${DS_INCLUDES}
    /usr/include
    /usr/local/cuda/include
)

target_compile_options(nvdsinfer_custom_arcface_r50_wbface PRIVATE -fPIC -Wall -O2)

set_target_properties(nvdsinfer_custom_arcface_r50_wbface PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "nvdsinfer_custom_arcface_r50_wbface"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../libs"
)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "LIBRARY_OUTPUT_DIRECTORY: ${CMAKE_CURRENT_SOURCE_DIR}/../../libs")
