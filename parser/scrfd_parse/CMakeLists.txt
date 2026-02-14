cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(scrfd_parser LANGUAGES CXX)

set(DS_INCLUDES "/opt/nvidia/deepstream/deepstream-8.0/sources/includes"
    CACHE PATH "DeepStream nvds infer includes (nvdsinfer.h, nvdsinfer_custom_impl.h)")

add_library(nvdsinfer_custom_scrfd SHARED nvdsinfer_custom_scrfd.cpp)

target_include_directories(nvdsinfer_custom_scrfd PRIVATE
    ${DS_INCLUDES}
    /usr/include
    /usr/local/cuda/include
)

target_compile_options(nvdsinfer_custom_scrfd PRIVATE -fPIC -Wall -O2)

set_target_properties(nvdsinfer_custom_scrfd PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "nvdsinfer_custom_scrfd"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../libs"
)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "LIBRARY_OUTPUT_DIRECTORY: ${CMAKE_CURRENT_SOURCE_DIR}/../../libs")
