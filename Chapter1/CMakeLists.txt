cmake_minimum_required(VERSION 3.25)

project(main LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE "Release" )
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_COMPILER "/opt/intel/oneapi/compiler/latest/bin/icx")
set(CMAKE_CXX_COMPILER "/opt/intel/oneapi/compiler/latest/bin/icpx")
set(IntelSYCL_DIR "/opt/intel/oneapi/compiler/latest/lib/cmake/IntelSYCL/")
set(CMAKE_CXX_FLAGS "-O2 -fsycl -Wall -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64")


get_filename_component (PROJECT_DIR ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_DIR}/lib)

file(GLOB SRC_FILES
    "${PROJECT_SOURCE_DIR}/include/*.h"
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
)

find_package(IntelSYCL REQUIRED)

include_directories(${PROJECT_DIR}/include)
include_directories(${IntelSYCL_INCLUDE_DIRS})

add_executable(${CMAKE_PROJECT_NAME} ${SRC_FILES})
add_sycl_to_target(TARGET ${CMAKE_PROJECT_NAME} SOURCES ${SRC_FILES})


target_link_libraries(${CMAKE_PROJECT_NAME} ${IntelSYCL_LIBRARIES})
