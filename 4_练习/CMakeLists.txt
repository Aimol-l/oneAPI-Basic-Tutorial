cmake_minimum_required(VERSION 3.23)


set(CMAKE_C_COMPILER icx)
set(CMAKE_CXX_COMPILER icpx)

# set(TARGET_DEVICE "CPU")
if(TARGET_DEVICE MATCHES "GPU")
    add_compile_definitions(ACC_DEVIDE_GPU)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -Wall -fsycl-targets=nvptx64-nvidia-cuda")
elseif(TARGET_DEVICE MATCHES "CPU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -Wall")
    add_compile_definitions(ACC_DEVIDE_CPU)
endif()

project(main LANGUAGES CXX)

get_filename_component (PROJECT_DIR ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_DIR}/bin)
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
