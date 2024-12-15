cmake_minimum_required(VERSION 3.5)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(imgprocess_node)
enable_language(CXX CUDA)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")# 对于GCC和Clang，使用 -w 禁止警告和注释
else ()
    message(WARNING "Unsupported compiler, you may still see warnings and notes.")
endif ()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(common_msgs REQUIRED)      
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)                          

ament_export_dependencies(rosidl_default_runtime)

if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    ament_lint_auto_find_test_dependencies()
endif()

ament_package()

find_package(rclcpp REQUIRED)

# CUDA
set(CUDA_DIR /usr/local/cuda)
find_package(CUDA REQUIRED)
message(STATUS "CUDA Libs: \n${CUDA_LIBRARIES}\n")
message(STATUS "CUDA Headers: \n${CUDA_INCLUDE_DIRS}\n")

# OpenCV
set(OpenCV_DIR /usr/local/share/opencv4)
find_package(OpenCV 4.5 REQUIRED)

set(TensorRT_INCLUDE_DIRS /home/gray/Documents/ubuntu/TensorRT-8.5.1.7.Linux.x86_64-gnu.cuda-11.8.cudnn8.6/TensorRT-8.5.1.7/include)
set(TensorRT_LIBRARIES /home/gray/Documents/ubuntu/TensorRT-8.5.1.7.Linux.x86_64-gnu.cuda-11.8.cudnn8.6/TensorRT-8.5.1.7/targets/x86_64-linux-gnu/lib/stubs)

FILE(GLOB SOURCE_LIST "${CMAKE_SOURCE_DIR}/src/*.cpp")
FILE(GLOB BYTETRACK_LIST "${CMAKE_SOURCE_DIR}/src/bytetrack/*.cpp")

list(APPEND INCLUDE_DIRS
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/bytetrack
    ./include
)

list(APPEND ALL_LIBS
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    nvinfer 
    nvinfer_plugin 
    cudart 
)

include_directories(${INCLUDE_DIRS})

add_executable(imgprocess_node ${SOURCE_LIST} ${BYTETRACK_LIST})
    ament_target_dependencies(imgprocess_node 
    rclcpp
    common_msgs
    sensor_msgs
    std_msgs
    cv_bridge
)

target_link_libraries(imgprocess_node ${ALL_LIBS})

install(TARGETS
    imgprocess_node
    DESTINATION lib/${PROJECT_NAME}
)
