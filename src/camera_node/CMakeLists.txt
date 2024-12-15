cmake_minimum_required(VERSION 3.5)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(camera_node)
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

# realsense
set(realsense2_DIR /usr/share/realsense2)
find_package(realsense2 REQUIRED)

FILE(GLOB SOURCE_LIST "${CMAKE_SOURCE_DIR}/src/*.cpp")

list(APPEND INCLUDE_DIRS
${CUDA_INCLUDE_DIRS}
${realsense2_INCLUDE_DIR} 
${OpenCV_INCLUDE_DIRS}
${PCL_INCLUDE_DIRS}
        ./include
        )

list(APPEND ALL_LIBS
${CUDA_LIBRARIES}
${OpenCV_LIBS}
${PCL_LIBRARIES}
${realsense2_LIBRARY}
        )

include_directories(${INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)

add_executable(camera_node ${SOURCE_LIST})
ament_target_dependencies(camera_node 
rclcpp
common_msgs
sensor_msgs
std_msgs
cv_bridge)

target_link_libraries(camera_node ${ALL_LIBS})

install(TARGETS camera_node DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
