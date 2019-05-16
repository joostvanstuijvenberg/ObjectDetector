cmake_minimum_required(VERSION 3.12)
project(ObjectDetector)
find_package(OpenCV REQUIRED)
find_package(X11 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${X11_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 14)

add_executable(ObjectDetector demo.cpp ObjectDetector.hpp ThresholdAlgorithm.hpp Filter.hpp Persistence.hpp)
target_link_libraries (ObjectDetector ${OpenCV_LIBS} ${X11_LIBRARIES})
