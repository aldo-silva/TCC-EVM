cmake_minimum_required(VERSION 3.10)
project(IrisLandmarkDemo)

# Define the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Finding packages
find_package(OpenCV REQUIRED)

# Include directories for CUDA, OpenCV
include_directories(${OpenCV_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS})

# Define your application's executable
set(APP_NAME "demo")
add_subdirectory(src)  # Adiciona o subdiretório que contém mais CMakeLists.txt ou código-fonte

# Print package versions to help with debugging and ensuring correct versions
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# Linking libraries
target_link_libraries(${APP_NAME} PRIVATE)


