cmake_minimum_required(VERSION 3.18)
project(demo)

# Include directories for TensorFlow Lite
set(TFLITE_DIR "/home/aldo/tensorflow/tensorflow/lite")
include_directories(${TFLITE_DIR})
include_directories(${TFLITE_DIR}/tools/make/downloads/flatbuffers/include)
link_directories(${TFLITE_DIR}/tools/make/gen/linux_aarch64/lib)

# Source and header files
set(SOURCES
    demo.cpp
    ModelLoader.cpp
    DetectionPostProcess.cpp
    FaceDetection.cpp
    evm.cpp
)
set(HEADERS
    ModelLoader.hpp
    DetectionPostProcess.hpp
    FaceDetection.hpp
    evm.hpp
)

# Add executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Include FFTW directories
include_directories(/usr/local/include)
link_directories(/usr/local/lib)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE 
    ${OpenCV_LIBS}
    tensorflow-lite
    dl
    pthread
    fftw3  # Use the library name directly
)

