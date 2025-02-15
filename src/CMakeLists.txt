cmake_minimum_required(VERSION 3.18)
project(demo)

# Inclua os diretórios necessários para TensorFlow Lite
set(TFLITE_DIR "/home/aldo/tensorflow/tensorflow/lite")
include_directories(
    ${TFLITE_DIR}
    ${TFLITE_DIR}/tools/make/downloads/flatbuffers/include
    /usr/local/include  # FFTW
)

link_directories(
    ${TFLITE_DIR}/tools/make/gen/linux_aarch64/lib
    /usr/local/lib  # FFTW
)

# Arquivos fonte e cabeçalhos
set(SOURCES
    demo_dlib.cpp
    ModelLoader.cpp
    DetectionPostProcess.cpp
    FaceDetection.cpp
    SignalProcessor.cpp
    evm.cpp
    Database.cpp
)
set(HEADERS
    ModelLoader.hpp
    DetectionPostProcess.hpp
    FaceDetection.hpp
    SignalProcessor.hpp
    evm.hpp
    Database.hpp
)

# Adiciona o executável
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

set(OpenCV_DIR "/usr/lib/aarch64-linux-gnu/cmake/opencv4") 
# Encontrar OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find SQLite3 using pkg-config
find_package(PkgConfig REQUIRED)
pkg_search_module(SQLITE3 REQUIRED sqlite3)

if(SQLITE3_FOUND)
    message(STATUS "SQLite3 found: ${SQLITE3_INCLUDE_DIRS}")
    message(STATUS "SQLite3 libraries: ${SQLITE3_LIBRARIES}")
    target_include_directories(${PROJECT_NAME} PRIVATE ${SQLITE3_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${SQLITE3_LIBRARIES})
else()
    message(FATAL_ERROR "SQLite3 not found")
endif()

find_package(dlib REQUIRED)
include_directories(${dlib_INCLUDE_DIRS})


# Linkar bibliotecas
target_link_libraries(${PROJECT_NAME} PRIVATE 
    ${OpenCV_LIBS}
    tensorflow-lite
    dl
    pthread
    fftw3f  # Nome da biblioteca diretamente
    fftw3
    dlib::dlib
)

