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
    demo.cpp
    ModelLoader.cpp
    DetectionPostProcess.cpp
    FaceDetection.cpp
    SignalProcessor.cpp
    evm.cpp
)
set(HEADERS
    ModelLoader.hpp
    DetectionPostProcess.hpp
    FaceDetection.hpp
    SignalProcessor.hpp
    evm.hpp
)

# Adiciona o executável
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Encontrar OpenCV (opcional, se necessário especificamente aqui)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Linkar bibliotecas
target_link_libraries(${PROJECT_NAME} PRIVATE 
    ${OpenCV_LIBS}
    tensorflow-lite
    dl
    pthread
    fftw3f  # Nome da biblioteca diretamente
    fftw3
)

