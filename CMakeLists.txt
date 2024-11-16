cmake_minimum_required(VERSION 3.18)  # Atualizado para 3.18 para compatibilidade com o src/CMakeLists.txt
project(IrisLandmarkDemo)

# Define o padrão C++
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Encontrar pacotes
find_package(OpenCV REQUIRED)

# Incluir diretórios para OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})

# Adiciona o subdiretório src que contém seu executável
add_subdirectory(src)

# Mensagens para depuração
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# Remover ou comentar a linha abaixo, pois está incompleta e desnecessária
# target_link_libraries(${APP_NAME} PRIVATE)

