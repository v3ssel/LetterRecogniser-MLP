cmake_minimum_required(VERSION 3.5)

project(EmnistMLPTrainer VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Werror -Wextra)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(${PROJECT_NAME}_SOURCES
    EmnistMLPTrainer.cpp
    EmnistMLPTrainer.h
    EmnistDatasetReader.cpp
    EmnistDatasetReader.h
    EmnistData.cpp
    EmnistData.h
    TFMetrics.h
)

add_library(${PROJECT_NAME} STATIC ${${PROJECT_NAME}_SOURCES})
