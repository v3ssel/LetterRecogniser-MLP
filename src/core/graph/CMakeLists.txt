cmake_minimum_required(VERSION 3.5)

project(GraphModel VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# add_compile_options(-Wall -Werror -Wextra)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(${PROJECT_NAME}_SOURCES 
    GraphModel.cpp
    GraphModel.h
    GraphLayer.cpp
    GraphLayer.h
    GraphNode.h
)

add_library(${PROJECT_NAME} STATIC ${${PROJECT_NAME}_SOURCES})
