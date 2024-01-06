cmake_minimum_required(VERSION 3.5)

project(Controller VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# add_compile_options(-Wall -Werror -Wextra)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(${PROJECT_NAME}_SOURCES 
    controller.cpp
    controller.h
)

add_subdirectory(../core ${CMAKE_CURRENT_BINARY_DIR}/core)

add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SOURCES})

target_link_libraries(${PROJECT_NAME}
    MLPCore
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/Release
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/Release
)
