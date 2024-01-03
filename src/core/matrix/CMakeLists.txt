cmake_minimum_required(VERSION 3.5)

project(MatrixModel VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Werror -Wextra)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
option(COVERAGE OFF)

add_library(MatrixLib STATIC
    libmatrix/matrix.cc
    libmatrix/matrix.h
)

if (COVERAGE)
    get_target_property(lib_flags MatrixLib COMPILE_OPTIONS)
    list(REMOVE_ITEM lib_flags "--coverage")
    list(REMOVE_ITEM lib_flags "-fno-elide-constructors")
    set_property(TARGET MatrixLib PROPERTY COMPILE_OPTIONS ${lib_flags})
endif()

set(${PROJECT_NAME}_SOURCES 
    MatrixModel.cpp
    MatrixModel.h
    MatrixLayer.cpp
    MatrixLayer.h
)

add_library(${PROJECT_NAME} STATIC ${${PROJECT_NAME}_SOURCES})

target_link_libraries(${PROJECT_NAME} MatrixLib)
