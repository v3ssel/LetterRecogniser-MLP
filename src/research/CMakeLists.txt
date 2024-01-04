cmake_minimum_required(VERSION 3.5)
project(MLPResearch VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# add_compile_options(-Wall -Werror -Wextra)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_subdirectory(../core ${CMAKE_BINARY_DIR}/core)

add_executable(${PROJECT_NAME} main_research.cpp)
target_link_libraries(${PROJECT_NAME} MLPCore)

set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/Release
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/Release
)

