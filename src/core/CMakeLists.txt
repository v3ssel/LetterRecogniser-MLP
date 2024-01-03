cmake_minimum_required(VERSION 3.5)

project(MLPCore VERSION 0.1 LANGUAGES CXX)
option(COVERAGE OFF)
option(TESTING OFF)

set(LIBGCOV "")
set(OUTPUT_PATH "bin/Release")

if (COVERAGE)
    set(LIBGCOV "gcov")
endif()

if (TESTING)
    set(OUTPUT_PATH "bin/Testing")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# add_compile_options(-Wall -Werror -Wextra)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(${PROJECT_NAME}_SOURCES
    MultilayerPerceptron.cpp
    MultilayerPerceptron.h
    ModelBuilder.cpp
    ModelBuilder.h
    MLPModel.h
    MLPSerializer.h
    MLPTrainer.h
    MLPTestMetrics.h
    MLPTrainStages.h
)

add_subdirectory(graph)
add_subdirectory(matrix)
add_subdirectory(serializer)
add_subdirectory(training)

add_library(${PROJECT_NAME} SHARED
    ${${PROJECT_NAME}_SOURCES}
)
    
target_link_libraries(${PROJECT_NAME}
    GraphModel
    MatrixModel
    FileMLPSerializer
    EmnistMLPTrainer
    ${LIBGCOV}
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/${OUTPUT_PATH}
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/${OUTPUT_PATH}
)
