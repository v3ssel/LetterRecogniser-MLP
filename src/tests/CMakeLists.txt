cmake_minimum_required(VERSION 3.5)

project(MLPTests VERSION 0.1 LANGUAGES CXX)
option(COVERAGE OFF)

if (COVERAGE)
    add_compile_options(--coverage -fno-elide-constructors)
endif()

if (APPLE OR UNIX)
    add_compile_options(-fPIC)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# add_compile_options(-Wall -Werror -Wextra)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
add_definitions(-DGTEST_LANGUAGE_CXX17)
enable_testing()

set(${PROJECT_NAME}_SOURCES
    main_test.cpp
    EmnistDatasetReaderTests.cpp
    EmnistMLPTrainerTests.cpp
    FileMLPSerializerTests.cpp
    GraphModelTests.cpp
    MatrixModelTests.cpp
    MultilayerPerceptronTests.cpp
    ModelBuilderTests.cpp
)

add_subdirectory(../core ${CMAKE_CURRENT_BINARY_DIR}/core)

add_executable(${PROJECT_NAME} ${${PROJECT_NAME}_SOURCES})

target_link_libraries(${PROJECT_NAME}
    MLPCore
    ${GTEST_LIBRARIES}
    ${GTEST_MAIN_LIBRARIES}
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/Testing
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/Testing
)
file(COPY assets DESTINATION ${CMAKE_SOURCE_DIR}/bin/Testing)

add_test(tests ${PROJECT_NAME})
