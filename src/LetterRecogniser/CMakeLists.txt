cmake_minimum_required(VERSION 3.5)

project(LetterRecogniser VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)

set(PROJECT_SOURCES
        main.cpp
        letterrecogniserwindow.cpp
        letterrecogniserwindow.h
        letterrecogniserwindow.ui
        drawablewidget.h
        drawablewidget.cpp
        resultgraphwidget.h
        resultgraphwidget.cpp
        ../controller/controller.h
        ../controller/controller.cpp
        ../core/MLPModel.h
        ../core/MLPSerializer.h
        ../core/MLPTestMetrics.h
        ../core/MLPTrainer.h
        ../core/MLPTrainStages.h
        ../core/MultilayerPerceptron.h
        ../core/MultilayerPerceptron.cpp
        ../core/ModelBuilder.h
        ../core/ModelBuilder.cpp
        ../core/serializer/FileMLPSerializer.h
        ../core/serializer/FileMLPSerializer.cpp
        ../core/training/EmnistMLPTrainer.h
        ../core/training/EmnistMLPTrainer.cpp
        ../core/training/EmnistDatasetReader.h
        ../core/training/EmnistDatasetReader.cpp
        ../core/training/EmnistData.h
        ../core/training/EmnistData.cpp
        ../core/training/TFMetrics.h
        ../core/matrix/MatrixModel.h
        ../core/matrix/MatrixModel.cpp
        ../core/matrix/MatrixLayer.h
        ../core/matrix/MatrixLayer.cpp
        ../core/graph/GraphModel.h
        ../core/graph/GraphModel.cpp
        ../core/graph/GraphLayer.h
        ../core/graph/GraphLayer.cpp
        ../core/matrix/libmatrix/matrix.h
        ../core/matrix/libmatrix/matrix.cc
)

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(LetterRecogniser
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
    )
# Define target properties for Android with Qt 6 as:
#    set_property(TARGET LetterRecogniser APPEND PROPERTY QT_ANDROID_PACKAGE_SOURCE_DIR
#                 ${CMAKE_CURRENT_SOURCE_DIR}/android)
# For more information, see https://doc.qt.io/qt-6/qt-add-executable.html#target-creation
else()
    if(ANDROID)
        add_library(LetterRecogniser SHARED
            ${PROJECT_SOURCES}
        )
# Define properties for Android with Qt 5 after find_package() calls as:
#    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
    else()
        add_executable(LetterRecogniser
            ${PROJECT_SOURCES}
        )
    endif()
endif()

target_link_libraries(LetterRecogniser PRIVATE Qt${QT_VERSION_MAJOR}::Widgets)

set_target_properties(LetterRecogniser PROPERTIES
    MACOSX_BUNDLE_GUI_IDENTIFIER my.example.com
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

install(TARGETS LetterRecogniser
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(LetterRecogniser)
endif()
