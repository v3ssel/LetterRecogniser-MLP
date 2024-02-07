cmake_minimum_required(VERSION 3.5)

project(LetterRecogniser VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Werror -Wextra)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)

set(app_icon "")

if (APPLE OR UNIX)
    add_compile_options(-fPIC)
endif()

if (APPLE)
    set(MACOSX_BUNDLE_ICON_FILE icon.icns)
    set(app_icon_macos "${CMAKE_CURRENT_SOURCE_DIR}/resources/icon.icns")
    set_source_files_properties(${app_icon_macos} PROPERTIES
            MACOSX_PACKAGE_LOCATION "Resources")

    set(app_icon ${app_icon_macos})
    
elseif(WIN32)
    set(app_icon_resource_windows "${CMAKE_CURRENT_SOURCE_DIR}/resources/resources.rc")
    set(app_icon ${app_icon_resource_windows})
endif()

add_subdirectory(../controller ${CMAKE_BINARY_DIR}/core)

set(PROJECT_SOURCES
        main.cpp
        letterrecogniserwindow.cpp
        letterrecogniserwindow.h
        letterrecogniserwindow.ui
        drawablewidget.h
        drawablewidget.cpp
        resultgraphwidget.h
        resultgraphwidget.cpp
        ${app_icon}
)

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(LetterRecogniser
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
)
else()
    add_executable(LetterRecogniser
        ${PROJECT_SOURCES}
)
endif()

target_link_libraries(LetterRecogniser PRIVATE
    Qt${QT_VERSION_MAJOR}::Widgets
    Controller
)

set_target_properties(LetterRecogniser PROPERTIES
    MACOSX_BUNDLE_GUI_IDENTIFIER my.example.com
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE

    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/Release
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/Release
)

install(TARGETS LetterRecogniser
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(LetterRecogniser)
endif()
