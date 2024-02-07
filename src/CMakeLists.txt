cmake_minimum_required(VERSION 3.5)
project(LetterRecogniserMainList VERSION 0.1 LANGUAGES CXX)

option(INSTALLING OFF)
option(TESTING OFF)
option(COVERAGE OFF)
option(RESEARCH OFF)

if (INSTALLING)
    add_subdirectory(gui)
endif()

if (TESTING AND (NOT COVERAGE))
    add_subdirectory(tests)
endif()

if (COVERAGE)
    add_subdirectory(tests)
endif()

if (RESEARCH)
    add_subdirectory(research)
endif()
