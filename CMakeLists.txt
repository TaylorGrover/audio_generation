cmake_minimum_required(VERSION 3.16...3.20 FATAL_ERROR)

project(Sampler VERSION 0.0.1 LANGUAGES C CXX)

if (POLICY CMP0069)
	cmake_policy(SET CMP0069 NEW)
endif()

if (POLICY CMP0077)
	cmake_policy(SET CMP0077 NEW)
endif()

# Targets organized into folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

string(COMPARE EQUAL "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_SOURCE_DIR}" Sampler)

set(CMAKE_CXX_COMPILER /usr/bin/g++-10)

option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

add_compile_options("-Wall" "-g" "-fPIC" "-DQT_NO_VERSION_TAGGING")

if (MATH_LIBRARY)
	link_libraries("${MATH_LIBRARY}")
endif()

include(GNUInstallDirs)
include(CMakeDependentOption)
include(CTest) # Include CTest https://cmake.org/cmake/help/book/mastering-cmake/chapter/Testing%20With%20CMake%20and%20CTest.html
enable_testing()

include_directories("${Sampler_SOURCE_DIR}/deps" "/usr/include/x86_64-linux-gnu/qt5")

find_package(Qt5 COMPONENTS Widgets REQUIRED)
find_package(Qt5 COMPONENTS Core REQUIRED)

set (CMAKE_CXX_FLAG_COVERAGE "${CMAKE_CXX_FLAGS_RELEASE} -fprofile-arcs -ftest-coverage")

set_target_properties(Qt5::Core PROPERTIES MAP_IMPORTED_CONFIG_COVERAGE "RELEASE")

if (Sampler_BUILD_WIN32)
    message(STATUS "Including Win32 support")
endif()
if (Sampler_BUILD_COCOA)
    message(STATUS "Including Cocoa support")
endif()
if (Sampler_BUILD_WAYLAND)
    message(STATUS "Including Wayland support")
endif()
if (Sampler_BUILD_X11)
    message(STATUS "Including X11 support")
endif()


#--------------------------------------------------------------------
# Apply Microsoft C runtime library option
# This is here because it also applies to tests and examples
#--------------------------------------------------------------------
if (MSVC AND NOT USE_MSVC_RUNTIME_LIBRARY_DLL)
    if (CMAKE_VERSION VERSION_LESS 3.15)
        foreach (flag CMAKE_C_FLAGS
                      CMAKE_C_FLAGS_DEBUG
                      CMAKE_C_FLAGS_RELEASE
                      CMAKE_C_FLAGS_MINSIZEREL
                      CMAKE_C_FLAGS_RELWITHDEBINFO)

            if (flag MATCHES "/MD")
                string(REGEX REPLACE "/MD" "/MT" ${flag} "${${flag}}")
            endif()
            if (flag MATCHES "/MDd")
                string(REGEX REPLACE "/MDd" "/MTd" ${flag} "${${flag}}")
            endif()

        endforeach()
    else()
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
endif()

set(SAMPLER_WINDOW "${Sampler_SOURCE_DIR}/deps/sampler_window.h"
                   "${Sampler_SOURCE_DIR}/deps/sampler_window.cpp")

set(DIAL "${Sampler_SOURCE_DIR}/deps/dial.h"
         "${Sampler_SOURCE_DIR}/deps/dial.cpp")

add_subdirectory(src)
