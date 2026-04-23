if (MATH_LIBRARY)
	link_libraries("${MATH_LIBRARY}")
endif()

add_executable(Sampler WIN32 MACOSX_BUNDLE Sampler.cpp ${SAMPLER_WINDOW} ${DIAL})


find_library(LIBRT rt)

target_link_libraries(Sampler PRIVATE ${CMAKE_DL_LIBS} Qt5::Widgets ${LIBRT} )

set_target_properties(Sampler PROPERTIES RUNTIME_OUTPUT_DIRECTORY ../)
# Automoc for Qt applications
set_property(TARGET Sampler PROPERTY AUTOMOC ON)

target_compile_options(Sampler PRIVATE "-Wall" "-g" "-std=c++11")

set(GUI_ONLY_BINARIES Sampler)

if (MSVC)
    # Tell MSVC to use main instead of WinMain
    set_target_properties(${GUI_ONLY_BINARIES} PROPERTIES LINK_FLAGS "/ENTRY:mainCRTStartup")
elseif (CMAKE_C_SIMULATE_ID STREQUAL "MSVC")
    # Tell Clang using MS CRT to use main instead of WinMain
    set_target_properties(${GUI_ONLY_BINARIES} PROPERTIES
        LINK_FLAGS "-Wl,/entry:mainCRTStartup")
endif()

if (APPLE)
    set_target_properties(Sampler PROPERTIES MACOSX_BUNDLE_BUNDLE_NAME "Qtwindow")
endif()
