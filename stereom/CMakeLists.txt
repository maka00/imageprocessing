cmake_minimum_required(VERSION 3.19)
project(stereom)


set(SOURCES main.cpp ssd_match.h ssd_match.cpp)

find_package(Threads)
find_package(PkgConfig REQUIRED)
find_package(Boost 1.76.0 COMPONENTS filesystem iostreams system program_options timer REQUIRED)
find_package( OpenCV REQUIRED )
find_package(PCL 1.11 REQUIRED COMPONENTS common io)
pkg_check_modules(GStreamer REQUIRED IMPORTED_TARGET gstreamer-1.0)
pkg_check_modules(GLib REQUIRED IMPORTED_TARGET glib-2.0)

enable_testing()

if (CMAKE_BUILD_TYPE MATCHES Debug)
    add_definitions(-D_DEBUG=1)
endif()
add_executable(${PROJECT_NAME} ${SOURCES})

set_target_properties(${PROJECT_NAME}
        PROPERTIES
        DEBUG_POSTFIX "d"
        CXX_STANDARD 20
        )

target_include_directories(${PROJECT_NAME}
        PUBLIC
        ${PROJECT_SOURCE_DIR}

        PRIVATE
        ${PCL_INCLUDE_DIRS}
        ${BOOST_INCLUDE_DIRS}
        ${CMAKE_CURRENT_BINARY_DIR}

       )
# CMake 3.14 onwards
target_link_directories(${PROJECT_NAME}
        PRIVATE
        ../lib
)

target_link_libraries(${PROJECT_NAME}
        PUBLIC
        ${CONAN_LIBS}
        Threads::Threads
        Boost::program_options
        Boost::filesystem
        Boost::timer
        opencv_core
        opencv_videoio
        opencv_highgui
        opencv_calib3d
        opencv_ximgproc
        ${PCL_LIBRARIES}
        )
target_compile_features(${PROJECT_NAME}
        PUBLIC
        cxx_std_20
        )

target_compile_options(${PROJECT_NAME}
        PUBLIC
        PRIVATE
        -Wall -Wextra
        )
