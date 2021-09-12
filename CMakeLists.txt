cmake_minimum_required (VERSION 3.19)
project(streaming-project)

set (${PROJECT_NAME}_MAJOR_VERSION 1)
set (${PROJECT_NAME}_MINOR_VERSION 0)
set (${PROJECT_NAME}_PATCH_VERSION 0)

# We don't want compiler warnings from included libraries
set(CONAN_SYSTEM_INCLUDES ON)
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()
# conan setup is done

if ( CMAKE_COMPILER_IS_GNUCC )
    add_compile_options(-Wall -Wextra)
endif()

foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string (TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
    set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${PROJECT_SOURCE_DIR}/bin)
    set( CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${PROJECT_SOURCE_DIR}/bin)
    set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${PROJECT_SOURCE_DIR}/lib)
endforeach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})

set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

add_subdirectory(${PROJECT_SOURCE_DIR}/stereom)
