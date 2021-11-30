cmake_minimum_required(VERSION 3.1)

# PROJECT CONFIGURATION
project(SCOPE LANGUAGES C CXX VERSION 1.0.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON) # We require C++ 17

# Enable code profiling using gperftools
set(ENABLE_PROFILING OFF CACHE BOOL "Enable code profiling using gperftools")

if(${ENABLE_PROFILING})
message(STATUS "Enabling code profiling using Google Performance Tools")
set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} -lprofiler)
endif()

# Set build type to 'RelWithDebInfo' if one was not specified by the user
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS RelWithDebInfo Release Debug MinSizeRel)
  message(STATUS "Setting build type to ${CMAKE_BUILD_TYPE}, as none was specified\n")
else()
  message(STATUS "Building in ${CMAKE_BUILD_TYPE} mode\n")
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
add_definitions(-fstandalone-debug)
endif()

add_definitions(-march=native)

# Directory for built libraries
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/lib CACHE PATH "The directory in which to place the library built by this project")
# Directory for built executables
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin CACHE PATH "The directory in which to place executables built by this project")


set(SCOPE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/scope/ CACHE PATH "Path to top-level directory of scope (the one containing CMakeLists.txt)")
set(examples ${CMAKE_CURRENT_SOURCE_DIR}/examples/ CACHE PATH "Path to top-level directory of examples (the one containing CMakeLists.txt)")

add_definitions(-DRETRACT_TOL=0.5 -DINVERSE_OF_RETRACT_TOL=3.76 -DDRETRACT_TOL=0.5)

include_directories(./)

find_package(Eigen3 REQUIRED)

set(GLOG_LIBRARIES glog)

include_directories(${EIGEN3_INCLUDE_DIR})

set(SCOPE_LIBRARIES ${GLOG_LIBRARIES})

add_subdirectory(./cnpy)
add_subdirectory(${SCOPE_DIR})
add_subdirectory(${examples})
