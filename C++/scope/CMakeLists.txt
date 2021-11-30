
# Enable faster instruction sets (SIMD/AVX)
set(ENABLE_FAST_INSTRUCTIONS ON CACHE BOOL "Enable faster instruction sets (SIMD/AVX)")
# Enable OpenMP (if available)
set(ENABLE_OPENMP OFF CACHE BOOL "Enable OpenMP (if available)")

if(${ENABLE_FAST_INSTRUCTIONS})
message(STATUS "Enabling SIMD/AVX instruction sets")
add_definitions(-march=native)
endif()

# PERFORMANCE IMPROVEMENTS
if(${ENABLE_OPENMP})
find_package(OpenMP)
if(OPENMP_FOUND)
message(STATUS "\nFound OpenMP! Turning on support for parallelization\n")
endif()
endif()

set(SCOPE_INCLUDE_DIR ${PROJECT_SOURCE_DIR})
set(SCOPE_INCLUDES ${SCOPE_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR} ${Boost_INCLUDE_DIRS} CACHE INTERNAL "")

set(SCOPE_SRC_DIR ${PROJECT_SOURCE_DIR})

file(GLOB_RECURSE SCOPE_SRCS ${SCOPE_SRC_DIR} "*.cpp")

file(GLOB_RECURSE SCOPE_HDRS ${SCOPE_SRC_DIR} "*.h")

add_library(SCOPE SHARED ${SCOPE_HDRS} ${SCOPE_SRCS})
target_include_directories(SCOPE PUBLIC ${SCOPE_INCLUDE_DIR})
target_link_libraries(SCOPE PUBLIC ${SCOPE_LIBRARIES})

if(OPENMP_FOUND)
# Add additional compilation flags to enable OpenMP support
set_target_properties(SCOPE PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
target_link_libraries(SCOPE PUBLIC omp)
endif()

# Add add entry for this project into CMake's package registry, so that this project can be found by other CMake projects
export(PACKAGE SCOPE)
# Create a configuration file for this project, so that it can be imported by other CMake projects
export(TARGETS SCOPE FILE SCOPEConfig.cmake)
