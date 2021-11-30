find_package(ZLIB REQUIRED)
include_directories(${ZLIB_LIBRARIES})

add_library(cnpy SHARED cnpy.cpp)
target_link_libraries(cnpy ${ZLIB_LIBRARIES})

export(PACKAGE cnpy)
export(TARGETS cnpy FILE cnpyConfig.cmake)
