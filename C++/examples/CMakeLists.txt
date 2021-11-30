find_package(Boost REQUIRED COMPONENTS program_options filesystem)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(run ./run.cpp)
target_link_libraries(run cnpy SCOPE ${Boost_LIBRARIES})
