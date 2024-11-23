cmake_minimum_required(VERSION 3.16)
project(c_neural_network LANGUAGES C)

set(CMAKE_CXX_FLAGS "--std=c99")

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(app)
