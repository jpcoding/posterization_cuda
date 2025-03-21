cmake_minimum_required(VERSION 3.18)
project(posterization_gpu LANGUAGES CUDA CXX C VERSION 0.10.0)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)



include_directories("/home/jp/git/SZ3/install/include/")

add_subdirectory(test)