cmake_minimum_required(VERSION 3.12.0)
project(space_time_adaptivity LANGUAGES CXX C)

include(common/cmake-common.cmake)

add_executable(space_time_adaptivity src/main.cpp src/Heat.cpp src/prm_parse.cpp)
deal_ii_setup_target(space_time_adaptivity)
