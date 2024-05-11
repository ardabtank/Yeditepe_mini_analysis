#author : Serhat Istin
cmake_minimum_required(VERSION 3.16...3.24)
set(CMAKE_CXX_STANDARD 20)
project(yeditepe_mini_analysis)
find_package(ROOT REQUIRED COMPONENTS RIO Net)


set(PROJECT_INSTALL_DIR  ${CMAKE_BINARY_DIR}/install )

set(CMAKE_INSTALL_RPATH ${PROJECT_INSTALL_DIR}/lib )
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_subdirectory(src)
add_subdirectory(util)