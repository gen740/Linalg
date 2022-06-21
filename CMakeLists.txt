cmake_minimum_required(VERSION 3.20)

project(linalg)
file(GLOB LINALG_SOURCES ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)

set(CMAKE_EXPORT_COMPILE_COMMANDS on)

message(${LINALG_SOURCES})
add_library(linalg ${LINALG_SOURCES})

target_include_directories(linalg PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)


# C++20 の有効化
enable_language(CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-std=c++20 ${CMAKE_CXX_FLAGS}")

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_LIST_DIR})
    message("this is root")
endif()

