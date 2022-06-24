cmake_minimum_required(VERSION 3.20)

project(linalg)
file(GLOB LINALG_SOURCES ${CMAKE_CURRENT_LIST_DIR}/src/*.cc)

set(CMAKE_EXPORT_COMPILE_COMMANDS on)

message(${LINALG_SOURCES})
add_library(linalg ${LINALG_SOURCES})

find_package(OpenBlas CONFIG)

target_include_directories(linalg PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_link_libraries(linalg PRIVATE openblas)

# C++20 の有効化
enable_language(CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-std=c++20 ${CMAKE_CXX_FLAGS}")

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_LIST_DIR})
    find_package(GTest)

    enable_testing()
    add_executable(
        linalg_test
        test/hello_test.cc
        test/Matrix_test.cc
        test/lapacke_test.cc
        )
    target_link_libraries(
        linalg_test
        GTest::gtest_main
        linalg
        )

    include(GoogleTest)
    gtest_discover_tests(linalg_test)
endif()

