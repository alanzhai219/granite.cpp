cmake_minimum_required(VERSION 3.20)
project(granite LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g -O0)
else()
    add_compile_options(-O2)
endif()

add_executable(granite
    main.cpp
    csrc/tokenizer.cpp
    csrc/weights.cpp
    csrc/ops.cpp
    csrc/modeling.cpp
)
