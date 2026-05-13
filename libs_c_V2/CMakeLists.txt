cmake_minimum_required(VERSION 3.16)

project(leq_levels_oct_weighting_C LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

pybind11_add_module(
    leq_levels_oct_weighting_C
    bindings.cpp
    acoustic_processor.cpp
)

target_compile_options(
    leq_levels_oct_weighting_C
    PRIVATE
        -O3
        -DNDEBUG
        -Wall
        -Wextra
        -fvisibility=hidden
)

# STM32MP13 / Cortex-A7 / armhf
target_compile_options(
    leq_levels_oct_weighting_C
    PRIVATE
        -mcpu=cortex-a7
        -mfpu=neon-vfpv4
        -mfloat-abi=hard
)