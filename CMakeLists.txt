cmake_minimum_required(VERSION 3.15)
project(opencg VERSION 0.1.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Options
option(BUILD_PYTHON_BINDINGS "Build Python bindings using pybind11" ON)
option(BUILD_TESTS "Build C++ tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(USE_BOOST "Use Boost Graph Library for SPPRC (optional)" ON)

# Find Boost (optional - for r_c_shortest_paths)
if(USE_BOOST)
    find_package(Boost 1.70 QUIET)
    if(Boost_FOUND)
        message(STATUS "Boost found: ${Boost_VERSION} - Boost SPPRC enabled")
        include_directories(${Boost_INCLUDE_DIRS})
        add_definitions(-DHAS_BOOST)
    else()
        message(STATUS "Boost not found - building without Boost SPPRC (use pure Python fallback)")
    endif()
else()
    message(STATUS "Boost disabled by option - building without Boost SPPRC")
endif()

# Find packages
if(BUILD_PYTHON_BINDINGS)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG QUIET)

    if(NOT pybind11_FOUND)
        # Try to find pybind11 via Python
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE pybind11_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(pybind11_DIR)
            find_package(pybind11 CONFIG REQUIRED PATHS ${pybind11_DIR})
        else()
            message(FATAL_ERROR "pybind11 not found. Install with: pip install pybind11")
        endif()
    endif()
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)

# Core library (header-only for now, can become static/shared later)
add_library(opencg_core INTERFACE)
target_include_directories(opencg_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    # Find threading support
    find_package(Threads REQUIRED)

    # Core binding sources (always included)
    set(BINDING_SOURCES
        src/bindings/module.cpp
        src/bindings/label_bindings.cpp
        src/bindings/pricing_bindings.cpp
        src/bindings/parallel_pricing_bindings.cpp
    )

    # Add Boost SPPRC bindings if Boost is available
    if(Boost_FOUND)
        list(APPEND BINDING_SOURCES src/bindings/boost_spprc_bindings.cpp)
    endif()

    pybind11_add_module(_core ${BINDING_SOURCES})
    target_link_libraries(_core PRIVATE opencg_core Threads::Threads)

    # Install the module to the opencg package
    # For scikit-build-core, install directly to opencg
    install(TARGETS _core
        LIBRARY DESTINATION opencg
        COMPONENT python
    )
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()

    # Simple test executable (uses catch2 or similar if available)
    add_executable(test_label tests/cpp/test_label.cpp)
    target_link_libraries(test_label PRIVATE opencg_core)
    add_test(NAME test_label COMMAND test_label)

    add_executable(test_pricing tests/cpp/test_pricing.cpp)
    target_link_libraries(test_pricing PRIVATE opencg_core)
    add_test(NAME test_pricing COMMAND test_pricing)
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
    add_executable(bench_pricing benchmarks/bench_pricing.cpp)
    target_link_libraries(bench_pricing PRIVATE opencg_core)
endif()

# Print configuration
message(STATUS "OpenCG C++ Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Python bindings: ${BUILD_PYTHON_BINDINGS}")
message(STATUS "  Boost SPPRC: ${Boost_FOUND}")
message(STATUS "  Tests: ${BUILD_TESTS}")
message(STATUS "  Benchmarks: ${BUILD_BENCHMARKS}")
