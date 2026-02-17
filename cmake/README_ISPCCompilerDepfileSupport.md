# ISPC Compiler Depfile Support for Ninja

## Problem

When using ISPC as a language with CMake 3.19+ and the Ninja generator, ISPC source files are always rebuilt even when they haven't changed. This is because CMake's built-in ISPC compiler module (`Compiler/Intel-ISPC.cmake`) only enables dependency file (depfile) support for Makefile generators, not for Ninja.

## Solution

ISPC provides a `ISPCCompilerDepfileSupport.cmake` module that enables depfile support for the Ninja generator. This allows Ninja to properly track dependencies and avoid unnecessary rebuilds.

## Usage

### For projects using ISPC as a CMake language

Add this after your `project()` command:

```cmake
cmake_minimum_required(VERSION 3.19)
project(myproject LANGUAGES CXX ISPC)

# Enable ISPC depfile support for Ninja
include(ISPCCompilerDepfileSupport)

# Rest of your CMakeLists.txt...
add_executable(myapp main.cpp kernel.ispc)
set_target_properties(myapp PROPERTIES ISPC_INSTRUCTION_SETS "avx2-i32x8;sse4-i32x4")
```

### Module location

The module is installed with ISPC in:
- `<install_prefix>/lib/cmake/ispc/ISPCCompilerDepfileSupport.cmake`

If CMake can't find the module automatically, you can add the path to CMAKE_MODULE_PATH:

```cmake
list(APPEND CMAKE_MODULE_PATH "<ispc_install_path>/lib/cmake/ispc")
include(ISPCCompilerDepfileSupport)
```

## When is this needed?

This workaround is needed when:
- Using CMake 3.19 or later
- Using ISPC as a language (`project(... LANGUAGES ISPC)`)
- Using the Ninja generator (`cmake -G Ninja ...`)
- Experiencing unnecessary rebuilds of ISPC files

## When can this be removed?

This module can be removed once CMake's upstream `Compiler/Intel-ISPC.cmake` is updated to support the Ninja generator natively.

## Technical details

The fix sets these CMake variables after the ISPC language is enabled:
- `CMAKE_ISPC_DEPFILE_FORMAT = "gcc"` - Tells CMake that ISPC uses GCC-style depfiles
- `CMAKE_ISPC_DEPENDS_USE_COMPILER = TRUE` - Tells CMake to use the compiler for dependencies
- `CMAKE_DEPFILE_FLAGS_ISPC` - Already set by CMake to "-M -MT <DEP_TARGET> -MF <DEP_FILE>"

These enable Ninja to use ISPC's built-in dependency generation (`-M -MT -MF` flags) instead of relying on CMake's dependency scanner.
