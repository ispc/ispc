# Simple Modern CMake Example

This example demonstrates using ISPC as a CMake language (available in CMake 3.19+) with proper Ninja generator support.

## Key Features

- Uses `project(... LANGUAGES ISPC)` instead of custom macros
- Includes the `ISPCCompilerDepfileSupport` module for proper Ninja caching
- Demonstrates modern CMake best practices for ISPC

## Building

### With Ninja (requires the depfile support module):

```bash
cmake -B build -G Ninja
cmake --build build
./build/simple_modern
```

### With Unix Makefiles:

```bash
cmake -B build -G "Unix Makefiles"
cmake --build build
./build/simple_modern
```

## Verifying Caching

To verify that Ninja properly caches ISPC builds:

```bash
# Clean build
cmake -B build -G Ninja
cmake --build build

# Rebuild without changes - should do nothing
cmake --build build
```

Without the `ISPCCompilerDepfileSupport` module, the second build would recompile `simple.ispc` even though nothing changed.

## Comparison with Legacy Approach

The traditional approach uses custom CMake macros:
- `ispc_compile()` to compile ISPC files
- Manual source management
- See other examples in `examples/cpu/` for this approach

This modern approach:
- Uses native CMake ISPC language support
- Simpler CMakeLists.txt
- Better IDE integration
- Requires CMake 3.19+
