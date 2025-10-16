# ISPC Project

## Project Overview

ISPC (Intel SPMD Program Compiler) is a performance-oriented compiler for
vectorized parallel programming.

## Build System

**CMake-based** (Unix Makefiles/Ninja)
- Build directory: `build/` (ask user permission before creating if missing)
- Generated compiler: `build/bin/ispc`

**Build commands:**
```bash
cmake -B build
cmake --build build -j $(nproc)
```

**Test commands:**
Lit tests:
```bash
cmake --build build --target check-all -j $(nproc)
```

To test the specific test, run:
```bash
TEST=/full/path/test.ispc cmake --build build --target check-one -j $(nproc)
```

Functional tests:
```bash
PATH=`pwd`/build/bin:$PATH ./scripts/run_tests.py --target=avx2-i32x8
```

**Important**: Both build and tests take several minutes - use longer timeouts.
Some tests may show UNSUPPORTED/XFAIL status normally.

**Testing strategy**: Lit tests for compilation/parsing changes,
functional tests for runtime behavior changes.

## Source Code Architecture
### Key directories:

- **src/**: Core compiler source (C++ frontend, AST, codegen, optimizations)
- **stdlib/**: ISPC standard library (built-in functions, target-specific implementations)
- **builtins/**: Low-level target-specific LLVM bitcode implementations
- **tests/**: Comprehensive test suite with functional tests
- **examples/**: Sample programs demonstrating ISPC usage
- **benchmarks/**: Performance benchmarks organized by complexity
- **ispcrt/**: Runtime library for host-device interaction
- **docs/**: Documentation and design specifications
- **scripts/**: Build automation and testing utilities

**Backend**: Heavy LLVM integration for x86/ARM/GPU code generation

### Code Conventions
- **C++17**, `l` prefix for static methods, `Assert()`/`AssertPos()` from `util.h`
- **ISPC**: Reference `docs/` for language spec, `.ispc` file extension

## Common Workflows

**Debugging a test failure:**
1. Run the specific test: `TEST=/full/path/test.ispc cmake --build build --target check-one -j $(nproc)`
2. Examine generated IR: `build/bin/ispc test.ispc -o test.ll --emit-llvm-text`
3. Check assembly: `build/bin/ispc test.ispc -o test.s --emit-asm`

**Investigating codegen/optimization issues:**
- Use `--debug-phase=first:last --dump-file=dbg` to dump IR after each phase to `dbg` folder

## Important Instructions
- Do only what is asked; nothing more, nothing less
- ALWAYS prefer editing existing files over creating new ones
- NEVER proactively create documentation (*.md) or README files unless explicitly requested
- NEVER leave trailing spaces in files
- Only use emojis if explicitly requested
