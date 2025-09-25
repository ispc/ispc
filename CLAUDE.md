# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Intel® ISPC (Implicit SPMD Program Compiler) - high-performance SIMD compiler providing 3x+ speedup on vector architectures (x86 SSE/AVX, ARM NEON, Intel GPU).

## Build System

**CMake-based** (Unix Makefiles/Ninja)
- Build directory: `build/` (ask user permission before creating if missing)
- Generated compiler: `build/bin/ispc`

**Build commands:**
```bash
# Manual setup
mkdir build && cd build && cmake ../
make -j$(nproc)  # or: cmake --build . -- -j$(nproc) or ninja

# Automated setup
python3 scripts/quick-start-build.py
```

## Testing

**Lit tests** (`tests/lit-tests/`, 650+ tests): Compilation correctness
```bash
cd build/ && make check-all  # or: cmake --build . --target check-all or ninja check-all
```

**Functional tests** (`tests/func-tests/`, 1500+ tests): Runtime behavior
```bash
python3 scripts/run_tests.py
```

## Source Code Architecture

**Entry point**: `src/main.cpp`
**Key modules**: `ast`, `parse.yy` (parser), `lex.ll` (lexer), `expr`, `stmt`, `type`, `func`, `module`, `opt`, `target_*`, `builtins`, `llvmutil`
**Backend**: Heavy LLVM integration for x86/ARM/GPU code generation

### Code Conventions
- **C++17**, `l` prefix for static methods, `Assert()`/`AssertPos()` from `util.h`
- **ISPC**: Reference `docs/` for language spec, `.ispc` file extension

## Development Workflow

1. Make changes to `src/` or `builtins/`
2. Build: `make -j$(nproc)` (or cmake/ninja equivalents)
3. Run tests: `make check-all` (lit) and/or `python3 scripts/run_tests.py` (functional)

**Important**: Both build and tests take several minutes - use longer timeouts. Some tests may show UNSUPPORTED/XFAIL status normally.

**Testing strategy**: Lit tests for compilation/parsing changes, functional tests for runtime behavior changes.
