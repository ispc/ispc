# ISPC Project

## Project Overview

ISPC (Intel SPMD Program Compiler) is a performance-oriented compiler for
vectorized parallel programming.

## Build System

**CMake-based** (Unix Makefiles/Ninja)
- Build directory: `build/`
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

**Implementing a feature or fixing a bug:**
1. If GitHub issue: use `gh issue view <number>` to get details
2. Explain the problem and reproduce it (use reproducer from issue if available)
3. Search codebase for relevant files (use `pattern-finder` agent to discover existing patterns and similar implementations)
4. Create a fix plan
5. Implement the fix
6. Write regression tests in `tests/lit-tests/` (use `ispc-lit-tests` skill) and/or functional tests in `tests/func-tests/`
7. Verify the fix
8. Run `code-review` agent to review the changes and address its feedback
9. Run precommit checks (see Precommit Rules below)
10. Commit with message: `Fix #<issue_number>: <summary>` (for issues) or descriptive summary

**Debugging a test failure:**
1. Run the specific test: `TEST=/full/path/test.ispc cmake --build build --target check-one -j $(nproc)`
2. Examine generated IR: `build/bin/ispc test.ispc -o test.ll --emit-llvm-text`
3. Check assembly: `build/bin/ispc test.ispc -o test.s --emit-asm`

**Investigating codegen/optimization issues:**
- Use `--debug-phase=first:last --dump-file=dbg` to dump IR after each phase to `dbg` folder

**Working with builtins:**
Use the `ispc-builtins` skill when modifying files in `builtins/` directory.

## Precommit Rules (MANDATORY)

**Before every commit, you MUST complete ALL of these checks:**

1. **Copyright year**: Verify the year is 2026 in copyright strings of modified files
2. **Code formatting**: Run `clang-format -i` on ALL modified C/C++/header files
3. **Lit tests**: Run `cmake --build build --target check-all -j $(nproc)` and verify tests pass
4. **Functional tests** (if runtime behavior changed): Run `PATH=$(pwd)/build/bin:$PATH ./scripts/run_tests.py --target=avx2-i32x8`

**Do not commit until all checks pass. Do not skip any of these steps.**

## Important Instructions
- Do only what is asked; nothing more, nothing less
- ALWAYS prefer editing existing files over creating new ones
- NEVER proactively create documentation (*.md) or README files unless explicitly requested
- NEVER leave trailing spaces in files
- Only use emojis if explicitly requested
- Use GitHub CLI (`gh`) for all GitHub-related tasks (issues, PRs, etc.)
- Stop and ask if anything is unclear or a step fails
