// RUN: %{ispc} %s --target=host --nostdlib --math-lib=fast --nowrap --emit-llvm-text -o - 2>&1 | FileCheck %s --check-prefix=CHECK_FAST
// RUN: %{ispc} %s --target=host --nostdlib --math-lib=default --nowrap --emit-llvm-text -o - 2>&1 | FileCheck %s --check-prefix=CHECK_ISPC

// Use constexpr with fast-math configuration and feed it into a template
// non-type parameter to verify constexpr + template integration.
constexpr uniform int fast_math_enabled() { return __math_lib == ISPC_MATH_LIB_ISPC_FAST_VAL; }

template <int FastMode> uniform int fast_mode_tag() { return FastMode; }

uniform int get_fast_mode_tag() { return fast_mode_tag<fast_math_enabled()>(); }

// CHECK_FAST-LABEL: @get_fast_mode_tag
// CHECK_FAST: ret i32 1

// CHECK_ISPC-LABEL: @get_fast_mode_tag
// CHECK_ISPC: ret i32 0
