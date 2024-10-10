// This test checks that the math library is correctly set based on the user
// provided command line option --math-lib.
// __math_lib value is defined in the core.isph file without the initialization
// when ISPC compiles stdlib bitcode files during ISPC build. Later, this value
// is initialized with the value provided by the user via macro definition in
// ISPC_MATH_LIB_VAL defined in module.cpp file (where preprocessor is
// initialized). This value is then used in the math functions to call the
// appropriate math library function.

// RUN: %{ispc} %s --target=host --nowrap 2>&1 --debug-phase=2:2 -o t.o | FileCheck %s --check-prefix=CHECK-ISPC
// RUN: %{ispc} %s --target=host --math-lib=default --nowrap 2>&1 --debug-phase=2:2 -o t.o | FileCheck %s --check-prefix=CHECK-ISPC
// RUN: %{ispc} %s --target=host --math-lib=fast --nowrap 2>&1 --debug-phase=2:2 -o t.o | FileCheck %s --check-prefix=CHECK-FAST
// RUN: %{ispc} %s --target=host --math-lib=svml --nowrap 2>&1 --debug-phase=2:2 -o t.o | FileCheck %s --check-prefix=CHECK-SVML
// RUN: %{ispc} %s --target=host --math-lib=system --nowrap 2>&1 --debug-phase=2:2 -o t.o | FileCheck %s --check-prefix=CHECK-SYSTEM

// RUN: %{ispc} %s --target=host --math-lib=default --nowrap 2>&1 --emit-llvm-text -o - | FileCheck %s --check-prefix=CHECK-FINAL-ISPC
// RUN: %{ispc} %s --target=host --math-lib=svml --nowrap 2>&1 --emit-llvm-text -o - | FileCheck %s --check-prefix=CHECK-FINAL-SVML
// RUN: %{ispc} %s --target=host --math-lib=system --nowrap 2>&1 --emit-llvm-text -o - | FileCheck %s --check-prefix=CHECK-FINAL-SYSTEM

// SVML is supported for x86 only
// REQUIRES: X86_ENABLED && !MACOS_HOST

// CHECK-ISPC: @__math_lib = constant i32 0
// CHECK-FINAL-ISPC-LABEL: @foo___vyf
// CHECK-FINAL-ISPC-NEXT: allocas:
// CHECK-FINAL-ISPC-NEXT: {{.*}} fmul
// CHECK-FINAL-ISPC-NEXT: {{.*}} @llvm.sqrt.
// CHECK-FINAL-ISPC-NEXT: ret

// CHECK-FAST: @__math_lib = constant i32 1

// CHECK-SVML: @__math_lib = constant i32 2
// CHECK-FINAL-SVML-LABEL: @foo___vyf
// CHECK-FINAL-SVML-NEXT: allocas:
// CHECK-FINAL-SVML-NEXT: {{.*}} fmul
// CHECK-FINAL-SVML-NEXT: {{.*}} @__svml_sqrtf
// CHECK-FINAL-SVML-NEXT: ret

// CHECK-SYSTEM: @__math_lib = constant i32 3
// CHECK-FINAL-SYSTEM: call float @sinf(
// TODO! tail call?

float foo(float a) {
    return sqrt(a / 2.);
}

float bar(float a) {
    return sin(a);
}
