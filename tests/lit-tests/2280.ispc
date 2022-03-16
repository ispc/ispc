// RUN:     %{ispc} %s --target=host --nostdlib --emit-llvm-text -E                       -o -
// RUN:     %{ispc} %s --target=host --nostdlib --emit-llvm-text                          -o - | FileCheck %s -check-prefix=CHECK_PASS_0
// RUN:     %{ispc} %s --target=host --nostdlib --emit-llvm-text                          -o - | FileCheck %s -check-prefix=CHECK_PASS_2
// RUN: not %{ispc} %s --target=host --nostdlib                                 -DDO_ELSE 2>&1 | FileCheck %s -check-prefix=CHECK_ERROR
// RUN: not %{ispc} %s --target=host --nostdlib                  -E             -DDO_ELSE 2>&1 | FileCheck %s -check-prefix=CHECK_ERROR

// RUN:     %{ispc} %s --target=host --nostdlib --emit-llvm-text -E --ignorecpp           -o -
// RUN:     %{ispc} %s --target=host --nostdlib --emit-llvm-text    --ignorecpp           -o - | FileCheck %s -check-prefix=CHECK_PASS_0
// RUN:     %{ispc} %s --target=host --nostdlib --emit-llvm-text    --ignorecpp           -o - | FileCheck %s -check-prefix=CHECK_PASS_2
// RUN:     %{ispc} %s --target=host --nostdlib --emit-llvm-text    --ignorecpp -DDO_ELSE -o - | FileCheck %s -check-prefix=CHECK_PASS_1
// RUN:     %{ispc} %s --target=host --nostdlib --emit-llvm-text    --ignorecpp -DDO_ELSE -o - | FileCheck %s -check-prefix=CHECK_PASS_2
// RUN:     %{ispc} %s --target=host --nostdlib --emit-llvm-text -E --ignorecpp -DDO_ELSE -o -


// CHECK_PASS_0: ret i32 1337
// CHECK_PASS_1: ret i32 7331
// CHECK_PASS_2: ret i32 314159

// CHECK_ERROR:  DO_ELSE: Preprocessor #error


#define TESTVALUE_0 1337
#define TESTVALUE_1 7331
#define TESTVALUE_2 314159

#ifndef DO_ELSE
uniform int foo() { return TESTVALUE_0; }
#else
uniform int bar() { return TESTVALUE_1; }
#error "DO_ELSE: Preprocessor #error"
#endif // DO_ELSE

uniform int baz() { return TESTVALUE_2; }
