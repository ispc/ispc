// RUN:     %{ispc} %s      --target=host --nostdlib -E                                                              | grep -v -e '^// ' | FileCheck %s -check-prefix=CHECK_PASS_0
// RUN:     %{ispc} %s -o - --target=host --nostdlib    --emit-llvm-text                                                                 | FileCheck %s -check-prefix=CHECK_PASS_1
// RUN: not %{ispc} %s      --target=host --nostdlib -E                                               -DDO_ELSE 2>&1 | grep -v -e '^// ' | FileCheck %s -check-prefix=CHECK_ERROR
// RUN: not %{ispc} %s -o - --target=host --nostdlib    --emit-llvm-text                              -DDO_ELSE 2>&1                     | FileCheck %s -check-prefix=CHECK_ERROR

// RUN:     %{ispc} %s      --target=host --nostdlib -E                  --ignore-preprocessor-errors                | grep -v -e '^// ' | FileCheck %s -check-prefix=CHECK_PASS_0
// RUN:     %{ispc} %s -o - --target=host --nostdlib    --emit-llvm-text --ignore-preprocessor-errors                                    | FileCheck %s -check-prefix=CHECK_PASS_1
// RUN:     %{ispc} %s      --target=host --nostdlib -E                  --ignore-preprocessor-errors -DDO_ELSE      | grep -v -e '^// ' | FileCheck %s -check-prefix=CHECK_PASS_2
// RUN:     %{ispc} %s -o - --target=host --nostdlib    --emit-llvm-text --ignore-preprocessor-errors -DDO_ELSE                          | FileCheck %s -check-prefix=CHECK_PASS_3


// CHECK_PASS_0: return 1337;
// CHECK_PASS_0: return 314159;

// CHECK_PASS_1: ret i32 1337
// CHECK_PASS_1: ret i32 314159

// CHECK_PASS_2: return 7331;
// CHECK_PASS_2: return 314159;

// CHECK_PASS_3: ret i32 7331
// CHECK_PASS_3: ret i32 314159

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
