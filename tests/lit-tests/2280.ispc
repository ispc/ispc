// RUN:     %{ispc} %s      --target=host --nostdlib -E                                                              | FileCheck %s.filecheck -check-prefix=CHECK_PASS_0
// RUN:     %{ispc} %s -o - --target=host --nostdlib    --emit-llvm-text                                             | FileCheck %s.filecheck -check-prefix=CHECK_PASS_1
// RUN: not %{ispc} %s      --target=host --nostdlib -E                                               -DDO_ELSE 2>&1 | FileCheck %s.filecheck -check-prefix=CHECK_ERROR
// RUN: not %{ispc} %s -o - --target=host --nostdlib    --emit-llvm-text                              -DDO_ELSE 2>&1 | FileCheck %s.filecheck -check-prefix=CHECK_ERROR

// RUN:     %{ispc} %s      --target=host --nostdlib -E                  --ignore-preprocessor-errors                | FileCheck %s.filecheck -check-prefix=CHECK_PASS_0
// RUN:     %{ispc} %s -o - --target=host --nostdlib    --emit-llvm-text --ignore-preprocessor-errors                | FileCheck %s.filecheck -check-prefix=CHECK_PASS_1
// RUN:     %{ispc} %s      --target=host --nostdlib -E                  --ignore-preprocessor-errors -DDO_ELSE      | FileCheck %s.filecheck -check-prefix=CHECK_PASS_2
// RUN:     %{ispc} %s -o - --target=host --nostdlib    --emit-llvm-text --ignore-preprocessor-errors -DDO_ELSE      | FileCheck %s.filecheck -check-prefix=CHECK_PASS_3

// RUN:     %{ispc} %s -o %t.i    --target=host --nostdlib -E
// RUN:     cat %t.i    | FileCheck %s.filecheck -check-prefix=CHECK_PASS_0
// RUN:     %{ispc} %s -o %t.ispi --target=host --nostdlib -E
// RUN:     cat %t.ispi | FileCheck %s.filecheck -check-prefix=CHECK_PASS_0

// RUN:     %{ispc} %s -o %t.j    --target=host --nostdlib -E 2>&1 | FileCheck %s.filecheck -check-prefix=CHECK_WARN
// RUN:     %{ispc} %s -o %t.ispj --target=host --nostdlib -E 2>&1 | FileCheck %s.filecheck -check-prefix=CHECK_WARN


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
