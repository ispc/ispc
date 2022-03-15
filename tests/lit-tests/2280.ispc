// RUN:     %{ispc} --target=host --nowrap --nostdlib -g -O0 -o %t.o %s -E
// RUN:     %{ispc} --target=host --nowrap --nostdlib -g -O0 -o %t.o %s --onlycpp

// RUN: not %{ispc} --target=host --nowrap --nostdlib -g -O0         %s                     2>&1 | FileCheck %s -check-prefix=CHECK_ERROR_0
// RUN: not %{ispc} --target=host --nowrap --nostdlib -g -O0         %s                     2>&1 | FileCheck %s -check-prefix=CHECK_ERROR_0

// RUN: not %{ispc} --target=host --nowrap --nostdlib -g -O0         %s           -DDO_ELSE 2>&1 | FileCheck %s -check-prefix=CHECK_ERROR_1
// RUN: not %{ispc} --target=host --nowrap --nostdlib -g -O0         %s           -DDO_ELSE 2>&1 | FileCheck %s -check-prefix=CHECK_ERROR_1

// RUN: not %{ispc} --target=host --nowrap --nostdlib -g -O0         %s -E        -DDO_ELSE 2>&1 | FileCheck %s -check-prefix=CHECK_ERROR_1
// RUN: not %{ispc} --target=host --nowrap --nostdlib -g -O0         %s --onlycpp -DDO_ELSE 2>&1 | FileCheck %s -check-prefix=CHECK_ERROR_1

// REQUIRES: X86_ENABLED


// CHECK_ERROR_0: Error:

// CHECK_ERROR_1: DO_ELSE: Preprocessor #error


#ifndef DO_ELSE
void foo();
#else
int i;
void bar();
#error "DO_ELSE: Preprocessor #error"
#endif // DO_ELSE

// Should not compile
void baz() { return 0; }
