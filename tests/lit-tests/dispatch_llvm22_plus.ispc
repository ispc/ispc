// RUN: %{ispc} --pic --target=sse2-i32x4,avx512icl-x4,avx512spr-x4,avx512gnr-x4,avx10.2dmr-x4,avx10.2nvl-x4 -h %t.h %s -o %t.o -g
// RUN: %{cc} -x c -c %s -o %t.c.o --include %t.h -g
// RUN: %{cc} %t.c.o -o %t.c.bin -g -rdynamic %t.o %t_sse2.o %t_avx512icl.o %t_avx512spr.o %t_avx512gnr.o %t_avx10_2dmr.o %t_avx10_2nvl.o

// RUN: sde -mrm -- %t.c.bin | FileCheck %s --check-prefix=CHECK-SSE2
// RUN: sde -icl -- %t.c.bin | FileCheck %s --check-prefix=CHECK-ICL
// RUN: sde -spr -- %t.c.bin | FileCheck %s --check-prefix=CHECK-SPR
// RUN: sde -gnr -- %t.c.bin | FileCheck %s --check-prefix=CHECK-GNR
// RUN: sde -dmr -- %t.c.bin | FileCheck %s --check-prefix=CHECK-DMR
// RUN: sde -nvl -- %t.c.bin | FileCheck %s --check-prefix=CHECK-NVL

// REQUIRES: LINUX_HOST && X86_ENABLED && SDE_INSTALLED && LLVM_22_0+

// CHECK-SSE2: ispc_foo_sse2
// CHECK-ICL: ispc_foo_avx512icl
// CHECK-SPR: ispc_foo_avx512spr
// CHECK-GNR: ispc_foo_avx512gnr
// CHECK-DMR: ispc_foo_avx10_2dmr
// CHECK-NVL: ispc_foo_avx10_2nvl

#ifdef ISPC
extern "C" void print_stack_trace();
void check_isa() {
  print_stack_trace();
  print("check_isa");
}
export void ispc_foo() {
  check_isa();
}
#else
#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>
void print_stack_trace() {
    const int max_frames = 64;
    void *buffer[max_frames];
    int num_frames = backtrace(buffer, max_frames);
    char **symbols = backtrace_symbols(buffer, num_frames);
    if (symbols == NULL) {
        perror("backtrace_symbols");
        exit(EXIT_FAILURE);
    }
    printf("Stack trace:\n");
    for (int i = 0; i < num_frames; i++) {
        printf("%s\n", symbols[i]);
    }
    free(symbols);
}
extern void ispc_foo();
int main() {
    ispc_foo();
    return 0;
}
#endif // ISPC
