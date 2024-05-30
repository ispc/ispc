// RUN: %{ispc} -DISPC --pic --target=sse2-i32x4,sse4.1-i32x4,avx1-i32x4,avx2-i32x4,avx2vnni-i32x4,avx512knl-x16,avx512icl-x4,avx512skx-x4,avx512spr-x4 -h %t.h %s -o %t.o -g
// RUN: %{cc} -x c -c %s -o %t.c.o --include %t.h -g
// RUN: %{cc} %t.c.o -o %t.c.bin -g -rdynamic %t.o %t_sse2.o %t_sse4.o %t_avx.o %t_avx2.o %t_avx2vnni.o %t_avx512knl.o %t_avx512icl.o %t_avx512skx.o %t_avx512spr.o

// RUN: sde -mrm -- %t.c.bin | FileCheck %s --check-prefix=CHECK-SSE2
// RUN: sde -pnr -- %t.c.bin | FileCheck %s --check-prefix=CHECK-SSE4
// RUN: sde -snb -- %t.c.bin | FileCheck %s --check-prefix=CHECK-AVX1
// RUN: sde -hsw -- %t.c.bin | FileCheck %s --check-prefix=CHECK-AVX2
// RUN: sde -adl -- %t.c.bin | FileCheck %s --check-prefix=CHECK-AVX2VNNI
// RUN: sde -skx -- %t.c.bin | FileCheck %s --check-prefix=CHECK-SKX
// RUN: sde -knl -- %t.c.bin | FileCheck %s --check-prefix=CHECK-KNL
// RUN: sde -icl -- %t.c.bin | FileCheck %s --check-prefix=CHECK-ICL
// RUN: sde -spr -- %t.c.bin | FileCheck %s --check-prefix=CHECK-SPR

// REQUIRES: LINUX_HOST && X86_ENABLED && SDE_INSTALLED

// CHECK-SSE2: ispc_foo_sse2
// CHECK-SSE4: ispc_foo_sse4
// CHECK-AVX1: ispc_foo_avx
// CHECK-AVX2: ispc_foo_avx2
// CHECK-AVX2VNNI: ispc_foo_avx2vnni
// CHECK-SKX: ispc_foo_avx512skx
// CHECK-KNL: ispc_foo_avx512knl
// CHECK-ICL: ispc_foo_avx512icl
// CHECK-SPR: ispc_foo_avx512spr

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

