// Runtime regression test for the AMX-capability requirement in the dispatcher.
//
// A binary built with an AMX-bearing server variant (avx512spr) but NO
// avx10.2nvl variant must NOT select the SPR variant on a Nova Lake (NVL)
// system. NVL implements AVX10.2 and full AVX-512, so the linear ">=" ISA walk
// alone (NVL=12 >= SPR=10) would select SPR and fault on its AMX instructions.
// The AMX requirement (__system_has_amx()) makes NVL fail the SPR candidate and
// fall back to the avx512icl variant instead. On a real SPR system the SPR
// variant is still selected, since SPR reports AMX.

// RUN: %{ispc} --pic --target=avx512icl-x4,avx512spr-x4 -h %t.h %s -o %t.o -g
// RUN: %{cc} -x c -c %s -o %t.c.o --include %t.h -g
// RUN: %{cc} %t.c.o -o %t.c.bin -g -rdynamic %t.o %t_avx512icl.o %t_avx512spr.o

// RUN: sde -spr -- %t.c.bin | FileCheck %s --check-prefix=CHECK-SPR
// RUN: sde -nvl -- %t.c.bin | FileCheck %s --check-prefix=CHECK-NVL

// REQUIRES: LINUX_HOST && X86_ENABLED && SDE_INSTALLED && LLVM_22_0+

// On SPR the SPR variant is selected (SPR has AMX).
// CHECK-SPR: ispc_foo_avx512spr
// On NVL the SPR variant must be rejected (no AMX); fall back to avx512icl.
// CHECK-NVL: ispc_foo_avx512icl

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
