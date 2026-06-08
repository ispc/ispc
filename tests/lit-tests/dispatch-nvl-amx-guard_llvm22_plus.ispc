// Companion to dispatch-nvl-amx-guard.ispc, exercising target sets that
// include the LLVM-22+ avx10.2nvl / avx10.2dmr targets. Verifies that:
//  - when an NVL variant IS present, the NVL candidate (i=12) itself is
//    selected by the plain ">=" test and does NOT carry the not_nvl guard
//    (the guard is keyed on the candidate index, not on target-set membership),
//    while a co-present SPR candidate still carries the guard;
//  - the DMR candidate (i=13) carries no guard, since it is already excluded
//    for an NVL system by the ">=" test (12 >= 13 is false).

// RUN: %{ispc} %s -O0 --target=avx512spr-x16,avx10.2nvl-x16 --emit-llvm-text -o %t.ll --nostdlib
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=SPRNVL
// RUN: %{ispc} %s -O0 --target=sse2-i32x4,avx10.2dmr-x16 --emit-llvm-text -o %t.ll --nostdlib
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=DMR

// REQUIRES: X86_ENABLED && LLVM_22_0+ && !MACOS_HOST

// NVL candidate (12) is selected by ">=" and must NOT get the not_nvl guard.
// SPRNVL:      %isa_ok = icmp sge i32 %{{.*}}, 12
// SPRNVL-NEXT: br i1 %isa_ok, label %do_call, label %next_try
// SPRNVL:      store atomic ptr @which_avx10_2nvl
// The co-present SPR candidate (10) still gets the guard.
// SPRNVL:      %isa_ok1 = icmp sge i32 %{{.*}}, 10
// SPRNVL-NEXT: %not_nvl = icmp ne i32 %{{.*}}, 12
// SPRNVL-NEXT: %isa_ok_nvl = and i1 %isa_ok1, %not_nvl
// SPRNVL-NEXT: br i1 %isa_ok_nvl, label %do_call2, label %next_try3
// SPRNVL:      store atomic ptr @which_avx512spr

// DMR candidate (13) is already excluded for NVL by ">="; no guard needed.
// DMR:      %isa_ok = icmp sge i32 %system_isa, 13
// DMR-NEXT: br i1 %isa_ok, label %do_call, label %next_try
// DMR-NOT:  not_nvl

export uniform int which() { return 0; }
