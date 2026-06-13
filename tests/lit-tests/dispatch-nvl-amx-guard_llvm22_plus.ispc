// Companion to dispatch-nvl-amx-guard.ispc, exercising target sets that
// include the LLVM-22+ avx10.2nvl / avx10.2dmr targets. Verifies that:
//  - when an NVL variant IS present, the NVL candidate (i=12) itself is
//    selected by the plain ">=" test and does NOT carry the AMX requirement
//    guard (the guard is keyed on the candidate's ISA, not on target-set
//    membership), while a co-present SPR candidate still carries it;
//  - the DMR candidate (i=13) DOES carry the AMX requirement guard, since DMR
//    is an AMX-bearing server tier (an NVL system sorts above the SPR/GNR tiers
//    but lacks AMX, so AMX servers must require the AMX capability).

// RUN: %{ispc} %s -O0 --target=avx512spr-x16,avx10.2nvl-x16 --emit-llvm-text -o %t.ll --nostdlib
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=SPRNVL
// RUN: %{ispc} %s -O0 --target=sse2-i32x4,avx10.2dmr-x16 --emit-llvm-text -o %t.ll --nostdlib
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=DMR

// REQUIRES: X86_ENABLED && LLVM_22_0+ && !MACOS_HOST

// NVL candidate (12) is selected by ">=" and must NOT get the AMX guard.
// SPRNVL:      %isa_ok = icmp sge i32 %{{.*}}, 12
// SPRNVL-NEXT: br i1 %isa_ok, label %do_call, label %next_try
// SPRNVL:      store atomic ptr @which_avx10_2nvl
// The co-present SPR candidate (10) still gets the AMX requirement guard.
// SPRNVL:      %isa_ok1 = icmp sge i32 %{{.*}}, 10
// SPRNVL-NEXT: %amx_ok = icmp ne i32 %{{.*}}, 0
// SPRNVL-NEXT: %isa_ok_amx = and i1 %isa_ok1, %amx_ok
// SPRNVL-NEXT: br i1 %isa_ok_amx, label %do_call2, label %next_try3
// SPRNVL:      store atomic ptr @which_avx512spr

// DMR candidate (13) is an AMX server tier and carries the AMX requirement.
// DMR:      %system_isa = call i32 @__get_system_best_isa()
// DMR-NEXT: %system_has_amx = call i32 @__system_has_amx()
// DMR-NEXT: %isa_ok = icmp sge i32 %system_isa, 13
// DMR-NEXT: %amx_ok = icmp ne i32 %system_has_amx, 0
// DMR-NEXT: %isa_ok_amx = and i1 %isa_ok, %amx_ok
// DMR-NEXT: br i1 %isa_ok_amx, label %do_call, label %next_try

export uniform int which() { return 0; }
