// The multi-target runtime dispatcher selects a variant via a linear ">="
// (ICMP_SGE) walk over the Target::ISA enumerant, assuming each ISA is a
// superset of all lower-numbered ones. Nova Lake (NVL=12) breaks that: it has
// AVX10.2 (sorts above SPR=9/GNR=11) but, unlike SPR/GNR/DMR, has no AMX. So an
// AMX-bearing SPR/GNR candidate must carry an extra "system_isa != NVL" guard,
// otherwise an NVL system would select it and fault on AMX instructions.
//
// This test verifies that the guard is emitted for SPR and GNR candidates, and
// is NOT emitted for non-AMX candidates (e.g. avx2). DMR (13) is already
// excluded by the ">=" test and needs no guard.

// RUN: %{ispc} %s -O0 --target=sse2-i32x4,avx512gnr-x16 --emit-llvm-text -o %t.ll --nostdlib
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=GNR
// RUN: %{ispc} %s -O0 --target=sse2-i32x4,avx512spr-x16 --emit-llvm-text -o %t.ll --nostdlib
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=SPR
// RUN: %{ispc} %s -O0 --target=sse2-i32x4,avx2-i32x8 --emit-llvm-text -o %t.ll --nostdlib
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=AVX2

// REQUIRES: X86_ENABLED && !MACOS_HOST

// GNR:      %system_isa = call i32 @__get_system_best_isa()
// GNR-NEXT: %isa_ok = icmp sge i32 %system_isa, 11
// GNR-NEXT: %not_nvl = icmp ne i32 %system_isa, 12
// GNR-NEXT: %isa_ok_nvl = and i1 %isa_ok, %not_nvl
// GNR-NEXT: br i1 %isa_ok_nvl, label %do_call, label %next_try

// SPR:      %system_isa = call i32 @__get_system_best_isa()
// SPR-NEXT: %isa_ok = icmp sge i32 %system_isa, 10
// SPR-NEXT: %not_nvl = icmp ne i32 %system_isa, 12
// SPR-NEXT: %isa_ok_nvl = and i1 %isa_ok, %not_nvl
// SPR-NEXT: br i1 %isa_ok_nvl, label %do_call, label %next_try

// A non-AMX candidate (avx2) must not get the NVL guard.
// AVX2:      %system_isa = call i32 @__get_system_best_isa()
// AVX2-NEXT: %isa_ok = icmp sge i32 %system_isa, {{[0-9]+}}
// AVX2-NEXT: br i1 %isa_ok, label %do_call, label %next_try
// AVX2-NOT:  not_nvl

export uniform int which() { return 0; }
