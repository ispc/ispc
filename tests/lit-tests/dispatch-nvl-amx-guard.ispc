// The multi-target runtime dispatcher selects a variant via a linear ">="
// (ICMP_SGE) walk over the Target::ISA enumerant, assuming each ISA is a
// superset of all lower-numbered ones. AMX breaks that: Nova Lake (NVL=12) has
// AVX10.2 (sorts above SPR=10/GNR=11) but, unlike SPR/GNR/DMR, has no AMX. So an
// AMX-bearing server candidate (SPR/GNR/DMR) must additionally require that the
// system has AMX -- queried via __system_has_amx() -- otherwise an NVL system
// would select it and fault on AMX instructions.
//
// This test verifies that the AMX requirement guard is emitted for SPR and GNR
// candidates, and is NOT emitted for non-AMX candidates (e.g. avx2).

// RUN: %{ispc} %s -O0 --target=sse2-i32x4,avx512gnr-x16 --emit-llvm-text -o %t.ll --nostdlib
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=GNR
// RUN: %{ispc} %s -O0 --target=sse2-i32x4,avx512spr-x16 --emit-llvm-text -o %t.ll --nostdlib
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=SPR
// RUN: %{ispc} %s -O0 --target=sse2-i32x4,avx2-i32x8 --emit-llvm-text -o %t.ll --nostdlib
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=AVX2

// REQUIRES: X86_ENABLED && !MACOS_HOST

// GNR:      %system_isa = call i32 @__get_system_best_isa()
// GNR-NEXT: %system_has_amx = call i32 @__system_has_amx()
// GNR-NEXT: %isa_ok = icmp sge i32 %system_isa, 11
// GNR-NEXT: %amx_ok = icmp ne i32 %system_has_amx, 0
// GNR-NEXT: %isa_ok_amx = and i1 %isa_ok, %amx_ok
// GNR-NEXT: br i1 %isa_ok_amx, label %do_call, label %next_try

// SPR:      %system_isa = call i32 @__get_system_best_isa()
// SPR-NEXT: %system_has_amx = call i32 @__system_has_amx()
// SPR-NEXT: %isa_ok = icmp sge i32 %system_isa, 10
// SPR-NEXT: %amx_ok = icmp ne i32 %system_has_amx, 0
// SPR-NEXT: %isa_ok_amx = and i1 %isa_ok, %amx_ok
// SPR-NEXT: br i1 %isa_ok_amx, label %do_call, label %next_try

// A non-AMX candidate (avx2) must not query AMX nor get the AMX guard.
// AVX2:      %system_isa = call i32 @__get_system_best_isa()
// AVX2-NEXT: %isa_ok = icmp sge i32 %system_isa, {{[0-9]+}}
// AVX2-NEXT: br i1 %isa_ok, label %do_call, label %next_try
// AVX2-NOT:  __system_has_amx
// AVX2-NOT:  amx_ok

export uniform int which() { return 0; }
