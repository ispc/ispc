// NVL requires APX but not AMX; SPR requires AMX; DMR requires both.
// Partial disable-apx preserves APX guards; full disable-apx removes only those guards.

// RUN: %{ispc} %s -O0 --target=avx512spr-x16,avx10.2nvl-x16 --emit-llvm-text -o %t.ll --nostdlib
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=SPRNVL
// RUN: %{ispc} %s -O0 --target=sse2-i32x4,avx10.2nvl-x16 --emit-llvm-text -o %t.ll --nostdlib
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=NVL-APX --implicit-check-not=__system_has_amx
// RUN: %{ispc} %s -O0 --target=sse2-i32x4,avx10.2nvl-x16 --opt=disable-apx=egpr --emit-llvm-text -o %t.ll --nostdlib
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=NVL-APX --implicit-check-not=__system_has_amx
// RUN: %{ispc} %s -O0 --target=sse2-i32x4,avx10.2nvl-x16 --opt=disable-apx --emit-llvm-text -o %t.ll --nostdlib
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=NVL-NOAPX --implicit-check-not=__system_has_apx --implicit-check-not=__system_has_amx
// RUN: %{ispc} %s -O0 --target=sse2-i32x4,avx10.2dmr-x16 --emit-llvm-text -o %t.ll --nostdlib
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=DMR
// RUN: %{ispc} %s -O0 --target=sse2-i32x4,avx10.2dmr-x16 --opt=disable-apx=egpr --emit-llvm-text -o %t.ll --nostdlib
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=DMR
// RUN: %{ispc} %s -O0 --target=sse2-i32x4,avx10.2dmr-x16 --opt=disable-apx --emit-llvm-text -o %t.ll --nostdlib
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=DMR-NOAPX --implicit-check-not=__system_has_apx

// REQUIRES: X86_ENABLED && LLVM_22_0+ && !MACOS_HOST

// The normal NVL candidate (12) is selected by ">=" but must require APX.
// SPRNVL:      %isa_ok = icmp sge i32 %{{.*}}, 12
// SPRNVL-NEXT: %apx_ok = icmp ne i32 %{{.*}}, 0
// SPRNVL-NEXT: %isa_ok_apx = and i1 %isa_ok, %apx_ok
// SPRNVL-NEXT: br i1 %isa_ok_apx, label %do_call, label %next_try
// SPRNVL:      store atomic ptr @which_avx10_2nvl
// The co-present SPR candidate (10) still gets the AMX requirement guard.
// SPRNVL:      %isa_ok1 = icmp sge i32 %{{.*}}, 10
// SPRNVL-NEXT: %amx_ok = icmp ne i32 %{{.*}}, 0
// SPRNVL-NEXT: %isa_ok_amx = and i1 %isa_ok1, %amx_ok
// SPRNVL-NEXT: br i1 %isa_ok_amx, label %do_call2, label %next_try3
// SPRNVL:      store atomic ptr @which_avx512spr

// Default and partially APX-disabled NVL variants require APX but not AMX.
// NVL-APX:      %system_has_apx = call i32 @__system_has_apx()
// NVL-APX:      %isa_ok = icmp sge i32 %system_isa, 12
// NVL-APX-NEXT: %apx_ok = icmp ne i32 %system_has_apx, 0
// NVL-APX-NEXT: %isa_ok_apx = and i1 %isa_ok, %apx_ok
// NVL-APX-NEXT: br i1 %isa_ok_apx, label %do_call, label %next_try

// Fully disabling APX leaves NVL selectable without querying APX.
// NVL-NOAPX:      %system_isa = call i32 @__get_system_best_isa()
// NVL-NOAPX:      %isa_ok = icmp sge i32 %system_isa, 12
// NVL-NOAPX-NEXT: br i1 %isa_ok, label %do_call, label %next_try

// Default and partially APX-disabled DMR variants require both AMX and APX.
// DMR:      %system_isa = call i32 @__get_system_best_isa()
// DMR-NEXT: %system_has_amx = call i32 @__system_has_amx()
// DMR-NEXT: %system_has_apx = call i32 @__system_has_apx()
// DMR-NEXT: %isa_ok = icmp sge i32 %system_isa, 13
// DMR-NEXT: %amx_ok = icmp ne i32 %system_has_amx, 0
// DMR-NEXT: %isa_ok_amx = and i1 %isa_ok, %amx_ok
// DMR-NEXT: %apx_ok = icmp ne i32 %system_has_apx, 0
// DMR-NEXT: %isa_ok_apx = and i1 %isa_ok_amx, %apx_ok
// DMR-NEXT: br i1 %isa_ok_apx, label %do_call, label %next_try

// Fully disabling APX leaves DMR selectable with its AMX requirement only.
// DMR-NOAPX:      %system_isa = call i32 @__get_system_best_isa()
// DMR-NOAPX-NEXT: %system_has_amx = call i32 @__system_has_amx()
// DMR-NOAPX:      %isa_ok = icmp sge i32 %system_isa, 13
// DMR-NOAPX-NEXT: %amx_ok = icmp ne i32 %system_has_amx, 0
// DMR-NOAPX-NEXT: %isa_ok_amx = and i1 %isa_ok, %amx_ok
// DMR-NOAPX-NEXT: br i1 %isa_ok_amx, label %do_call, label %next_try

export uniform int which() { return 0; }
