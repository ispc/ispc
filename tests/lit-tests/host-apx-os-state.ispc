// When the target is detected from the host (--target=host, with or without
// --cpu, or neither --target nor --cpu), the detected ISA is NVL or DMR and
// the OS has not enabled APX state, all APX sub-features are disabled,
// whatever --opt=disable-apx says. Otherwise the backend defaults are kept,
// minus the sub-features in --opt=disable-apx.
// --mock-host-target and --mock-host-os-apx supply the host ISA and OS APX
// state, so no NVL/DMR hardware is needed.

// Host selection with OS APX off: APX is off, even with a partial user mask.
// RUN: %{ispc} %s --target=host --mock-host-target=avx10.2dmr-x16 --mock-host-os-apx=off --print-target --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefixes=DMR,OFF
// RUN: %{ispc} %s --mock-host-target=avx10.2nvl-x8 --mock-host-os-apx=off --print-target --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefixes=IMPLICIT-NVL,NVL,OFF
// RUN: %{ispc} %s --target=host --mock-host-target=avx10.2nvl-x8 --mock-host-os-apx=off --opt=disable-apx=egpr --print-target --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefixes=NVL,OFF

// Host selection with OS APX on: the backend defaults are kept, minus the user mask.
// RUN: %{ispc} %s --mock-host-target=avx10.2dmr-x16 --mock-host-os-apx=on --print-target --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefixes=IMPLICIT-DMR,DMR,DEFAULT
// RUN: %{ispc} %s --target=host --mock-host-target=avx10.2nvl-x8 --mock-host-os-apx=on --print-target --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefixes=NVL,DEFAULT
// RUN: %{ispc} %s --target=host --mock-host-target=avx10.2dmr-x16 --mock-host-os-apx=on --opt=disable-apx=egpr --print-target --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefixes=DMR,NO-EGPR

// --target=host with --cpu: the host still selects the ISA, so the OS APX
// state applies, and --cpu must be compatible with the host ISA.
// RUN: %{ispc} %s --target=host --cpu=dmr --mock-host-target=avx10.2dmr-x16 --mock-host-os-apx=off --print-target --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefixes=DMR,OFF
// RUN: not %{ispc} %s --target=host --cpu=dmr --mock-host-target=avx10.2nvl-x8 --mock-host-os-apx=off --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefix=HOST-CPU-MISMATCH

// An explicitly named NVL/DMR target (or --cpu) with OS APX off: the backend
// defaults are kept, minus the user mask. The target or --cpu selects the ISA,
// not the host, so the mocked host ISA differs from the selected one.
// RUN: %{ispc} %s --target=avx10.2dmr-x16 --mock-host-target=avx10.2nvl-x8 --mock-host-os-apx=off --print-target --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefixes=DMR,DEFAULT
// RUN: %{ispc} %s --cpu=dmr --mock-host-target=avx10.2nvl-x8 --mock-host-os-apx=off --print-target --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefixes=CPU-DMR,DMR,DEFAULT
// RUN: %{ispc} %s --cpu=nvl --mock-host-target=avx10.2dmr-x16 --mock-host-os-apx=off --print-target --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefixes=CPU-NVL,NVL,DEFAULT
// RUN: %{ispc} %s --target=avx10.2nvl-x8 --mock-host-target=avx10.2dmr-x16 --mock-host-os-apx=off --opt=disable-apx=egpr --print-target --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefixes=NVL,NO-EGPR

// An unknown or non-x86 --mock-host-target is rejected.
// RUN: not %{ispc} %s --mock-host-target=bogus --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefix=BAD-TARGET
// RUN: not %{ispc} %s --mock-host-target=neon-i32x4 --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefix=NON-X86-TARGET

// REQUIRES: X86_ENABLED && X86_64_HOST && !MACOS_HOST && LLVM_23_0+

// IMPLICIT-DMR: Warning: No --target specified on command-line. Using default system target "avx10.2dmr-x16".
// IMPLICIT-NVL: Warning: No --target specified on command-line. Using default system target "avx10.2nvl-x8".
// CPU-DMR: Warning: No --target specified on command-line. Using ISA "avx10.2dmr-x16" based on specified device "dmr".
// CPU-NVL: Warning: No --target specified on command-line. Using ISA "avx10.2nvl-x8" based on specified device "nvl".

// DMR: Target CPU: diamondrapids
// NVL: Target CPU: novalake

// OFF-NEXT: Target Feature String: -egpr,-ndd,-push2pop2,-ppx,-ccmp,-cf,-nf,-zu,-jmpabs{{$}}
// OFF-NEXT: Target APX Features: none

// LLVM 24 (llvm/llvm-project#242389) added CF to the DMR and NVL defaults, so
// cf is optional in the backend default lists.
// DEFAULT-NEXT: Target Feature String: {{$}}
// DEFAULT-NEXT: Target APX Features: egpr,ndd,push2pop2,ppx,ccmp,{{(cf,)?}}nf,zu,jmpabs{{$}}

// NO-EGPR-NEXT: Target Feature String: -egpr{{$}}
// NO-EGPR-NEXT: Target APX Features: ndd,push2pop2,ppx,ccmp,{{(cf,)?}}nf,zu,jmpabs{{$}}

// BAD-TARGET: Error: Unknown or non-x86 target "bogus" in --mock-host-target.
// NON-X86-TARGET: Error: Unknown or non-x86 target "neon-i32x4" in --mock-host-target.
// HOST-CPU-MISMATCH: Error: The requested device (dmr) is incompatible with the device required for avx10.2nvl-x8 target (novalake)

void foo(uniform int a[], uniform int n) {
    for (uniform int i = 0; i < n; ++i)
        a[i] = a[i] + 1;
}
