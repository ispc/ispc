// When the target is detected from the host (--target=host or no --target),
// the detected ISA is NVL or DMR and the OS has not enabled APX state, all APX
// sub-features are disabled, whatever --opt=disable-apx says. Otherwise the
// backend defaults are kept, minus the sub-features in --opt=disable-apx.
// --mock-host-target and --mock-host-os-apx supply the host ISA and OS APX
// state, so no NVL/DMR hardware is needed.

// Host selection with OS APX off: the detected ISA is kept and APX is off.
//; RUN: %{ispc} %s --target=host --mock-host-target=avx10.2dmr-x16 --mock-host-os-apx=off --print-target --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefixes=DMR,OFF
//; RUN: %{ispc} %s --mock-host-target=avx10.2dmr-x16 --mock-host-os-apx=off --print-target --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefixes=IMPLICIT-DMR,DMR,OFF
//; RUN: %{ispc} %s --target=host --mock-host-target=avx10.2nvl-x8 --mock-host-os-apx=off --print-target --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefixes=NVL,OFF
//; RUN: %{ispc} %s --mock-host-target=avx10.2nvl-x8 --mock-host-os-apx=off --print-target --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefixes=IMPLICIT-NVL,NVL,OFF

// Host selection with OS APX off and a partial user mask: APX is still off.
//; RUN: %{ispc} %s --target=host --mock-host-target=avx10.2dmr-x16 --mock-host-os-apx=off --opt=disable-apx=egpr --print-target --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefixes=DMR,OFF
//; RUN: %{ispc} %s --target=host --mock-host-target=avx10.2nvl-x8 --mock-host-os-apx=off --opt=disable-apx=egpr --print-target --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefixes=NVL,OFF

// Host selection with OS APX on: the backend defaults are kept, minus the user mask.
//; RUN: %{ispc} %s --target=host --mock-host-target=avx10.2dmr-x16 --mock-host-os-apx=on --print-target --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefixes=DMR,DEFAULT
//; RUN: %{ispc} %s --mock-host-target=avx10.2dmr-x16 --mock-host-os-apx=on --print-target --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefixes=IMPLICIT-DMR,DMR,DEFAULT
//; RUN: %{ispc} %s --target=host --mock-host-target=avx10.2dmr-x16 --mock-host-os-apx=on --opt=disable-apx=egpr --print-target --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefixes=DMR,NO-EGPR
//; RUN: %{ispc} %s --target=host --mock-host-target=avx10.2nvl-x8 --mock-host-os-apx=on --print-target --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefixes=NVL,DEFAULT
//; RUN: %{ispc} %s --mock-host-target=avx10.2nvl-x8 --mock-host-os-apx=on --print-target --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefixes=IMPLICIT-NVL,NVL,DEFAULT
//; RUN: %{ispc} %s --target=host --mock-host-target=avx10.2nvl-x8 --mock-host-os-apx=on --opt=disable-apx=egpr --print-target --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefixes=NVL,NO-EGPR

// An explicitly named NVL/DMR target (or --cpu) with OS APX off: the backend
// defaults are kept, minus the user mask.
//; RUN: %{ispc} %s --target=avx10.2dmr-x16 --mock-host-target=avx10.2dmr-x16 --mock-host-os-apx=off --print-target --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefixes=DMR,DEFAULT
//; RUN: %{ispc} %s --target=avx10.2dmr-x16 --mock-host-target=avx10.2dmr-x16 --mock-host-os-apx=off --opt=disable-apx=egpr --print-target --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefixes=DMR,NO-EGPR
//; RUN: %{ispc} %s --cpu=dmr --mock-host-target=avx10.2dmr-x16 --mock-host-os-apx=off --print-target --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefixes=DMR,DEFAULT
//; RUN: %{ispc} %s --target=avx10.2nvl-x8 --mock-host-target=avx10.2nvl-x8 --mock-host-os-apx=off --print-target --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefixes=NVL,DEFAULT
//; RUN: %{ispc} %s --target=avx10.2nvl-x8 --mock-host-target=avx10.2nvl-x8 --mock-host-os-apx=off --opt=disable-apx=egpr --print-target --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefixes=NVL,NO-EGPR
//; RUN: %{ispc} %s --cpu=nvl --mock-host-target=avx10.2nvl-x8 --mock-host-os-apx=off --print-target --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefixes=NVL,DEFAULT

// An unknown --mock-host-target is rejected.
//; RUN: not %{ispc} %s --mock-host-target=bogus --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefix=BAD-TARGET

// REQUIRES: X86_ENABLED && X86_64_HOST && !MACOS_HOST && LLVM_22_0+

// IMPLICIT-DMR: Warning: No --target specified on command-line. Using default system target "avx10.2dmr-x16".
// IMPLICIT-NVL: Warning: No --target specified on command-line. Using default system target "avx10.2nvl-x8".

// DMR: Target CPU: diamondrapids
// NVL: Target CPU: novalake

// OFF-NEXT: Target Feature String: -egpr,-ndd,-push2pop2,-ppx,-ccmp,-cf,-nf,-zu{{(,-jmpabs)?}}{{$}}
// OFF-NEXT: Target APX Features: none

// DEFAULT-NEXT: Target Feature String: {{$}}
// DEFAULT-NEXT: Target APX Features: egpr,ndd,push2pop2,ppx,ccmp,nf,zu{{(,jmpabs)?}}{{$}}

// NO-EGPR-NEXT: Target Feature String: -egpr{{$}}
// NO-EGPR-NEXT: Target APX Features: ndd,push2pop2,ppx,ccmp,nf,zu{{(,jmpabs)?}}{{$}}

// BAD-TARGET: Error: Unknown or unsupported target "bogus" in --mock-host-target.

void foo(uniform int a[], uniform int n) {
    for (uniform int i = 0; i < n; ++i)
        a[i] = a[i] + 1;
}
