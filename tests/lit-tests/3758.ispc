// By default APX is on: EGPR (R16-R31) is allocated and {nf} forms are emitted.
//; RUN: %{ispc} %s --target=avx10.2dmr-x16 --nowrap --emit-asm -o - | FileCheck %s --check-prefix=APX-ON

// Disabling only egpr removes EGPR usage (the #3758 workaround) but keeps the
// other APX sub-features such as {nf}.
//; RUN: %{ispc} %s --target=avx10.2dmr-x16 --opt=disable-apx=egpr --nowrap --emit-asm -o - | FileCheck %s --check-prefix=NO-EGPR

// Disabling all APX sub-features removes both EGPR and {nf} forms.
//; RUN: %{ispc} %s --target=avx10.2dmr-x16 --opt=disable-apx --nowrap --emit-asm -o - | FileCheck %s --check-prefix=NO-APX

// Reproduce the original multi-target + debug-info (-g) compilation from the
// issue.
//; RUN: %{ispc} %s --target=sse4-i32x8,avx2-i32x16,avx512skx-x16,avx10.2dmr-x16 -g --opt=disable-apx=egpr --nowrap --emit-asm -o %t.s
//; RUN: FileCheck %s --check-prefix=MULTI-G --input-file %t_avx10_2dmr.s

// Exact LLVM target feature strings for each option variant.
//; RUN: %{ispc} %s --target=avx10.2dmr-x16 --opt=disable-apx --print-target -o /dev/null | FileCheck %s --check-prefix=FEAT-ALL
//; RUN: %{ispc} %s --target=avx10.2dmr-x16 --opt=disable-apx=egpr --print-target -o /dev/null | FileCheck %s --check-prefix=FEAT-EGPR
//; RUN: %{ispc} %s --target=avx10.2dmr-x16 --opt=disable-apx=ndd,nf --print-target -o /dev/null | FileCheck %s --check-prefix=FEAT-NDD-NF

// REQUIRES: X86_ENABLED && !MACOS_HOST && LLVM_20_0+

// APX-ON: {nf}
// APX-ON: %r1{{[6-9]}}

// Only egpr is disabled, so {nf} forms must still be present while no EGPR
// (R16-R31) is allocated.
// NO-EGPR: {nf}
// NO-EGPR-NOT: %r{{(1[6-9]|2[0-9]|3[01])}}

// All APX sub-features are disabled: the function still compiles, but no EGPR
// and no {nf} forms appear.
// NO-APX: crunch
// NO-APX-NOT: %r{{(1[6-9]|2[0-9]|3[01])}}
// NO-APX-NOT: {nf}

// In the multi-target build the avx10.2dmr variant is compiled and contains no
// EGPR despite -g being enabled.
// MULTI-G: crunch
// MULTI-G-NOT: %r{{(1[6-9]|2[0-9]|3[01])}}

// jmpabs is only known to the LLVM X86 backend from 23.0 on, so it is appended
// to the feature string only there.
// FEAT-ALL: Target Feature String: -egpr,-ndd,-push2pop2,-ppx,-ccmp,-cf,-nf,-zu{{(,-jmpabs)?}}
// FEAT-EGPR: Target Feature String: -egpr
// FEAT-NDD-NF: Target Feature String: -ndd,-nf

uniform int64 crunch(uniform int64 a, uniform int64 b, uniform int64 c, uniform int64 d, uniform int64 e,
                     uniform int64 f, uniform int64 g, uniform int64 h, uniform int64 i, uniform int64 j,
                     uniform int64 k, uniform int64 l) {
    uniform int64 r = a * b + c * d;
    r = r * e + f * g;
    r = r * h + i * j;
    r = r * k + l * a;
    r = r ^ (b * c) ^ (d * e);
    r = r + (f ^ g) * (h ^ i);
    return r * j + k * l;
}
