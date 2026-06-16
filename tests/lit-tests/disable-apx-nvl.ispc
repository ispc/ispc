// On the avx10.2nvl target (LLVM 23.0+), --opt=disable-apx with no list turns
// off the whole APX sub-feature set, including jmpabs, even with -g enabled.

//; RUN: %{ispc} %s --target=avx10.2nvl-x16 -g --opt=disable-apx --print-target -o /dev/null | FileCheck %s --check-prefix=FEAT
//; RUN: %{ispc} %s --target=avx10.2nvl-x16 -g --opt=disable-apx --nowrap --emit-asm -o - | FileCheck %s --check-prefix=ASM

// REQUIRES: X86_ENABLED && !MACOS_HOST && LLVM_23_0+

// FEAT: Target Feature String: -egpr,-ndd,-push2pop2,-ppx,-ccmp,-cf,-nf,-zu,-jmpabs

// All APX sub-features are disabled: no EGPR (R16-R31) and no {nf} forms.
// ASM: crunch
// ASM-NOT: %r{{(1[6-9]|2[0-9]|3[01])}}
// ASM-NOT: {nf}

uniform int64 crunch(uniform int64 a, uniform int64 b, uniform int64 c, uniform int64 d, uniform int64 e,
                     uniform int64 f, uniform int64 g, uniform int64 h) {
    uniform int64 r = a * b + c * d;
    r = r * e + f * g;
    return r * h + a * b;
}
