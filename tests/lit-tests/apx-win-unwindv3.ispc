// On Windows x64, APX-capable targets enable the EGPR registers (R16-R31) by
// default, which cannot be described by the legacy v1/v2 SEH unwind encodings.
// ISPC mirrors clang by selecting unwind v3 (the "winx64-eh-unwind" module
// flag with value 3) so the X86 backend does not error out.

// EGPR enabled (default): the v3 unwind module flag is set.
//; RUN: %{ispc} %s --target=avx10.2dmr-x16 --arch=x86-64 --target-os=windows --emit-llvm-text -o - | FileCheck %s --check-prefix=V3

// EGPR disabled: no v3 flag is needed.
//; RUN: %{ispc} %s --target=avx10.2dmr-x16 --arch=x86-64 --target-os=windows --opt=disable-apx=egpr --emit-llvm-text -o - | FileCheck %s --check-prefix=NO-V3

// Non-APX target: no v3 flag.
//; RUN: %{ispc} %s --target=avx512skx-x16 --arch=x86-64 --target-os=windows --emit-llvm-text -o - | FileCheck %s --check-prefix=NO-V3

// The Linux counterpart (unwind v3 is Windows-only) is verified separately in
// apx-win-unwindv3-linux.ispc, which is gated on LINUX_ENABLED.

// REQUIRES: X86_ENABLED && WINDOWS_ENABLED && LLVM_23_0+

// V3: !{i32 2, !"winx64-eh-unwind", i32 3}

// NO-V3-NOT: winx64-eh-unwind

void foo(uniform int a[], uniform int n) {
    for (uniform int i = 0; i < n; ++i)
        a[i] = a[i] + 1;
}
