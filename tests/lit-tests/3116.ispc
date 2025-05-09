// RUN: not %{ispc} --target=avx2-i32x8 --nowrap --nostdlib t.ispc --emit-llvm-text -o - --enable-llvm-intrinsics

// REQUIRES: X86_ENABLED

// CHECK: Error: Unable to find any matching overload for call to function "llvm.x86.ssse3.pmul.hr.sw".
// CHECK-NEXT: Passed types: (varying int16, varying int16)
// CHECK-NEXT: Candidate: uniform int64<1>(uniform int64<1> , uniform int64<1> )

int16 test(int16 t) {
    return @llvm.x86.ssse3.pmul.hr.sw(t, t);
}
