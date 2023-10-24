// RUN: %{ispc} --target=host --nowrap --nostdlib -O2 --emit-llvm-text %s -o - | FileCheck %s --check-prefix=COMMON
// RUN: %{ispc} --target=avx512skx-x32 --nowrap --nostdlib -O2 --emit-llvm-text %s -o - | FileCheck %s --check-prefix=AVX512_X32
// RUN: %{ispc} --target=avx512skx-x16 --nowrap --nostdlib -O2 --emit-llvm-text %s -o - | FileCheck %s --check-prefix=COMMON
// RUN: %{ispc} --target=avx2-i32x4 --nowrap --nostdlib -O2 --emit-llvm-text %s -o - | FileCheck %s --check-prefix=COMMON

// COMMON-LABEL: @foo___unT_3C_4_3E_unT_3C_4_3E_(
// COMMON-NEXT: allocas:
// COMMON-NEXT: [[CMP:%.*]] = icmp ult <4 x i8> %a, %b
// COMMON-NEXT: [[CAST:%.*]] = zext <4 x i1> [[CMP]] to <4 x i8>
// COMMON-NEXT: ret <4 x i8> [[CAST]]

// By some reason, avx512-x32 has 8 wide vector for uint8<4>, x64 has 16
// AVX512_X32-LABEL: @foo___unT_3C_4_3E_unT_3C_4_3E_(
// AVX512_X32-NEXT: allocas:
// AVX512_X32-NEXT: [[CMP:%.*]] = icmp ult <8 x i8> %a, %b
// AVX512_X32-NEXT: [[CAST:%.*]] = zext <8 x i1> [[CMP]] to <8 x i8>
// AVX512_X32-NEXT: ret <8 x i8> [[CAST]]

// REQUIRES: X86_ENABLED

uniform uint8<4> foo(uniform uint8<4> a, uniform uint8<4> b)
{
	return a < b;
}
