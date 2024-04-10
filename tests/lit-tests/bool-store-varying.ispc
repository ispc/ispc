// RUN: %{ispc} --target=host --nowrap --nostdlib -O2 --emit-llvm-text %s -o - | FileCheck %s --check-prefix=COMMON
// RUN: %{ispc} --target=avx512skx-x64 --nowrap --nostdlib -O2 --emit-llvm-text %s -o - | FileCheck %s --check-prefixes=COMMON,AVX512
// RUN: %{ispc} --target=avx512skx-x32 --nowrap --nostdlib -O2 --emit-llvm-text %s -o - | FileCheck %s --check-prefixes=COMMON,AVX512
// RUN: %{ispc} --target=avx512skx-x16 --nowrap --nostdlib -O2 --emit-llvm-text %s -o - | FileCheck %s --check-prefixes=COMMON,AVX512
// RUN: %{ispc} --target=avx2-i32x4 --nowrap --nostdlib -O2 --emit-llvm-text %s -o - | FileCheck %s --check-prefixes=COMMON,AVX2

// REQUIRES: X86_ENABLED

// COMMON-LABEL: @y
// COMMON: local_unnamed_addr global <[[WIDTH:.*]] x i8> zeroinitializer

// COMMON-LABEL: @x
// COMMON: local_unnamed_addr global i8 0
bool y;

uniform bool x;

// COMMON-LABEL: @foo___vyTvyT(
// COMMON-NEXT: allocas:
// COMMON-NEXT: [[CMP:%.*]] = icmp ult <[[WIDTH]] x i8> %a, %b
// COMMON-NEXT: [[CAST:%.*]] = zext <[[WIDTH]] x i1> [[CMP]] to <[[WIDTH]] x i8>
// COMMON-NEXT: store <[[WIDTH]] x i8> [[CAST]], {{.*}} @y
// COMMON-NEXT: ret void

void foo(uint8 a, uint8 b)
{
    y = a < b;
}

// Note:
// bool type has two different representations in ISPC: storage and register.
// Storage representation is governed by C++ compatiblity (i8), whereas
// register representation matches ABI (arguments passing) (i1). Moreover,
// varying vectors of bool matches masks bit count, so they have different
// element type (i1, i8, i16, i32, i64) for different archs (see
// src/llvmutil.cpp BoolVectoType).

// AVX2: define void @boo___vyb(<[[WIDTH]] x [[RTYPE:i[0-9]+]]> %a, <[[WIDTH]] x [[RTYPE]]> %__mask)
// AVX2-NEXT: allocas:
// AVX2-NEXT: [[TRUNC:%.*]] = trunc <[[WIDTH]] x [[RTYPE]]> %a to <[[WIDTH]] x i8>
// AVX2-NEXT: [[CAST:%.*]] = and <[[WIDTH]] x i8> [[TRUNC]], <i8 1, i8 1, i8 1, i8 1>
// AVX2-NEXT: store <[[WIDTH]] x i8> [[CAST]], {{.*}} @y
// AVX2-NEXT: ret void

// AVX512: define void @boo___vyb(<[[WIDTH]] x [[RTYPE:i[0-9]+]]> %a, <[[WIDTH]] x [[RTYPE]]> %__mask)
// AVX512-NEXT: allocas:
// AVX512-NEXT: [[CAST:%.*]] = zext <[[WIDTH]] x [[RTYPE]]> %a to <[[WIDTH]] x i8>
// AVX512-NEXT: store <[[WIDTH]] x i8> [[CAST]], {{.*}} @y
// AVX512-NEXT: ret void

void boo(bool a) {
    y = a;
}

// COMMON: define void @coo___unb(i1 %a
// COMMON-NEXT: allocas:
// COMMON-NEXT: [[CAST:%.*]] = zext i1 %a to i8
// COMMON-NEXT: store i8 [[CAST]], {{.*}} @x
// COMMON-NEXT: ret void

void coo(uniform bool a) {
    x = a;
}
