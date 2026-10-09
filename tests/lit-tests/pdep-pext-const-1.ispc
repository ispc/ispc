// pdep/pext with compile-time constant masks compile to straight-line
// shift/AND/OR code on all x86 targets, including the varying versions on
// BMI2 targets. The uniform versions on BMI2 targets keep the native
// instruction when the mask is not a contiguous field, also for 32-bit
// operations on 32-bit x86.
// RUN: %{ispc} %s --target=sse2-i32x4 -O2 --nowrap --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=sse4.2-i32x4 -O2 --nowrap --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx1-i32x8 -O2 --nowrap --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=generic-i1x16 -O2 --nowrap --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx2-i32x8 -O2 --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefixes=CHECK,NATIVE
// RUN: %{ispc} %s --target=avx512skx-x16 -O2 --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefixes=CHECK,NATIVE
// RUN: %{ispc} %s --target=avx2-i32x8 --arch=x86 -O2 --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefixes=CHECK,NATIVE

// REQUIRES: X86_ENABLED && !MACOS_HOST

// A contiguous field is a single shift and AND.
// CHECK-LABEL: define {{.*}} @pdep_u32___
// CHECK: [[S:%.*]] = shl i32 %x, 8
// CHECK-NEXT: [[A:%.*]] = and i32 [[S]], 65280
// CHECK-NEXT: ret i32 [[A]]
uniform uint32 pdep_u32(uniform uint32 x) { return pdep32(x, 0x0000FF00u); }

// CHECK-LABEL: define {{.*}} @pext_u32___
// CHECK: [[S:%.*]] = lshr i32 %x, 8
// CHECK-NEXT: [[A:%.*]] = and i32 [[S]], 255
// CHECK-NEXT: ret i32 [[A]]
uniform uint32 pext_u32(uniform uint32 x) { return pext32(x, 0x0000FF00u); }

// CHECK-LABEL: define {{.*}} @pdep_u64___
// CHECK: [[S:%.*]] = shl i64 %x, 40
// CHECK-NEXT: [[A:%.*]] = and i64 [[S]], 72056494526300160
// CHECK-NEXT: ret i64 [[A]]
uniform uint64 pdep_u64(uniform uint64 x) { return pdep64(x, 0x00FFFF0000000000ull); }

// CHECK-LABEL: define {{.*}} @pext_u64___
// CHECK: [[S:%.*]] = lshr i64 %x, 40
// CHECK-NEXT: [[A:%.*]] = and i64 [[S]], 65535
// CHECK-NEXT: ret i64 [[A]]
uniform uint64 pext_u64(uniform uint64 x) { return pext64(x, 0x00FFFF0000000000ull); }

// The varying versions select zero for inactive program instances.
// CHECK-LABEL: define {{.*}} @pdep_v32___
// CHECK: [[S:%.*]] = shl <[[W:[0-9]+]] x i32> %x, splat (i32 8)
// CHECK-NEXT: [[A:%.*]] = and <[[W]] x i32> [[S]], splat (i32 65280)
// CHECK-NEXT: [[R:%.*]] = select <[[W]] x i1> {{.*}}[[A]]
// CHECK-NEXT: ret <[[W]] x i32> [[R]]
uint32 pdep_v32(uint32 x) { return pdep32(x, 0x0000FF00u); }

// Two fields are two shift/AND pairs combined with OR.
// CHECK-LABEL: define {{.*}} @pext_v32___
// CHECK-DAG: [[HS:%.*]] = lshr <[[W:[0-9]+]] x i32> %x, splat (i32 20)
// CHECK-DAG: [[HA:%.*]] = and <[[W]] x i32> [[HS]], splat (i32 240)
// CHECK-DAG: [[LS:%.*]] = lshr <[[W]] x i32> %x, splat (i32 4)
// CHECK-DAG: [[LA:%.*]] = and <[[W]] x i32> [[LS]], splat (i32 15)
// CHECK: [[O:%.*]] = or {{.*}}<[[W]] x i32>
// CHECK-NEXT: [[R:%.*]] = select <[[W]] x i1> {{.*}}[[O]]
// CHECK-NEXT: ret <[[W]] x i32> [[R]]
uint32 pext_v32(uint32 x) { return pext32(x, 0x0F0000F0u); }

// CHECK-LABEL: define {{.*}} @pdep_v64___
// CHECK: [[S:%.*]] = shl <[[W:[0-9]+]] x i64> %x, splat (i64 40)
// CHECK-NEXT: [[A:%.*]] = and <[[W]] x i64> [[S]], splat (i64 72056494526300160)
// CHECK-NEXT: [[R:%.*]] = select <[[W]] x i1> {{.*}}[[A]]
// CHECK-NEXT: ret <[[W]] x i64> [[R]]
uint64 pdep_v64(uint64 x) { return pdep64(x, 0x00FFFF0000000000ull); }

// CHECK-LABEL: define {{.*}} @pext_v64___
// CHECK: [[S:%.*]] = lshr <[[W:[0-9]+]] x i64> %x, splat (i64 40)
// CHECK-NEXT: [[A:%.*]] = and <[[W]] x i64> [[S]], splat (i64 65535)
// CHECK-NEXT: [[R:%.*]] = select <[[W]] x i1> {{.*}}[[A]]
// CHECK-NEXT: ret <[[W]] x i64> [[R]]
uint64 pext_v64(uint64 x) { return pext64(x, 0x00FFFF0000000000ull); }

// A sparse mask keeps the native instruction for uniform operands.
// NATIVE-LABEL: define {{.*}} @pdep_u32_sparse___
// NATIVE: call i32 @llvm.{{x86.bmi.pdep.32|pdep.i32}}(i32 %x, i32 {{.*}}1431655765)
uniform uint32 pdep_u32_sparse(uniform uint32 x) { return pdep32(x, 0x55555555u); }
