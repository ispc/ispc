// Pins the gathers and scatters that ImproveMemoryOps replaces with ordinary
// loads and stores: a varying pointer into a global array, all lanes reading
// one element, and reversed consecutive elements. The checks describe today's
// output; each case also gives the ideal output in an "Ideal:" comment, and
// the change that reaches it updates the checks of that case.
//
// sse4 has no gather instruction: a remaining gather is expanded inline into
// scalar loads there, so the sse4 runs check for those instead of a call.

// RUN: %{ispc} %s --target=sse4-i32x4 --nostdlib --nowrap --emit-llvm-text -o - \
// RUN:   | FileCheck %s --check-prefixes=CHECK,SSE4
// RUN: %{ispc} %s --target=sse4-i32x4 --arch=x86 --nostdlib --nowrap --emit-llvm-text -o - \
// RUN:   | FileCheck %s --check-prefixes=CHECK,SSE4
// RUN: %{ispc} %s --target=avx2-i32x8 --nostdlib --nowrap --emit-llvm-text -o - \
// RUN:   | FileCheck %s --check-prefixes=CHECK,AVX2
// RUN: %{ispc} %s --target=avx512skx-x16 --nostdlib --nowrap --emit-llvm-text -o - \
// RUN:   | FileCheck %s --check-prefixes=CHECK,SKX

// REQUIRES: X86_ENABLED

// Indexing a global array gives a constant vector of per-lane addresses
// (@garr + 12 + 4 * lane). ImproveMemoryOps finds the common base in it, and
// the consecutive offsets become one vector load. The second run (--arch=x86)
// covers the 32-bit offsets of a 32-bit target.
// Ideal: = today.
// CHECK-LABEL: @global_load___
// CHECK-NOT: call {{.*}}gather
// SSE4-NOT: load float
// CHECK: load <{{4|8|16}} x float>
// CHECK-NOT: call {{.*}}gather
// SSE4-NOT: load float
uniform float garr[100];
void global_load(uniform float r[]) { r[programIndex] = garr[programIndex + 3]; }

// Every lane reads the same element through a varying index: one scalar load
// and a broadcast.
// Ideal: = today.
// CHECK-LABEL: @same_element___
// CHECK-NOT: call {{.*}}gather
// CHECK: load float
// CHECK-NOT: call {{.*}}gather
// SSE4-NOT: load float
// CHECK: shufflevector <[[W:[0-9]+]] x float> {{%.*}}, <[[W]] x float> poison, <[[W]] x i32> zeroinitializer
// CHECK-NOT: call {{.*}}gather
// SSE4-NOT: load float
void same_element(uniform float p[], uniform float b, uniform float r[]) {
    int x = b - 5;
    r[programIndex] = p[x];
}

// Consecutive elements in reverse order. ImproveMemoryOps turns the gather
// into a vector load and a reversing shuffle only where the mask is known to
// be all on. On sse4 and avx2 the function checks its mask on entry (the
// front end adds this check when the body is expensive) and has an all-on
// path, where the gather is replaced; the partial path keeps the gather. On
// avx512 masking is cheap, there is no all-on path, and the gather stays.
// Ideal: the load and shuffle for any mask (a masked load with a reversed
// mask), so also on avx512.
// SSE4-LABEL: @reverse_load___
// SSE4: load <4 x float>
// SSE4-NEXT: shufflevector <4 x float> {{%.*}}, <4 x float> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
// AVX2-LABEL: @reverse_load___
// AVX2: load <8 x float>
// AVX2-NEXT: shufflevector <8 x float> {{%.*}}, <8 x float> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
// SKX-LABEL: @reverse_load___
// SKX: call <16 x float> @llvm.x86.avx512.mask.gather.dps.512(
void reverse_load(uniform float p[], uniform float r[]) { r[programIndex] = p[programCount - 1 - programIndex]; }

// The same for a scatter: reverse the values, then one vector store.
// Ideal: as reverse_load.
// SSE4-LABEL: @reverse_store___
// SSE4: shufflevector <4 x float> {{%.*}}, <4 x float> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
// SSE4-NEXT: store <4 x float>
// AVX2-LABEL: @reverse_store___
// AVX2: shufflevector <8 x float> {{%.*}}, <8 x float> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
// AVX2-NEXT: store <8 x float>
// SKX-LABEL: @reverse_store___
// SKX: call void @llvm.x86.avx512.mask.scatter.dps.512(
void reverse_store(uniform float p[], uniform float v[]) { p[programCount - 1 - programIndex] = v[programIndex]; }
