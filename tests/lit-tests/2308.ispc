// Pins today's code for discussion #2308: stores to members of an soa<>
// struct with short-vector fields, at foreach indices, are reported as
// scatters although each member is contiguous in the soa<> layout.
// Today: 3 "Scatter required" warnings on every x86 target, and
// avx512skx-x16 emits 6 hardware scatters.
// Ideal: no warnings and plain vector stores.

// RUN: %{ispc} %s --target=avx512skx-x16 --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefix=SKX
// RUN: %{ispc} %s --target=sse4-i32x4 --nostdlib --nowrap -o /dev/null 2> %t.sse4
// RUN: FileCheck %s --input-file=%t.sse4 --check-prefixes=WARN
// RUN: FileCheck %s --input-file=%t.sse4 --check-prefix=COUNT3
// RUN: %{ispc} %s --target=avx2-i32x8 --nostdlib --nowrap -o /dev/null 2> %t.avx2
// RUN: FileCheck %s --input-file=%t.avx2 --check-prefixes=WARN
// RUN: FileCheck %s --input-file=%t.avx2 --check-prefix=COUNT3
// RUN: %{ispc} %s --target=avx512skx-x16 --nostdlib --nowrap -o /dev/null 2> %t.avx512skx
// RUN: FileCheck %s --input-file=%t.avx512skx --check-prefixes=WARN
// RUN: FileCheck %s --input-file=%t.avx512skx --check-prefix=COUNT3

// REQUIRES: X86_ENABLED

// The warning checks match in any order; the COUNT prefix checks that
// there are no other warnings.
// COUNT3-COUNT-3: Performance Warning
// COUNT3-NOT: Performance Warning

typedef float<3> Vec3f;
typedef int<2> Vec2i;

struct RayState {
    Vec2i pixel;
    Vec3f color;
    int currentSample;
};

// SKX-LABEL: @"soaTest___
// SKX: call void @llvm.x86.avx512.mask.scatter.qpi.512(
void soaTest(soa<TARGET_WIDTH> RayState rayStates[]) {
    foreach (i = 0 ... 128) {
        // WARN-DAG: 2308.ispc:[[#@LINE+1]]:9: Performance Warning: Scatter required to store value.
        rayStates[i].pixel.x = i;
        // WARN-DAG: 2308.ispc:[[#@LINE+1]]:9: Performance Warning: Scatter required to store value.
        rayStates[i].pixel.y = i * 2;
        // WARN-DAG: 2308.ispc:[[#@LINE+1]]:9: Performance Warning: Scatter required to store value.
        rayStates[i].currentSample = i * 3;
    }
}
