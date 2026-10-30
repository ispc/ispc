// Pins today's performance warnings for discussion #2919: bilinear remap in
// HW and HWC layouts. Each output element reads 4 data-dependent image
// elements (genuine gathers) and 2 adjacent grid elements (a stride-2 read).
// Today: on AVX targets the grid read is 2 gathers per kernel (12 gather
// warnings in total); sse4-i32x4 coalesces it into one load (8 gather
// warnings and 2 coalescing warnings). The HWC store with a runtime channel
// count is a scatter on every target.
// Ideal: the 8 image gather warnings and the scatter warning stay; the grid
// read becomes vector loads plus a deinterleave on every target.

// RUN: %{ispc} %s --target=sse4-i32x4 --nowrap -o /dev/null 2> %t.sse4
// RUN: FileCheck %s --input-file=%t.sse4 --check-prefixes=CHECK,SSE4
// RUN: FileCheck %s --input-file=%t.sse4 --check-prefix=COUNT11
// RUN: %{ispc} %s --target=avx2-i32x8 --nowrap -o /dev/null 2> %t.avx2
// RUN: FileCheck %s --input-file=%t.avx2 --check-prefixes=CHECK,AVX
// RUN: FileCheck %s --input-file=%t.avx2 --check-prefix=COUNT13
// RUN: %{ispc} %s --target=avx512skx-x16 --nowrap -o /dev/null 2> %t.avx512skx
// RUN: FileCheck %s --input-file=%t.avx512skx --check-prefixes=CHECK,AVX
// RUN: FileCheck %s --input-file=%t.avx512skx --check-prefix=COUNT13

// REQUIRES: X86_ENABLED

// The warning checks match in any order; the COUNT prefix checks that
// there are no other warnings.
// COUNT11-COUNT-11: Performance Warning
// COUNT11-NOT: Performance Warning
// COUNT13-COUNT-13: Performance Warning
// COUNT13-NOT: Performance Warning

#define IDX2D(H, W, y, x) ((y) * (W) + (x))
#define IDX3D(H, W, C, y, x, c) (((y) * (W) + (x)) * (C) + (c))

export void remap(uniform float img_out[], uniform float img_in[], uniform float grid_in[],
                  uniform int32 IW, uniform int32 IH, uniform int32 OW, uniform int32 OH) {
    foreach (y = 0 ... OH, x = 0 ... OW) {
        // SSE4-DAG: 2919.ispc:[[#@LINE+2]]:22: Performance Warning: Coalesced 2 gathers starting here (other at line [[#@LINE+4]]) into 1 load (1 x 8-wide).
        // AVX-DAG: 2919.ispc:[[#@LINE+1]]:22: Performance Warning: Gather required to load value.
        float px_f = grid_in[IDX3D(OH, OW, 2, y, x, 0)];
        // AVX-DAG: 2919.ispc:[[#@LINE+1]]:22: Performance Warning: Gather required to load value.
        float py_f = grid_in[IDX3D(OH, OW, 2, y, x, 1)];
        int32 py = floor(py_f);
        int32 px = floor(px_f);
        float py_m = py_f - py;
        float px_m = px_f - px;
        int32 px0 = clamp(px, 0, IW - 1);
        int32 px1 = clamp(px + 1, 0, IW - 1);
        int32 py0 = clamp(py, 0, IH - 1);
        int32 py1 = clamp(py + 1, 0, IH - 1);
        // CHECK-DAG: 2919.ispc:[[#@LINE+1]]:19: Performance Warning: Gather required to load value.
        float v = img_in[IDX2D(IH, IW, py0, px0)] * (1.0 - py_m) * (1.0 - px_m);
        // CHECK-DAG: 2919.ispc:[[#@LINE+1]]:14: Performance Warning: Gather required to load value.
        v += img_in[IDX2D(IH, IW, py0, px1)] * (1.0 - py_m) * px_m;
        // CHECK-DAG: 2919.ispc:[[#@LINE+1]]:14: Performance Warning: Gather required to load value.
        v += img_in[IDX2D(IH, IW, py1, px0)] * py_m * (1.0 - px_m);
        // CHECK-DAG: 2919.ispc:[[#@LINE+1]]:14: Performance Warning: Gather required to load value.
        v += img_in[IDX2D(IH, IW, py1, px1)] * py_m * px_m;
        img_out[IDX2D(OH, OW, y, x)] = v;
    }
}

export void remap_hwc(uniform float img_out[], uniform float img_in[], uniform float grid_in[],
                      uniform int32 IW, uniform int32 IH, uniform int32 C, uniform int32 OW, uniform int32 OH) {
    foreach (y = 0 ... OH, x = 0 ... OW) {
        // SSE4-DAG: 2919.ispc:[[#@LINE+2]]:22: Performance Warning: Coalesced 2 gathers starting here (other at line [[#@LINE+4]]) into 1 load (1 x 8-wide).
        // AVX-DAG: 2919.ispc:[[#@LINE+1]]:22: Performance Warning: Gather required to load value.
        float px_f = grid_in[IDX3D(OH, OW, 2, y, x, 0)];
        // AVX-DAG: 2919.ispc:[[#@LINE+1]]:22: Performance Warning: Gather required to load value.
        float py_f = grid_in[IDX3D(OH, OW, 2, y, x, 1)];
        int32 py = floor(py_f);
        int32 px = floor(px_f);
        float py_m = py_f - py;
        float px_m = px_f - px;
        int32 px0 = clamp(px, 0, IW - 1);
        int32 px1 = clamp(px + 1, 0, IW - 1);
        int32 py0 = clamp(py, 0, IH - 1);
        int32 py1 = clamp(py + 1, 0, IH - 1);
        for (uniform int32 c = 0; c < C; c++) {
            // CHECK-DAG: 2919.ispc:[[#@LINE+1]]:23: Performance Warning: Gather required to load value.
            float v = img_in[IDX3D(IH, IW, C, py0, px0, c)] * (1.0 - py_m) * (1.0 - px_m);
            // CHECK-DAG: 2919.ispc:[[#@LINE+1]]:18: Performance Warning: Gather required to load value.
            v += img_in[IDX3D(IH, IW, C, py0, px1, c)] * (1.0 - py_m) * px_m;
            // CHECK-DAG: 2919.ispc:[[#@LINE+1]]:18: Performance Warning: Gather required to load value.
            v += img_in[IDX3D(IH, IW, C, py1, px0, c)] * py_m * (1.0 - px_m);
            // CHECK-DAG: 2919.ispc:[[#@LINE+1]]:18: Performance Warning: Gather required to load value.
            v += img_in[IDX3D(IH, IW, C, py1, px1, c)] * py_m * px_m;
            // CHECK-DAG: 2919.ispc:[[#@LINE+1]]:13: Performance Warning: Scatter required to store value.
            img_out[IDX3D(OH, OW, C, y, x, c)] = v;
        }
    }
}
