// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// Bilinear remap of an image through a grid of sample coordinates (discussion #2919), in HW (one channel) and HWC
// (C interleaved channels) layouts. Every output element reads 4 data-dependent image elements, which are genuine
// gathers, and the 2 adjacent coordinates of its grid point, a stride-2 read.
#define IDX2D(H, W, y, x) ((y) * (W) + (x))
#define IDX3D(H, W, C, y, x, c) (((y) * (W) + (x)) * (C) + (c))

export void remap(uniform float img_out[], uniform float img_in[], uniform float grid_in[], uniform int32 IW,
                  uniform int32 IH, uniform int32 OW, uniform int32 OH) {
    foreach (y = 0 ... OH, x = 0 ... OW) {
        float px_f = grid_in[IDX3D(OH, OW, 2, y, x, 0)];
        float py_f = grid_in[IDX3D(OH, OW, 2, y, x, 1)];
        int32 py = floor(py_f);
        int32 px = floor(px_f);
        float py_m = py_f - py;
        float px_m = px_f - px;
        int32 px0 = clamp(px, 0, IW - 1);
        int32 px1 = clamp(px + 1, 0, IW - 1);
        int32 py0 = clamp(py, 0, IH - 1);
        int32 py1 = clamp(py + 1, 0, IH - 1);
        float v = img_in[IDX2D(IH, IW, py0, px0)] * (1.0 - py_m) * (1.0 - px_m);
        v += img_in[IDX2D(IH, IW, py0, px1)] * (1.0 - py_m) * px_m;
        v += img_in[IDX2D(IH, IW, py1, px0)] * py_m * (1.0 - px_m);
        v += img_in[IDX2D(IH, IW, py1, px1)] * py_m * px_m;
        img_out[IDX2D(OH, OW, y, x)] = v;
    }
}

export void remap_hwc(uniform float img_out[], uniform float img_in[], uniform float grid_in[], uniform int32 IW,
                      uniform int32 IH, uniform int32 C, uniform int32 OW, uniform int32 OH) {
    foreach (y = 0 ... OH, x = 0 ... OW) {
        float px_f = grid_in[IDX3D(OH, OW, 2, y, x, 0)];
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
            float v = img_in[IDX3D(IH, IW, C, py0, px0, c)] * (1.0 - py_m) * (1.0 - px_m);
            v += img_in[IDX3D(IH, IW, C, py0, px1, c)] * (1.0 - py_m) * px_m;
            v += img_in[IDX3D(IH, IW, C, py1, px0, c)] * py_m * (1.0 - px_m);
            v += img_in[IDX3D(IH, IW, C, py1, px1, c)] * py_m * px_m;
            img_out[IDX3D(OH, OW, C, y, x, c)] = v;
        }
    }
}

// References: the same, with the grid read as vector loads and a deinterleave (aos_to_soa2) for full gangs.
static inline void sample(uniform float img_out[], uniform float img_in[], uniform int32 IW, uniform int32 IH,
                          uniform int32 C, uniform int32 OW, uniform int32 y, int x, float px_f, float py_f) {
    int32 py = floor(py_f);
    int32 px = floor(px_f);
    float py_m = py_f - py;
    float px_m = px_f - px;
    int32 px0 = clamp(px, 0, IW - 1);
    int32 px1 = clamp(px + 1, 0, IW - 1);
    int32 py0 = clamp(py, 0, IH - 1);
    int32 py1 = clamp(py + 1, 0, IH - 1);
    for (uniform int32 c = 0; c < C; c++) {
        float v = img_in[IDX3D(IH, IW, C, py0, px0, c)] * (1.0 - py_m) * (1.0 - px_m);
        v += img_in[IDX3D(IH, IW, C, py0, px1, c)] * (1.0 - py_m) * px_m;
        v += img_in[IDX3D(IH, IW, C, py1, px0, c)] * py_m * (1.0 - px_m);
        v += img_in[IDX3D(IH, IW, C, py1, px1, c)] * py_m * px_m;
        img_out[IDX3D(OH, OW, C, y, x, c)] = v;
    }
}

static inline void remap_c(uniform float img_out[], uniform float img_in[], uniform float grid_in[], uniform int32 IW,
                           uniform int32 IH, uniform int32 C, uniform int32 OW, uniform int32 OH) {
    for (uniform int32 y = 0; y < OH; y++) {
        uniform int32 x0 = 0;
        for (; x0 + programCount <= OW; x0 += programCount) {
            float px_f, py_f;
            aos_to_soa2(&grid_in[IDX3D(OH, OW, 2, y, x0, 0)], &px_f, &py_f);
            sample(img_out, img_in, IW, IH, C, OW, y, x0 + programIndex, px_f, py_f);
        }
        foreach (x = x0 ... OW) {
            sample(img_out, img_in, IW, IH, C, OW, y, x, grid_in[IDX3D(OH, OW, 2, y, x, 0)],
                   grid_in[IDX3D(OH, OW, 2, y, x, 1)]);
        }
    }
}

export void remap_deinterleave(uniform float img_out[], uniform float img_in[], uniform float grid_in[],
                               uniform int32 IW, uniform int32 IH, uniform int32 OW, uniform int32 OH) {
    remap_c(img_out, img_in, grid_in, IW, IH, 1, OW, OH);
}

export void remap_hwc_deinterleave(uniform float img_out[], uniform float img_in[], uniform float grid_in[],
                                   uniform int32 IW, uniform int32 IH, uniform int32 C, uniform int32 OW,
                                   uniform int32 OH) {
    remap_c(img_out, img_in, grid_in, IW, IH, C, OW, OH);
}
