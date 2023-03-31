/*
  Copyright (c) 2010-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/
static inline int mandel(float c_re, float c_im, uniform int count) {
    float z_re = c_re, z_im = c_im;
    int j = 0;
    for (uniform int i = 0; i < count; ++i) {
        bool done = (z_re * z_re + z_im * z_im > 4.f);
        if (all(done))
            break;

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;

        if (!done) {
            z_re = c_re + new_re;
            z_im = c_im + new_im;

            j++;
        }
    }

    return j;
}

struct Parameters {
    float x0;
    float dx;
    float y0;
    float dy;
    int width;
    int height;
    int tile_size;
    int maxIterations;
    int *output;
};

task void mandelbrot_tile(void *uniform _p) {
    Parameters *uniform p = (Parameters * uniform) _p;

    uniform const int x_begin = taskIndex0 * p->tile_size;
    uniform const int y_begin = taskIndex1 * p->tile_size;

    uniform const int x_edge = x_begin + p->tile_size;
    uniform const int y_edge = y_begin + p->tile_size;

    uniform const int x_end = min(x_edge, p->width);
    uniform const int y_end = min(y_edge, p->height);

    foreach (yi = y_begin... y_end, xi = x_begin... x_end) {
        const float x = p->x0 + xi * p->dx;
        const float y = p->y0 + yi * p->dy;

        const int index = yi * p->width + xi;
        p->output[index] = mandel(x, y, p->maxIterations);
    }
}

#include "ispcrt.isph"
DEFINE_CPU_ENTRY_POINT(mandelbrot_tile)
