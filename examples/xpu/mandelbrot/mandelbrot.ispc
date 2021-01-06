/*
  Copyright (c) 2010-2020, Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.


   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
