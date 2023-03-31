/*
  Copyright (c) 2010-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

//#define _3D_TASKING //uncomment for using "3d tasking" model. This can influence performance.

static inline int
mandel(float c_re, float c_im, int count) {
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < count; ++i) {
        if (z_re * z_re + z_im * z_im > 4.)
            break;

        float new_re = z_re*z_re - z_im*z_im;
        float new_im = 2.f * z_re * z_im;
        unmasked {
            z_re = c_re + new_re;
            z_im = c_im + new_im;
        }
    }

    return i;
}


/* Task to compute the Mandelbrot iterations for a single scanline.
 */
task void
mandelbrot_scanline(uniform float x0, uniform float dx,
                    uniform float y0, uniform float dy,
                    uniform int width, uniform int height,
#ifdef _3D_TASKING
                    uniform int xspan, uniform int yspan,
#else
                    uniform int span,
#endif
                    uniform int maxIterations, uniform int output[]) {
#ifdef _3D_TASKING
    const uniform int xstart = taskIndex0 * xspan;
    const uniform int xend   = min(xstart  + xspan, width);
    const uniform int ystart = taskIndex1 * yspan;
    const uniform int yend   = min(ystart  + yspan, height);

    foreach (yi = ystart ... yend, xi = xstart ... xend) {
#else
    uniform int ystart = taskIndex * span;
    uniform int yend = min((taskIndex+1) * span, (uniform unsigned int)height);

    foreach (yi = ystart ... yend, xi = 0 ... width) {
#endif
        float x = x0 + xi * dx;
        float y = y0 + yi * dy;

        int index = yi * width + xi;
        output[index] = mandel(x, y, maxIterations);
    }
}


export void
mandelbrot_ispc(uniform float x0, uniform float y0,
                uniform float x1, uniform float y1,
                uniform int width, uniform int height,
                uniform int maxIterations, uniform int output[]) {
    uniform float dx = (x1 - x0) / width;
    uniform float dy = (y1 - y0) / height;
#ifdef _3D_TASKING
    const uniform int xspan = max(32, programCount*2);  /* make sure it is big enough to avoid false-sharing */
    const uniform int yspan = 16;

    launch [width/xspan, height/yspan]
    mandelbrot_scanline(x0, dx, y0, dy, width, height, xspan, yspan,
                          maxIterations, output);
#else
    uniform int span = 4;

    launch[height/span] mandelbrot_scanline(x0, dx, y0, dy, width, height, span,
                                            maxIterations, output);
#endif
}
