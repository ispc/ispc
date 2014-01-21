/*
  Copyright (c) 2010-2012, Intel Corporation
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
    uniform int yend = min((taskIndex+1) * span, (unsigned int)height);

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
