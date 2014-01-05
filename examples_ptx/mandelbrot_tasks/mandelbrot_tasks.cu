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

#include "cuda_helpers.cuh"

__device__
static inline int
mandel(float c_re, float c_im, int count) {
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < count; ++i) {
        if (z_re * z_re + z_im * z_im > 4.0f)
            break;

        float new_re = z_re*z_re - z_im*z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return i;
}


/* Task to compute the Mandelbrot iterations for a single scanline.
 */
__global__ void
mandelbrot_scanline( float x0,  float dx, 
                     float y0,  float dy,
                     int width,  int height, 
                     int xspan,  int yspan,
                     int maxIterations,  int output[]) {
    const  int xstart = taskIndex0 * xspan;
    const  int xend   = min(xstart  + xspan, width);

    const  int ystart = taskIndex1 * yspan;
    const  int yend   = min(ystart  + yspan, height);
   
    for ( int yi = ystart; yi < yend; yi++)
      for ( int xi = xstart+programIndex; xi < xend; xi += programCount)
      {
        const float x = x0 + xi * dx;
        const float y = y0 + yi * dy;

        const int res = mandel(x,y,maxIterations);
        const int index = yi * width + xi;
        if (xi < xend)
          output[index] = res;
      }
}

extern "C" __global__ void
mandelbrot_ispc___export( float x0,  float y0, 
                 float x1,  float y1,
                 int width,  int height, 
                 int maxIterations,  int output[]) {
     float dx = (x1 - x0) / width;
     float dy = (y1 - y0) / height;
     const  int xspan = 64;  /* make sure it is big enough to avoid false-sharing */
     const  int yspan = 16;


    launch(width/xspan, height/yspan, 1, mandelbrot_scanline)
      (x0, dx, y0, dy, width, height, xspan, yspan,  maxIterations, output);
    cudaDeviceSynchronize();
}

extern "C" __host__ void
mandelbrot_ispc( float x0,  float y0, 
                 float x1,  float y1,
                 int width,  int height, 
                 int maxIterations,  int output[]) 
{
  mandelbrot_ispc___export<<<1,32>>>(x0,y0,x1,y1,width,height,maxIterations,output);
  cudaDeviceSynchronize();
}
