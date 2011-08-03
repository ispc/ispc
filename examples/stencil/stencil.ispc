/*
  Copyright (c) 2010-2011, Intel Corporation
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


static task void
stencil_step(uniform int x0, uniform int x1,
             uniform int y0, uniform int y1,
             uniform int z0, uniform int z1,
             uniform int Nx, uniform int Ny, uniform int Nz,
             uniform const float coef[4], uniform const float vsq[],
             uniform const float Ain[], uniform float Aout[]) {
    const uniform int Nxy = Nx * Ny;

    for (uniform int z = z0; z < z1; ++z) {
        for (uniform int y = y0; y < y1; ++y) {
            // Assumes that (x1-x0) % programCount == 0
            for (uniform int x = x0; x < x1; x += programCount) {
                int index = (z * Nxy) + (y * Nx) + x + programIndex;
#define A_cur(x, y, z) Ain[index + (x) + ((y) * Nx) + ((z) * Nxy)]
#define A_next(x, y, z) Aout[index + (x) + ((y) * Nx) + ((z) * Nxy)]
                float div = coef[0] * A_cur(0, 0, 0) +
                            coef[1] * (A_cur(+1, 0, 0) + A_cur(-1, 0, 0) +
                                       A_cur(0, +1, 0) + A_cur(0, -1, 0) +
                                       A_cur(0, 0, +1) + A_cur(0, 0, -1)) +
                            coef[2] * (A_cur(+2, 0, 0) + A_cur(-2, 0, 0) +
                                       A_cur(0, +2, 0) + A_cur(0, -2, 0) +
                                       A_cur(0, 0, +2) + A_cur(0, 0, -2)) +
                            coef[3] * (A_cur(+3, 0, 0) + A_cur(-3, 0, 0) +
                                       A_cur(0, +3, 0) + A_cur(0, -3, 0) +
                                       A_cur(0, 0, +3) + A_cur(0, 0, -3));

                A_next(0, 0, 0) = 2 * A_cur(0, 0, 0) - A_next(0, 0, 0) + 
                    vsq[index] * div;
            }
        }
    }
}


export void loop_stencil_ispc(uniform int t0, uniform int t1, 
                              uniform int x0, uniform int x1,
                              uniform int y0, uniform int y1,
                              uniform int z0, uniform int z1,
                              uniform int Nx, uniform int Ny, uniform int Nz,
                              uniform const float coef[4], 
                              uniform const float vsq[],
                              uniform float Aeven[], uniform float Aodd[])
{
    for (uniform int t = t0; t < t1; ++t) {
        // Parallelize across cores as well: each task will work on a slice
        // of "dz" in the z extent of the volume.  (dz=1 seems to work
        // better than any larger values.)
        uniform int dz = 1;
        for (uniform int z = z0; z < z1; z += dz) {
            if ((t & 1) == 0)
                launch < stencil_step(x0, x1, y0, y1, z, z+dz, Nx, Ny, Nz, coef, vsq, 
                                      Aeven, Aodd) >;
            else
                launch < stencil_step(x0, x1, y0, y1, z, z+dz, Nx, Ny, Nz, coef, vsq, 
                                      Aodd, Aeven) >;
        }
        // We need to wait for all of the launched tasks to finish before
        // starting the next iteration.
        sync;
    }
}
