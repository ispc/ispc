/*
  Copyright (c) 2010-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/


static void
stencil_step(uniform int x0, uniform int x1,
             uniform int y0, uniform int y1,
             uniform int z0, uniform int z1,
             uniform int Nx, uniform int Ny, uniform int Nz,
             uniform const float coef[4], uniform const float vsq[],
             uniform const float Ain[], uniform float Aout[]) {
    const uniform int Nxy = Nx * Ny;

    foreach (z = z0 ... z1, y = y0 ... y1, x = x0 ... x1) {
        int index = (z * Nxy) + (y * Nx) + x;
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


static task void
stencil_step_task(uniform int x0, uniform int x1,
                  uniform int y0, uniform int y1,
                  uniform int z0,
                  uniform int Nx, uniform int Ny, uniform int Nz,
                  uniform const float coef[4], uniform const float vsq[],
                  uniform const float Ain[], uniform float Aout[]) {
    stencil_step(x0, x1, y0, y1, z0+taskIndex, z0+taskIndex+1,
                 Nx, Ny, Nz, coef, vsq, Ain, Aout);
}


export void
loop_stencil_ispc_tasks(uniform int t0, uniform int t1,
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
        // of 1 in the z extent of the volume.
        if ((t & 1) == 0)
            launch[z1-z0] stencil_step_task(x0, x1, y0, y1, z0, Nx, Ny, Nz,
                                            coef, vsq, Aeven, Aodd);
        else
            launch[z1-z0] stencil_step_task(x0, x1, y0, y1, z0, Nx, Ny, Nz,
                                            coef, vsq, Aodd, Aeven);

        // We need to wait for all of the launched tasks to finish before
        // starting the next iteration.
        sync;
    }
}


export void
loop_stencil_ispc(uniform int t0, uniform int t1,
                  uniform int x0, uniform int x1,
                  uniform int y0, uniform int y1,
                  uniform int z0, uniform int z1,
                  uniform int Nx, uniform int Ny, uniform int Nz,
                  uniform const float coef[4],
                  uniform const float vsq[],
                  uniform float Aeven[], uniform float Aodd[])
{
    for (uniform int t = t0; t < t1; ++t) {
        if ((t & 1) == 0)
            stencil_step(x0, x1, y0, y1, z0, z1, Nx, Ny, Nz, coef, vsq,
                         Aeven, Aodd);
        else
            stencil_step(x0, x1, y0, y1, z0, z1, Nx, Ny, Nz, coef, vsq,
                         Aodd, Aeven);
    }
}
