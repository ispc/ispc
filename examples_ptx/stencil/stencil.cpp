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

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX
#pragma warning (disable: 4244)
#pragma warning (disable: 4305)
#endif

#include <cstdio>
#include <algorithm>
#include <cstring>
#include <math.h>
#include "../timing.h"
#include "../ispc_malloc.h"
#include "stencil_ispc.h"
using namespace ispc;


extern void loop_stencil_serial(int t0, int t1, int x0, int x1,
                                int y0, int y1, int z0, int z1,
                                int Nx, int Ny, int Nz,
                                const double coef[5], 
                                const double vsq[],
                                double Aeven[], double Aodd[]);


void InitData(int Nx, int Ny, int Nz, double *A[2], double *vsq) {
    int offset = 0;
    for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y)
            for (int x = 0; x < Nx; ++x, ++offset) {
                A[0][offset] = (x < Nx / 2) ? x / double(Nx) : y / double(Ny);
                A[1][offset] = 0;
                vsq[offset] = x*y*z / double(Nx * Ny * Nz);
            }
}


int main(int argc, char *argv[]) {
    static unsigned int test_iterations[] = {3, 3, 3};//the last two numbers must be equal here
    int Nx = 256, Ny = 256, Nz = 256;
    int width = 4;

    if (argc > 1) {
        if (strncmp(argv[1], "--scale=", 8) == 0) {
            double scale = atof(argv[1] + 8);
            Nx *= scale;
            Ny *= scale;
            Nz *= scale;
        }
    }
    if ((argc == 4) || (argc == 5)) {
        for (int i = 0; i < 3; i++) {
            test_iterations[i] = atoi(argv[argc - 3 + i]);
        }
    }

    double *Aserial[2], *Aispc[2];
    Aserial[0] = new double [Nx * Ny * Nz];
    Aserial[1] = new double [Nx * Ny * Nz];
    Aispc[0] = new double [Nx * Ny * Nz];
    Aispc[1] = new double [Nx * Ny * Nz];
    double *vsq = new double [Nx * Ny * Nz];
    double *coeff = new double[4];

    coeff[0] = 0.5;
    coeff[1] = -.25;
    coeff[2] = .125;
    coeff[3] = -.0625;

    InitData(Nx, Ny, Nz, Aispc, vsq);
    //
    // Compute the image using the ispc implementation on one core; report
    // the minimum time of three runs.
    // 
#if 0    
//#ifndef _CUDA_
    double minTimeISPC = 1e30;
    for (int i = 0; i < test_iterations[0]; ++i) {
        reset_and_start_timer();
        loop_stencil_ispc(0, 6, width, Nx - width, width, Ny - width,
                          width, Nz - width, Nx, Ny, Nz, coeff, vsq,
                          Aispc[0], Aispc[1]);
        double dt = get_elapsed_msec();
        printf("@time of ISPC run:\t\t\t[%.3f] msec\n", dt);
        minTimeISPC = std::min(minTimeISPC, dt);
    }

    printf("[stencil ispc 1 core]:\t\t[%.3f] msec\n", minTimeISPC);

    InitData(Nx, Ny, Nz, Aispc, vsq);
#endif

    //
    // Compute the image using the ispc implementation with tasks; report
    // the minimum time of three runs.
    //
    double minTimeISPCTasks = 1e30;
    for (int i = 0; i < test_iterations[1]; ++i) {
        reset_and_start_timer();
        loop_stencil_ispc_tasks(0, 6, width, Nx - width, width, Ny - width,
                                width, Nz - width, Nx, Ny, Nz, coeff, vsq,
                                Aispc[0], Aispc[1]);
        double dt = get_elapsed_msec();
        printf("@time of ISPC + TASKS run:\t\t\t[%.3f] msec\n", dt);
        minTimeISPCTasks = std::min(minTimeISPCTasks, dt);
    }

    printf("[stencil ispc + tasks]:\t\t[%.3f] msec\n", minTimeISPCTasks);

    InitData(Nx, Ny, Nz, Aserial, vsq);

    // 
    // And run the serial implementation 3 times, again reporting the
    // minimum time.
    //
    double minTimeSerial = 1e30;
    for (int i = 0; i < test_iterations[2]; ++i) {
        reset_and_start_timer();
        loop_stencil_serial(0, 6, width, Nx-width, width, Ny - width,
                            width, Nz - width, Nx, Ny, Nz, coeff, vsq,
                            Aserial[0], Aserial[1]);
        double dt = get_elapsed_msec();
        printf("@time of serial run:\t\t\t[%.3f] msec\n", dt);
        minTimeSerial = std::min(minTimeSerial, dt);
    }

    printf("[stencil serial]:\t\t[%.3f] msec\n", minTimeSerial);

#if 0
//#ifndef _CUDA_
    printf("\t\t\t\t(%.2fx speedup from ISPC, %.2fx speedup from ISPC + tasks)\n", 
           minTimeSerial / minTimeISPC, minTimeSerial / minTimeISPCTasks);
#else
    printf("\t\t\t\t(%.2fx speedup from ISPC + tasks)\n", 
           minTimeSerial / minTimeISPCTasks);
#endif

    // Check for agreement
    int offset = 0;
    for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y)
            for (int x = 0; x < Nx; ++x, ++offset) {
                double error = fabsf((Aserial[1][offset] - Aispc[1][offset]) /
                                    Aserial[1][offset]);
                if (error > 1e-4)
                    printf("Error @ (%d,%d,%d): ispc = %f, serial = %f\n",
                           x, y, z, Aispc[1][offset], Aserial[1][offset]);
            }

    return 0;
}
