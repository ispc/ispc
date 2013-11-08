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

#include <stdio.h>
#include <algorithm>
#include <math.h>
#include "../timing.h"
#include "stencil_ispc.h"
using namespace ispc;

#include <sys/time.h>


double rtc(void)
{
  struct timeval Tvalue;
  double etime;
  struct timezone dummy;

  gettimeofday(&Tvalue,&dummy);
  etime =  (double) Tvalue.tv_sec +
    1.e-6*((double) Tvalue.tv_usec);
  return etime;
}


extern void loop_stencil_serial(int t0, int t1, int x0, int x1,
    int y0, int y1, int z0, int z1,
    int Nx, int Ny, int Nz,
    const double coef[5], 
    const double vsq[],
    double Aeven[], double Aodd[]);


void InitData(int Nx, int Ny, int Nz, double *A[2], double *vsq) {
  int offset = 0;
#pragma omp parallel for collapse(2) private(offset)
  for (int z = 0; z < Nz; ++z)
    for (int y = 0; y < Ny; ++y)
      for (int x = 0; x < Nx; ++x, ++offset) {
        A[0][offset] = (x < Nx / 2) ? x / double(Nx) : y / double(Ny);
        A[1][offset] = 0;
        vsq[offset] = x*y*z / double(Nx * Ny * Nz);
      }
}


int main() {
  int Nx = 256, Ny = 256, Nz = 256;
  int width = 4;
  double *Aserial[2], *Aispc[2];
  Aserial[0] = new double [Nx * Ny * Nz];
  Aserial[1] = new double [Nx * Ny * Nz];
  Aispc[0] = new double [Nx * Ny * Nz];
  Aispc[1] = new double [Nx * Ny * Nz];
  double *vsq = new double [Nx * Ny * Nz];

  double coeff[4] = { 0.5, -.25, .125, -.0625 }; 

//  InitData(Nx, Ny, Nz, Aispc, vsq);

  //
  // Compute the image using the ispc implementation on one core; report
  // the minimum time of three runs.
  //
  double minTimeISPC = 1e30;
#if 0
  for (int i = 0; i < 3; ++i) {
    reset_and_start_timer();
    loop_stencil_ispc(0, 6, width, Nx - width, width, Ny - width,
        width, Nz - width, Nx, Ny, Nz, coeff, vsq,
        Aispc[0], Aispc[1]);
    double dt = get_elapsed_mcycles();
    minTimeISPC = std::min(minTimeISPC, dt);
  }

  printf("[stencil ispc 1 core]:\t\t[%.3f] million cycles\n", minTimeISPC);
#endif

  fprintf(stderr, " -- init -- \n");
  InitData(Nx, Ny, Nz, Aispc, vsq);
  fprintf(stderr, " -- done init -- \n");

  //
  // Compute the image using the ispc implementation with tasks; report
  // the minimum time of three runs.
  //
  double minTimeISPCTasks = 1e30;
  for (int i = 0; i < 3; ++i) {
    reset_and_start_timer();
    const double t0 = rtc();
    loop_stencil_ispc_tasks(0, 6, width, Nx - width, width, Ny - width,
        width, Nz - width, Nx, Ny, Nz, coeff, vsq,
        Aispc[0], Aispc[1]);
    double dt = rtc() - t0; //get_elapsed_mcycles();
    minTimeISPCTasks = std::min(minTimeISPCTasks, dt);
  }

  fprintf(stderr, "[stencil ispc + tasks]:\t\t[%.3f] million cycles\n", minTimeISPCTasks);


  InitData(Nx, Ny, Nz, Aserial, vsq);

  // 
  // And run the serial implementation 3 times, again reporting the
  // minimum time.
  //
  double minTimeSerial = 1e30;
  for (int i = 0; i < 3; ++i) {
    reset_and_start_timer();
    loop_stencil_serial(0, 6, width, Nx-width, width, Ny - width,
        width, Nz - width, Nx, Ny, Nz, coeff, vsq,
        Aserial[0], Aserial[1]);
    double dt = get_elapsed_mcycles();
    minTimeSerial = std::min(minTimeSerial, dt);
  }

  printf("[stencil serial]:\t\t[%.3f] million cycles\n", minTimeSerial);

  printf("\t\t\t\t(%.2fx speedup from ISPC, %.2fx speedup from ISPC + tasks)\n", 
      minTimeSerial / minTimeISPC, minTimeSerial / minTimeISPCTasks);

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
