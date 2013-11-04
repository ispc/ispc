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

#include <cassert>
#include <iostream>
#include <cuda.h>
#include "drvapi_error_string.h"

#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)
// These are the inline versions for all of the SDK helper functions
void __checkCudaErrors(CUresult err, const char *file, const int line) {
  if(CUDA_SUCCESS != err) {
    std::cerr << "checkCudeErrors() Driver API error = " << err << "\""
           << getCudaDrvErrorString(err) << "\" from file <" << file
           << ", line " << line << "\n";
    exit(-1);
  }
}

/**********************/
/* Basic CUDriver API */
CUcontext context;

void createContext(const int deviceId = 0)
{
  CUdevice device;
  int devCount;
  checkCudaErrors(cuInit(0));
  checkCudaErrors(cuDeviceGetCount(&devCount));
  assert(devCount > 0);
  checkCudaErrors(cuDeviceGet(&device, deviceId < devCount ? deviceId : 0));

  char name[128];
  checkCudaErrors(cuDeviceGetName(name, 128, device));
  std::cout << "Using CUDA Device [0]: " << name << "\n";

  int devMajor, devMinor;
  checkCudaErrors(cuDeviceComputeCapability(&devMajor, &devMinor, device));
  std::cout << "Device Compute Capability: " 
    << devMajor << "." << devMinor << "\n";
  if (devMajor < 2) {
    std::cerr << "ERROR: Device 0 is not SM 2.0 or greater\n";
    exit(1); 
  }

  // Create driver context
  checkCudaErrors(cuCtxCreate(&context, 0, device));
}
void destroyContext()
{
  checkCudaErrors(cuCtxDestroy(context));
}

CUmodule loadModule(const char * module)
{
  CUmodule cudaModule;
  checkCudaErrors(cuModuleLoadData(&cudaModule, module));
  return cudaModule;
}
void unloadModule(CUmodule &cudaModule)
{
  checkCudaErrors(cuModuleUnload(cudaModule));
}

CUfunction getFunction(CUmodule &cudaModule, const char * function)
{
  CUfunction cudaFunction;
  checkCudaErrors(cuModuleGetFunction(&cudaFunction, cudaModule, function));
  return cudaFunction;
}
  
CUdeviceptr deviceMalloc(const size_t size)
{
  CUdeviceptr d_buf;
  checkCudaErrors(cuMemAlloc(&d_buf, size));
  return d_buf;
}
void deviceFree(CUdeviceptr d_buf)
{
  checkCudaErrors(cuMemFree(d_buf));
}
void memcpyD2H(void * h_buf, CUdeviceptr d_buf, const size_t size)
{
  checkCudaErrors(cuMemcpyDtoH(h_buf, d_buf, size));
}
void memcpyH2D(CUdeviceptr d_buf, void * h_buf, const size_t size)
{
  checkCudaErrors(cuMemcpyHtoD(d_buf, h_buf, size));
}
#define deviceLaunch(func,nbx,nby,nbz,params) \
  checkCudaErrors( \
      cuLaunchKernel( \
        (func), \
        (nbx), (nby), (nbz), \
        32, 1, 1, \
        0, NULL, (params), NULL \
        ));

typedef CUdeviceptr devicePtr;


/**************/

extern "C" 
{

  void *CUDAAlloc(void **handlePtr, int64_t size, int32_t alignment)
  {
    return NULL;
  }
  void CUDALaunch(
      void **handlePtr, 
      const char * module_name,
      const char * module, 
      const char * func_name,
      void **func_args, 
      int countx, int county, int countz)
  {
    assert(module_name != NULL);
    assert(module != NULL);
    assert(func_name != NULL);
    assert(func_args != NULL);
    CUmodule   cudaModule   = loadModule(module);
    CUfunction cudaFunction = getFunction(cudaModule, func_name);
    deviceLaunch(cudaFunction, countx, county, countz, func_args);
    unloadModule(cudaModule);
  }
  void CUDASync(void *handle)
  {
    checkCudaErrors(cuStreamSynchronize(0));
  }
  void ISPCSync(void *handle)
  {
    checkCudaErrors(cuStreamSynchronize(0));
  }
  void CUDAFree(void *handle)
  {
  }
}


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
    
    /*******************/
    createContext();
    /*******************/

    const size_t bufsize = sizeof(double)*Nx*Ny*Nz;
    devicePtr d_Aispc0   = deviceMalloc(bufsize);
    devicePtr d_Aispc1   = deviceMalloc(bufsize);
    devicePtr d_vsq      = deviceMalloc(bufsize);
    devicePtr d_coeff    = deviceMalloc(4*sizeof(double));


    InitData(Nx, Ny, Nz, Aispc, vsq);

    //
    // Compute the image using the ispc implementation on one core; report
    // the minimum time of three runs.
    //
    double minTimeISPC = 1e30;
    for (int i = 0; i < 3; ++i) {
        reset_and_start_timer();
        loop_stencil_ispc(0, 6, width, Nx - width, width, Ny - width,
                          width, Nz - width, Nx, Ny, Nz, coeff, vsq,
                          Aispc[0], Aispc[1]);
        double dt = get_elapsed_mcycles();
        minTimeISPC = std::min(minTimeISPC, dt);
    }

    printf("[stencil ispc 1 core]:\t\t[%.3f] million cycles\n", minTimeISPC);
  
    InitData(Nx, Ny, Nz, Aispc, vsq);

    memcpyH2D(d_Aispc0, Aispc[0], bufsize);
    memcpyH2D(d_Aispc1, Aispc[1], bufsize);
    memcpyH2D(d_vsq,    vsq,      bufsize);
    memcpyH2D(d_coeff,  coeff,    4*sizeof(double));
    //
    // Compute the image using the ispc implementation with tasks; report
    // the minimum time of three runs.
    //
    double minTimeISPCTasks = 1e30;
    for (int i = 0; i < 3; ++i) {
        reset_and_start_timer();
        loop_stencil_ispc_tasks(0, 6, width, Nx - width, width, Ny - width,
                                width, Nz - width, Nx, Ny, Nz, (double*)d_coeff, (double*)d_vsq,
                                (double*)d_Aispc0, (double*)d_Aispc1);
        double dt = get_elapsed_mcycles();
        minTimeISPCTasks = std::min(minTimeISPCTasks, dt);
    }
    memcpyD2H(Aispc[1], d_Aispc1, bufsize);
    //memcpyD2H(Aispc[1], d_vsq, bufsize);

    printf("[stencil ispc + tasks]:\t\t[%.3f] million cycles\n", minTimeISPCTasks);

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
    int nerr = 0;
    for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y)
            for (int x = 0; x < Nx; ++x, ++offset) {

                double error = fabsf((Aserial[1][offset] - Aispc[1][offset]) /
                                    Aserial[1][offset]);
                if (error > 1e-3)
                {
                  if (nerr < 100)
                    printf("Error @ (%d,%d,%d): ispc = %g, serial = %g error= %g\n",
                           x, y, z, Aispc[1][offset], Aserial[1][offset], error);
                    nerr++;
                }
            }

    fprintf(stderr, " nerr= %d  frac= %g \n", nerr, 1.0*nerr/(1.0*Nx*Ny*Nz));
  
    /*******************/
    destroyContext();
    /*******************/

    return 0;
}
