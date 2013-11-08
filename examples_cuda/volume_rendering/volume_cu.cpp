/*
  Copyright (c) 2011, Intel Corporation
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
#include "../timing.h"
#include "volume_ispc.h"
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
  // in this branch we use compilation with parameters
  const unsigned int jitNumOptions = 1;
  CUjit_option *jitOptions = new CUjit_option[jitNumOptions];
  void **jitOptVals = new void*[jitNumOptions];
  // set up pointer to set the Maximum # of registers for a particular kernel
  jitOptions[0] = CU_JIT_MAX_REGISTERS;
  int jitRegCount = 64;
  jitOptVals[0] = (void *)(size_t)jitRegCount;

#if 0
        // set up size of compilation log buffer
        jitOptions[0] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
        int jitLogBufferSize = 1024;
        jitOptVals[0] = (void *)(size_t)jitLogBufferSize;

        // set up pointer to the compilation log buffer
        jitOptions[1] = CU_JIT_INFO_LOG_BUFFER;
        char *jitLogBuffer = new char[jitLogBufferSize];
        jitOptVals[1] = jitLogBuffer;

        // set up pointer to set the Maximum # of registers for a particular kernel
        jitOptions[2] = CU_JIT_MAX_REGISTERS;
        int jitRegCount = 32;
        jitOptVals[2] = (void *)(size_t)jitRegCount;
#endif

  checkCudaErrors(cuModuleLoadDataEx(&cudaModule, module,jitNumOptions, jitOptions, (void **)jitOptVals));
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
  checkCudaErrors(cuFuncSetCacheConfig((func), CU_FUNC_CACHE_PREFER_L1)); \
  checkCudaErrors( \
      cuLaunchKernel( \
        (func), \
        ((nbx-1)/(128/32)+1), (nby), (nbz), \
        128, 1, 1, \
        0, NULL, (params), NULL \
        ));

typedef CUdeviceptr devicePtr;


/**************/
#include <vector>
std::vector<char> readBinary(const char * filename)
{
  std::vector<char> buffer;
  FILE *fp = fopen(filename, "rb");
  if (!fp )
  {
    fprintf(stderr, "file %s not found\n", filename);
    assert(0);
  }
#if 0
  char c;
  while ((c = fgetc(fp)) != EOF)
    buffer.push_back(c);
#else
  fseek(fp, 0, SEEK_END); 
  const unsigned long long size = ftell(fp);         /*calc the size needed*/
  fseek(fp, 0, SEEK_SET); 
  buffer.resize(size);

  if (fp == NULL){ /*ERROR detection if file == empty*/
    fprintf(stderr, "Error: There was an Error reading the file %s \n",filename);           
    exit(1);
  }
  else if (fread(&buffer[0], sizeof(char), size, fp) != size){ /* if count of read bytes != calculated size of .bin file -> ERROR*/
    fprintf(stderr, "Error: There was an Error reading the file %s \n", filename);
    exit(1);
  }
#endif
  fprintf(stderr, " read buffer of size= %d bytes \n", (int)buffer.size());
  return buffer;
}

extern "C" 
{

  void *CUDAAlloc(void **handlePtr, int64_t size, int32_t alignment)
  {
    return NULL;
  }
  void CUDALaunch(
      void **handlePtr, 
      const char * module_name,
      const char * module_1,
      const char * func_name,
      void **func_args, 
      int countx, int county, int countz)
  {
    assert(module_name != NULL);
    assert(module_1 != NULL);
    assert(func_name != NULL);
    assert(func_args != NULL);
#if 1
    const char * module = module_1;
#else
    const std::vector<char> module_str = readBinary("kernel.cubin");
    const char *  module = &module_str[0];
#endif
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

extern void volume_serial(float density[], int nVoxels[3], 
                          const float raster2camera[4][4],
                          const float camera2world[4][4], 
                          int width, int height, float image[]);

/* Write a PPM image file with the image */
static void
writePPM(float *buf, int width, int height, const char *fn) {
    FILE *fp = fopen(fn, "wb");
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "255\n");
    for (int i = 0; i < width*height; ++i) {
        float v = buf[i] * 255.f;
        if (v < 0.f) v = 0.f;
        else if (v > 255.f) v = 255.f;
        unsigned char c = (unsigned char)v;
        for (int j = 0; j < 3; ++j)
            fputc(c, fp);
    }
    fclose(fp);
    printf("Wrote image file %s\n", fn);
}


/* Load image and viewing parameters from a camera data file.
   FIXME: we should add support to be able to specify viewing parameters
   in the program here directly. */
static void
loadCamera(const char *fn, int *width, int *height, float raster2camera[4][4],
           float camera2world[4][4]) {
    FILE *f = fopen(fn, "r");
    if (!f) {
        perror(fn);
        exit(1);
    }
    if (fscanf(f, "%d %d", width, height) != 2) {
        fprintf(stderr, "Unexpected end of file in camera file\n");
        exit(1);
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (fscanf(f, "%f", &raster2camera[i][j]) != 1) {
                fprintf(stderr, "Unexpected end of file in camera file\n");
                exit(1);
            }
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (fscanf(f, "%f", &camera2world[i][j]) != 1) {
                fprintf(stderr, "Unexpected end of file in camera file\n");
                exit(1);
            }
        }
    }
    fclose(f);
}


/* Load a volume density file.  Expects the number of x, y, and z samples
   as the first three values (as integer strings), then x*y*z
   floating-point values (also as strings) to give the densities.  */
static float *
loadVolume(const char *fn, int n[3]) {
    FILE *f = fopen(fn, "r");
    if (!f) {
        perror(fn);
        exit(1);
    }

    if (fscanf(f, "%d %d %d", &n[0], &n[1], &n[2]) != 3) {
        fprintf(stderr, "Couldn't find resolution at start of density file\n");
        exit(1);
    }

    int count = n[0] * n[1] * n[2];
    float *v = new float[count];
    for (int i = 0; i < count; ++i) {
        if (fscanf(f, "%f", &v[i]) != 1) {
            fprintf(stderr, "Unexpected end of file at %d'th density value\n", i);
            exit(1);
        }
    }

    return v;
}


int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "usage: volume <camera.dat> <volume_density.vol>\n");
        return 1;
    }

    //
    // Load viewing data and the volume density data
    //
    int width, height;
    float raster2camera[4][4], camera2world[4][4];
    loadCamera(argv[1], &width, &height, raster2camera, camera2world);
    float *image = new float[width*height];

    int n[3];
    float *density = loadVolume(argv[2], n);

    /*******************/
    createContext();
    /*******************/

    devicePtr d_raster2camera = deviceMalloc(4*4*sizeof(float));
    devicePtr d_camera2world  = deviceMalloc(4*4*sizeof(float));
    devicePtr d_n             = deviceMalloc(3*sizeof(int));
    devicePtr d_density       = deviceMalloc(n[0]*n[1]*n[2]*sizeof(float));
    devicePtr d_image         = deviceMalloc(width*height*sizeof(float));



    //
    // Compute the image using the ispc implementation; report the minimum
    // time of three runs.
    //
    double minISPC = 1e30;
#if 0
    for (int i = 0; i < 3; ++i) {
        reset_and_start_timer();
        volume_ispc(density, n, raster2camera, camera2world,
                    width, height, image);
        double dt = get_elapsed_mcycles();
        minISPC = std::min(minISPC, dt);
    }

    for (int i = 0; i < width*height; i += 4)
    {
      if (image[i] != 0.0f)
      {
        fprintf(stderr, " i= %d  image= %g %g %g %g \n", i,
            image[i+0],
            image[i+1],
            image[i+2],
            image[i+3]);
        break;
      }
    }

    printf("[volume ispc 1 core]:\t\t[%.3f] million cycles\n", minISPC);
    writePPM(image, width, height, "volume-ispc-1core.ppm");
#endif
    

    // Clear out the buffer
    for (int i = 0; i < width * height; ++i)
        image[i] = 0.;

    memcpyH2D(d_raster2camera, raster2camera, 4*4*sizeof(float));
    memcpyH2D(d_camera2world,  camera2world,  4*4*sizeof(float));
    memcpyH2D(d_n,             n,             3*sizeof(int));
    memcpyH2D(d_density,       density,       n[0]*n[1]*n[2]*sizeof(float));
    memcpyH2D(d_image,         image,         width*height*sizeof(float));

    //
    // Compute the image using the ispc implementation that also uses
    // tasks; report the minimum time of three runs.
    //
    double minISPCtasks = 1e30;
    for (int i = 0; i < 3; ++i) {
        reset_and_start_timer();
        volume_ispc_tasks(
              (float*)d_density, 
              (int*)d_n, 
              (float(*)[4])d_raster2camera, 
              (float(*)[4])d_camera2world,
              width, height, 
              (float*)d_image);
        double dt = get_elapsed_mcycles();
        minISPCtasks = std::min(minISPCtasks, dt);
    }

    memcpyD2H(image, d_image, width*height*sizeof(float));
    for (int i = 0; i < width*height; i += 4)
    {
      if (image[i] != 0.0f)
      {
        fprintf(stderr, " i= %d  image= %g %g %g %g \n", i,
            image[i+0],
            image[i+1],
            image[i+2],
            image[i+3]);
        break;
      }
    }

    printf("[volume ispc + tasks]:\t\t[%.3f] million cycles\n", minISPCtasks);
    writePPM(image, width, height, "volume-cuda.ppm");

#if 0
    // Clear out the buffer
    for (int i = 0; i < width * height; ++i)
        image[i] = 0.;

    // 
    // And run the serial implementation 3 times, again reporting the
    // minimum time.
    //
    double minSerial = 1e30;
    for (int i = 0; i < 3; ++i) {
        reset_and_start_timer();
        volume_serial(density, n, raster2camera, camera2world,
                      width, height, image);
        double dt = get_elapsed_mcycles();
        minSerial = std::min(minSerial, dt);
    }

    printf("[volume serial]:\t\t[%.3f] million cycles\n", minSerial);
    writePPM(image, width, height, "volume-serial.ppm");

    printf("\t\t\t\t(%.2fx speedup from ISPC, %.2fx speedup from ISPC + tasks)\n", 
           minSerial/minISPC, minSerial / minISPCtasks);
#else
    printf("\t\t\t\t %.2fx speedup from ISPC + tasks)\n", 
            minISPC / minISPCtasks);
#endif
  
    /*******************/
    destroyContext();
    /*******************/

    return 0;
}
