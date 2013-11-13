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
#include <string.h>
#include "../timing.h"

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

#include <iostream>
#include <cuda.h>
#include <vector>
#include <cassert>
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
  const double t0 = rtc();
  CUmodule cudaModule;
  // in this branch we use compilation with parameters

#if 0
  unsigned int jitNumOptions = 1;
  CUjit_option *jitOptions = new CUjit_option[jitNumOptions];
  void **jitOptVals = new void*[jitNumOptions];
  // set up pointer to set the Maximum # of registers for a particular kernel
  jitOptions[0] = CU_JIT_MAX_REGISTERS;
  int jitRegCount = 64;
  jitOptVals[0] = (void *)(size_t)jitRegCount;
#if 0

  {
    jitNumOptions = 3;
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
  }
#endif

  checkCudaErrors(cuModuleLoadDataEx(&cudaModule, module,jitNumOptions, jitOptions, (void **)jitOptVals));
#else
  CUlinkState  CUState;
  CUlinkState *lState = &CUState;
  const int nOptions = 7;
    CUjit_option options[nOptions];
    void* optionVals[nOptions];
    float walltime;
    const unsigned int logSize = 32768;
    char error_log[logSize],
         info_log[logSize];
    void *cuOut;
    size_t outSize;
    int myErr = 0;

    // Setup linker options
    // Return walltime from JIT compilation
    options[0] = CU_JIT_WALL_TIME;
    optionVals[0] = (void*) &walltime;
    // Pass a buffer for info messages
    options[1] = CU_JIT_INFO_LOG_BUFFER;
    optionVals[1] = (void*) info_log;
    // Pass the size of the info buffer
    options[2] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
    optionVals[2] = (void*) logSize;
    // Pass a buffer for error message
    options[3] = CU_JIT_ERROR_LOG_BUFFER;
    optionVals[3] = (void*) error_log;
    // Pass the size of the error buffer
    options[4] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
    optionVals[4] = (void*) logSize;
    // Make the linker verbose
    options[5] = CU_JIT_LOG_VERBOSE;
    optionVals[5] = (void*) 1;
    // Max # of registers/pthread
    options[6] = CU_JIT_MAX_REGISTERS;
    int jitRegCount = 48;
    optionVals[6] = (void *)(size_t)jitRegCount;

    // Create a pending linker invocation
    checkCudaErrors(cuLinkCreate(nOptions,options, optionVals, lState));

#if 0
    if (sizeof(void *)==4)
    {
        // Load the PTX from the string myPtx32
        printf("Loading myPtx32[] program\n");
        // PTX May also be loaded from file, as per below.
        myErr = cuLinkAddData(*lState, CU_JIT_INPUT_PTX, (void*)myPtx32, strlen(myPtx32)+1, 0, 0, 0, 0);
    }
    else
#endif
    {
        // Load the PTX from the string myPtx (64-bit)
        fprintf(stderr, "Loading ptx..\n");
        myErr = cuLinkAddData(*lState, CU_JIT_INPUT_PTX, (void*)module, strlen(module)+1, 0, 0, 0, 0);
        myErr = cuLinkAddFile(*lState, CU_JIT_INPUT_LIBRARY, "libcudadevrt.a", 0,0,0); 
        // PTX May also be loaded from file, as per below.
        // myErr = cuLinkAddFile(*lState, CU_JIT_INPUT_PTX, "myPtx64.ptx",0,0,0);
    }

    // Complete the linker step
    myErr = cuLinkComplete(*lState, &cuOut, &outSize);

    if ( myErr != CUDA_SUCCESS )
    {
      // Errors will be put in error_log, per CU_JIT_ERROR_LOG_BUFFER option above. 
      fprintf(stderr,"PTX Linker Error:\n%s\n",error_log);
      assert(0);
    }    

    // Linker walltime and info_log were requested in options above.
    fprintf(stderr, "CUDA Link Completed in %fms [ %g ms]. Linker Output:\n%s\n",walltime,info_log,1e3*(rtc() - t0));

    // Load resulting cuBin into module
    checkCudaErrors(cuModuleLoadData(&cudaModule, cuOut));

    // Destroy the linker invocation
    checkCudaErrors(cuLinkDestroy(*lState));
#endif
  fprintf(stderr, " loadModule took %g ms \n", 1e3*(rtc() - t0));
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
#define deviceLaunch(func,params) \
  checkCudaErrors(cuFuncSetCacheConfig((func), CU_FUNC_CACHE_PREFER_EQUAL)); \
  checkCudaErrors( \
      cuLaunchKernel( \
        (func), \
        1,1,1, \
        32, 1, 1, \
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
  double CUDALaunch(
      void **handlePtr, 
      const char * func_name,
      void **func_args)
  {
    const std::vector<char> module_str = readBinary("__kernels.ptx");
    const char *  module = &module_str[0];
    CUmodule   cudaModule   = loadModule(module);
    CUfunction cudaFunction = getFunction(cudaModule, func_name);
    const double t0 = rtc();
    deviceLaunch(cudaFunction, func_args);
    checkCudaErrors(cuStreamSynchronize(0));
    const double dt = rtc() - t0;
    unloadModule(cudaModule);
    return dt;
  }

}

/********************/


extern void mandelbrot_serial(float x0, float y0, float x1, float y1,
    int width, int height, int maxIterations,
    int output[]);

/* Write a PPM image file with the image of the Mandelbrot set */
static void
writePPM(int *buf, int width, int height, const char *fn) {
  FILE *fp = fopen(fn, "wb");
  fprintf(fp, "P6\n");
  fprintf(fp, "%d %d\n", width, height);
  fprintf(fp, "255\n");
  for (int i = 0; i < width*height; ++i) {
    // Map the iteration count to colors by just alternating between
    // two greys.
    char c = (buf[i] & 0x1) ? 240 : 20;
    for (int j = 0; j < 3; ++j)
      fputc(c, fp);
  }
  fclose(fp);
  printf("Wrote image file %s\n", fn);
}


static void usage() {
  fprintf(stderr, "usage: mandelbrot [--scale=<factor>]\n");
  exit(1);
}

int main(int argc, char *argv[]) {
  unsigned int width = 1536;
  unsigned int height = 1024;
  float x0 = -2;
  float x1 = 1;
  float y0 = -1;
  float y1 = 1;

  if (argc == 1)
    ;
  else if (argc == 2) {
    if (strncmp(argv[1], "--scale=", 8) == 0) {
      float scale = atof(argv[1] + 8);
      if (scale == 0.f)
        usage();
      width *= scale;
      height *= scale;
      // round up to multiples of 16
      width = (width + 0xf) & ~0xf;
      height = (height + 0xf) & ~0xf;
    }
    else 
      usage();
  }
  else
    usage();

  /*******************/
  createContext();
  /*******************/

  int maxIterations = 512;
  int *buf = new int[width*height];

  for (unsigned int i = 0; i < width*height; i++)
    buf[i] = 0;
  const size_t bufsize = sizeof(int)*width*height;
  devicePtr d_buf = deviceMalloc(bufsize);
  memcpyH2D(d_buf, buf, bufsize);

  //
  // Compute the image using the ispc implementation; report the minimum
  // time of three runs.
  //
  double minISPC = 1e30;
#if 1
  for (int i = 0; i < 3; ++i) {
    // Clear out the buffer
    for (unsigned int i = 0; i < width * height; ++i)
      buf[i] = 0;
    reset_and_start_timer();
#if 0
    const double t0 = rtc();
    mandelbrot_ispc(x0, y0, x1, y1, width, height, maxIterations, (int*)d_buf);
    double dt = rtc() - t0; //get_elapsed_mcycles();
#else
    const char * func_name = "mandelbrot_ispc";
    void *func_args[] = {&x0, &y0, &x1, &y1, &width, &height, &maxIterations, &d_buf};
    const double dt = CUDALaunch(NULL, func_name, func_args);
#endif
    minISPC = std::min(minISPC, dt);
  }
#endif

  memcpyD2H(buf, d_buf, bufsize);
  deviceFree(d_buf);

  printf("[mandelbrot ispc+tasks]:\t[%.3f] million cycles\n", minISPC);
  writePPM(buf, width, height, "mandelbrot-ispc.ppm");


  // 
  // And run the serial implementation 3 times, again reporting the
  // minimum time.
  //
  double minSerial = 1e30;
  for (int i = 0; i < 3; ++i) {
    // Clear out the buffer
    for (unsigned int i = 0; i < width * height; ++i)
      buf[i] = 0;
    reset_and_start_timer();
    const double t0 = rtc();
    mandelbrot_serial(x0, y0, x1, y1, width, height, maxIterations, buf);
    double dt = rtc() - t0; //get_elapsed_mcycles();
    minSerial = std::min(minSerial, dt);
  }

  printf("[mandelbrot serial]:\t\t[%.3f] million cycles\n", minSerial);
  writePPM(buf, width, height, "mandelbrot-serial.ppm");

  printf("\t\t\t\t(%.2fx speedup from ISPC + tasks)\n", minSerial/minISPC);

  return 0;
}
