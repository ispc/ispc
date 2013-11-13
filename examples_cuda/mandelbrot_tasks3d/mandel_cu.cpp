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
extern "C"
void mandelbrot_ispc(
    float x0,  float y0, 
    float x1,  float y1,
    int width,  int height, 
    int maxIterations,  int output[]) ;


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
#if 0
  struct ModuleManager
  {
    private:
      typedef std::pair<std::string, CUModule> ModulePair;
      typedef std::map <std::string, CUModule> ModuleMap;
      ModuleMap module_list;

      ModuleMap::iterator findModule(const char * module_name)
      {
        return module_list.find(std::string(module_name));
      }

    public:

      CUmodule loadModule(const char * module_name, const char * module_data)
      {
        const ModuleMap::iterator it = findModule(module_name)
          if (it != ModuleMap::end)
          {
            CUmodule cudaModule = loadModule(module);
            module_list.insert(std::make_pair(std::string(module_name), cudaModule));
            return cudaModule
          }
        return it->second;
      }
      void unloadModule(const char * module_name)
      {
        ModuleMap::iterator it = findModule(module_name)
          if (it != ModuleMap::end)
            module_list.erase(it);
      }
  };
#endif

  void *CUDAAlloc(void **handlePtr, int64_t size, int32_t alignment)
  {
#if 0
    fprintf(stderr, " ptr= %p\n", *handlePtr);
    fprintf(stderr, " size= %d\n", (int)size);
    fprintf(stderr, " alignment= %d\n", (int)alignment);
    fprintf(stderr, " ------- \n\n");
#endif
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
#if 0
    const char * module = module_1;
#else
    const std::vector<char> module_str = readBinary("__kernels.cubin");
    const char *  module = &module_str[0];
#endif
#if 1
    CUmodule   cudaModule   = loadModule(module);
    CUfunction cudaFunction = getFunction(cudaModule, func_name);
    deviceLaunch(cudaFunction, countx, county, countz, func_args);
    unloadModule(cudaModule);
#else
    fprintf(stderr, " handle= %p\n", *handlePtr);
    fprintf(stderr, " count= %d %d %d\n", countx, county, countz);

    fprintf(stderr, " module_name= %s \n", module_name);
    fprintf(stderr, " func_name= %s \n", func_name);
    //    fprintf(stderr, " ptx= %s \n", module);
    fprintf(stderr, " x0= %g  \n", *((float*)(func_args[0])));
    fprintf(stderr, " dx= %g  \n", *((float*)(func_args[1])));
    fprintf(stderr, " y0= %g  \n", *((float*)(func_args[2])));
    fprintf(stderr, " dy= %g  \n", *((float*)(func_args[3])));
    fprintf(stderr, " w= %d  \n", *((int*)(func_args[4])));
    fprintf(stderr, " h= %d  \n", *((int*)(func_args[5])));
    fprintf(stderr, " xs= %d  \n", *((int*)(func_args[6])));
    fprintf(stderr, " ys= %d  \n", *((int*)(func_args[7])));
    fprintf(stderr, " maxit= %d  \n", *((int*)(func_args[8])));
    fprintf(stderr, " ptr= %p  \n", *((int**)(func_args[9])));
    fprintf(stderr, " ------- \n\n");
#endif
  }
  void CUDASync(void *handle)
  {
    checkCudaErrors(cuStreamSynchronize(0));
  }
  void ISPCSync(void *handle)
  {
  }
  void CUDAFree(void *handle)
  {
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
    const double t0 = rtc();
    mandelbrot_ispc(x0, y0, x1, y1, width, height, maxIterations, (int*)d_buf);
    double dt = rtc() - t0; //get_elapsed_mcycles();
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
