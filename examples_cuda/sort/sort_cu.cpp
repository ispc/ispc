/*
  Copyright (c) 2013, Durham University
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Durham University nor the names of its
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

/* Author: Tomasz Koziara */

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include "../timing.h"
//#include "sort_ispc.h"
//using namespace ispc;

#include <sys/time.h>
static inline double rtc(void)
{
  struct timeval Tvalue;
  double etime;
  struct timezone dummy;

  gettimeofday(&Tvalue,&dummy);
  etime =  (double) Tvalue.tv_sec +
    1.e-6*((double) Tvalue.tv_usec);
  return etime;
}
/******************************/
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
  checkCudaErrors(cuCtxSetLimit(CU_LIMIT_MALLOC_HEAP_SIZE,1024*1024*1024));
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
  int jitRegCount = 32;
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
  checkCudaErrors(cuFuncSetCacheConfig((func), CU_FUNC_CACHE_PREFER_L1)); \
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
/******************************/



extern void sort_serial (int n, unsigned int code[], int order[]);

/* progress bar by Ross Hemsley;
 * http://www.rosshemsley.co.uk/2011/02/creating-a-progress-bar-in-c-or-any-other-console-app/ */
static inline void progressbar (unsigned int x, unsigned int n, unsigned int w = 50)
{
  if (n < 100)
  {
    x *= 100/n;
    n = 100;
  }

  if ((x != n) && (x % (n/100) != 0)) return;

  using namespace std;
  float ratio  =  x/(float)n;
  int c =  ratio * w;

  cout << setw(3) << (int)(ratio*100) << "% [";
  for (int x=0; x<c; x++) cout << "=";
  for (int x=c; x<w; x++) cout << " ";
  cout << "]\r" << flush;
}

int main (int argc, char *argv[])
{
  int i, j, n = argc == 1 ? 1000000 : atoi(argv[1]), m = n < 100 ? 1 : 50, l = n < 100 ? n : RAND_MAX;
  double tISPC1 = 0.0, tISPC2 = 0.0, tSerial = 0.0;
  printf("n= %d \n", n);
  unsigned int *code = new unsigned int [n];
  int *order = new int [n];

  srand (0);

#if 0
  for (i = 0; i < m; i ++)
  {
    for (j = 0; j < n; j ++) code [j] = random() % l;

    reset_and_start_timer();

    const double t0 = rtc();
    sort_ispc (n, code, order, 1);

    tISPC1 += (rtc() - t0); //get_elapsed_mcycles();

    if (argc != 3)
        progressbar (i, m);
  }

  printf("[sort ispc]:\t[%.3f] million cycles\n", tISPC1);
#endif

  srand (0);

  /*******************/
  createContext();
  /*******************/

  int ntask = 13*4*2;
  devicePtr d_code   = deviceMalloc(n*sizeof(int));
  devicePtr d_order  = deviceMalloc(n*sizeof(int));
  devicePtr d_pair   = deviceMalloc(n*2*sizeof(int));
  devicePtr d_temp   = deviceMalloc(n*2*sizeof(int));
  devicePtr d_hist   = deviceMalloc(256*32 * ntask * sizeof(int));
  devicePtr d_g      = deviceMalloc((ntask + 1) * sizeof(int));

  for (i = 0; i < m; i ++)
  {
    for (j = 0; j < n; j ++) code [j] = random() % l;
    memcpyH2D(d_code, code, n*sizeof(int));

#if 0
    reset_and_start_timer();

    const double t0 = rtc();
    sort_ispc (n, code, order, 0);

    tISPC2 += (rtc() - t0); // get_elapsed_mcycles();
#else
    const char * func_name = "sort_ispc";
#if 0
    void *func_args[] = {&n, &d_code, &d_order, &ntask};
#else
    void *func_args[] = {&n, &d_code, &d_order, &ntask, &d_hist, &d_pair, &d_temp, &d_g};
#endif
    const double dt = CUDALaunch(NULL, func_name, func_args);
    tISPC2 += dt;
#endif

    if (argc != 3)
        progressbar (i, m);
  }

  printf("[sort cuda]:\t[%.3f] million cycles\n", tISPC2);
  memcpyD2H(code,  d_code,  n*sizeof(int));
  memcpyD2H(order, d_order, n*sizeof(int));
  for (int i = 0; i < n-1; i++)
  {
    assert(code[i+1] >=  code[i]);
  }

  srand (0);

  for (i = 0; i < m; i ++)
  {
    for (j = 0; j < n; j ++) code [j] = random() % l;

    reset_and_start_timer();

    const double t0 = rtc();
    sort_serial (n, code, order);

    tSerial += (rtc() - t0);//get_elapsed_mcycles();

    if (argc != 3)
        progressbar (i, m);
  }

  printf("[sort serial]:\t\t[%.3f] million cycles\n", tSerial);

  printf("\t\t\t\t(%.2fx speedup from ISPC, %.2fx speedup from ISPC + tasks)\n", tSerial/tISPC1, tSerial/tISPC2);

  delete code;
  delete order;
  return 0;
}
