#pragma once

/******************************/

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


/******************************/
/****  Basic CUDriver API  ****/
/******************************/

CUcontext context;

static void createContext(
    const int deviceId = 0, 
    const size_t stackLimit = 4*1024,
    const size_t heapLimit = 1024*1024*1024
    )
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
#if 0
  size_t limit;
  checkCudaErrors(cuCtxGetLimit(&limit, CU_LIMIT_STACK_SIZE));
  fprintf(stderr, " stack_limit= %llu KB\n", limit/1024);
  checkCudaErrors(cuCtxGetLimit(&limit, CU_LIMIT_MALLOC_HEAP_SIZE));
  fprintf(stderr, " heap_limit= %llu KB\n", limit/1024);
  checkCudaErrors(cuCtxSetLimit(CU_LIMIT_STACK_SIZE,stackLimit));
  checkCudaErrors(cuCtxSetLimit(CU_LIMIT_MALLOC_HEAP_SIZE,heapLimit));
#endif
}
static void destroyContext()
{
  checkCudaErrors(cuCtxDestroy(context));
}

static CUmodule loadModule(
    const char * module,
    const int maxrregcount = 64,
    const char cudadevrt_lib[] = "libcudadevrt.a",
    const size_t log_size = 32768,
    const bool print_log = true
    )
{
  const double t0 = rtc();
  CUmodule cudaModule;
  // in this branch we use compilation with parameters

  CUlinkState  CUState;
  CUlinkState *lState = &CUState;
  const int nOptions = 8;
  CUjit_option options[nOptions];
  void* optionVals[nOptions];
  float walltime;
  size_t logSize = log_size;
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
  int jitRegCount = maxrregcount;
  optionVals[6] = (void *)(size_t)jitRegCount;
  // Caching
  options[7] = CU_JIT_CACHE_MODE;
  optionVals[7] = (void *)CU_JIT_CACHE_OPTION_CA;
  // Create a pending linker invocation

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
    if (print_log)
      fprintf(stderr, "Loading ptx..\n");
    myErr = cuLinkAddData(*lState, CU_JIT_INPUT_PTX, (void*)module, strlen(module)+1, 0, 0, 0, 0);
    myErr = cuLinkAddFile(*lState, CU_JIT_INPUT_LIBRARY, cudadevrt_lib, 0,0,0); 
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
 if (print_log)
   fprintf(stderr, "CUDA Link Completed in %fms [ %g ms]. Linker Output:\n%s\n",walltime,info_log,1e3*(rtc() - t0));

 // Load resulting cuBin into module
 checkCudaErrors(cuModuleLoadData(&cudaModule, cuOut));

 // Destroy the linker invocation
 checkCudaErrors(cuLinkDestroy(*lState));
 if (print_log)
   fprintf(stderr, " loadModule took %g ms \n", 1e3*(rtc() - t0));
 return cudaModule;
}
static void unloadModule(CUmodule &cudaModule)
{
  checkCudaErrors(cuModuleUnload(cudaModule));
}

static CUfunction getFunction(CUmodule &cudaModule, const char * function)
{
  CUfunction cudaFunction;
  checkCudaErrors(cuModuleGetFunction(&cudaFunction, cudaModule, function));
  return cudaFunction;
}

static CUdeviceptr deviceMalloc(const size_t size)
{
  CUdeviceptr d_buf;
  checkCudaErrors(cuMemAlloc(&d_buf, size));
  return d_buf;
}
static void deviceFree(CUdeviceptr d_buf)
{
  checkCudaErrors(cuMemFree(d_buf));
}
static void memcpyD2H(void * h_buf, CUdeviceptr d_buf, const size_t size)
{
  checkCudaErrors(cuMemcpyDtoH(h_buf, d_buf, size));
}
static void memcpyH2D(CUdeviceptr d_buf, void * h_buf, const size_t size)
{
  checkCudaErrors(cuMemcpyHtoD(d_buf, h_buf, size));
}
#define deviceLaunch(func,params) \
  checkCudaErrors(cuFuncSetCacheConfig((func), CU_FUNC_CACHE_PREFER_SHARED)); \
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
static std::vector<char> readBinary(const char * filename, const bool print_size = false)
{
  std::vector<char> buffer;
  FILE *fp = fopen(filename, "rb");
  if (!fp )
  {
    fprintf(stderr, "file %s not found\n", filename);
    assert(0);
  }
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
  if (print_size)
    fprintf(stderr, " read buffer of size= %d bytes \n", (int)buffer.size());
  return buffer;
}

static double CUDALaunch(
    void **handlePtr, 
    const char * func_name,
    void **func_args,
    const bool print_log = true,
    const int maxrregcount = 64,
    const char kernel_file[] = "__kernels.ptx",
    const char cudadevrt_lib[] = "libcudadevrt.a",
    const int log_size = 32768)
{
  const std::vector<char> module_str = readBinary(kernel_file, print_log);
  const char *  module = &module_str[0];
  CUmodule   cudaModule   = loadModule(module, maxrregcount, cudadevrt_lib, log_size, print_log);
  CUfunction cudaFunction = getFunction(cudaModule, func_name);
  checkCudaErrors(cuStreamSynchronize(0));
  const double t0 = rtc();
  deviceLaunch(cudaFunction, func_args);
  checkCudaErrors(cuStreamSynchronize(0));
  const double dt = rtc() - t0;
  unloadModule(cudaModule);
  return dt;
}
/******************************/

