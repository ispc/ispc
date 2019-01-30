/*
  Copyright (c) 2010-2014, Intel Corporation
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

#if defined(_WIN32) || defined(_WIN64)
#define ISPC_IS_WINDOWS
#elif defined(__linux__)
#define ISPC_IS_LINUX
#elif defined(__APPLE__)
#define ISPC_IS_APPLE
#endif

#ifdef ISPC_IS_WINDOWS
#include <windows.h>
#endif // ISPC_IS_WINDOWS

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#ifdef ISPC_IS_LINUX
#include <malloc.h>
#endif

/******************************/

#include "drvapi_error_string.h"
#include "ispc_malloc.h"
#include <cassert>
#include <cuda.h>
#include <iostream>

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
// These are the inline versions for all of the SDK helper functions
void __checkCudaErrors(CUresult err, const char *file, const int line) {
    if (CUDA_SUCCESS != err) {
        std::cerr << "checkCudeErrors() Driver API error = " << err << "\"" << getCudaDrvErrorString(err)
                  << "\" from file <" << file << ", line " << line << "\n";
        exit(-1);
    }
}

/******************************/
/****  Basic CUDriver API  ****/
/******************************/

CUcontext context;

static void createContext(const int deviceId = 0, const bool verbose = true) {
    CUdevice device;
    int devCount;
    checkCudaErrors(cuInit(0));
    checkCudaErrors(cuDeviceGetCount(&devCount));
    assert(devCount > 0);
    checkCudaErrors(cuDeviceGet(&device, deviceId < devCount ? deviceId : 0));

    char name[128];
    checkCudaErrors(cuDeviceGetName(name, 128, device));
    if (verbose)
        std::cout << "Using CUDA Device [0]: " << name << "\n";

    int devMajor, devMinor;
    checkCudaErrors(cuDeviceComputeCapability(&devMajor, &devMinor, device));
    if (verbose)
        std::cout << "Device Compute Capability: " << devMajor << "." << devMinor << "\n";
    if (devMajor < 2) {
        if (verbose)
            std::cerr << "ERROR: Device 0 is not SM 2.0 or greater\n";
        exit(1);
    }

    // Create driver context
    checkCudaErrors(cuCtxCreate(&context, 0, device));
}
static void destroyContext() { checkCudaErrors(cuCtxDestroy(context)); }

static CUmodule loadModule(const char *module, const int maxrregcount = 64,
                           const char cudadevrt_lib[] = "libcudadevrt.a", const size_t log_size = 32768,
                           const bool print_log = true) {
    CUmodule cudaModule;
    // in this branch we use compilation with parameters

    CUlinkState CUState;
    CUlinkState *lState = &CUState;
    const int nOptions = 8;
    CUjit_option options[nOptions];
    void *optionVals[nOptions];
    float walltime;
    size_t logSize = log_size;
    char error_log[logSize], info_log[logSize];
    void *cuOut;
    size_t outSize;
    int myErr = 0;

    // Setup linker options
    // Return walltime from JIT compilation
    options[0] = CU_JIT_WALL_TIME;
    optionVals[0] = (void *)&walltime;
    // Pass a buffer for info messages
    options[1] = CU_JIT_INFO_LOG_BUFFER;
    optionVals[1] = (void *)info_log;
    // Pass the size of the info buffer
    options[2] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
    optionVals[2] = (void *)logSize;
    // Pass a buffer for error message
    options[3] = CU_JIT_ERROR_LOG_BUFFER;
    optionVals[3] = (void *)error_log;
    // Pass the size of the error buffer
    options[4] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
    optionVals[4] = (void *)logSize;
    // Make the linker verbose
    options[5] = CU_JIT_LOG_VERBOSE;
    optionVals[5] = (void *)1;
    // Max # of registers/pthread
    options[6] = CU_JIT_MAX_REGISTERS;
    int jitRegCount = maxrregcount;
    optionVals[6] = (void *)(size_t)jitRegCount;
    // Caching
    options[7] = CU_JIT_CACHE_MODE;
    optionVals[7] = (void *)CU_JIT_CACHE_OPTION_CA;
    // Create a pending linker invocation

    // Create a pending linker invocation
    checkCudaErrors(cuLinkCreate(nOptions, options, optionVals, lState));

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
        myErr = cuLinkAddData(*lState, CU_JIT_INPUT_PTX, (void *)module, strlen(module) + 1, 0, 0, 0, 0);
        myErr = cuLinkAddFile(*lState, CU_JIT_INPUT_LIBRARY, cudadevrt_lib, 0, 0, 0);
        // PTX May also be loaded from file, as per below.
        // myErr = cuLinkAddFile(*lState, CU_JIT_INPUT_PTX, "myPtx64.ptx",0,0,0);
    }

    // Complete the linker step
    myErr = cuLinkComplete(*lState, &cuOut, &outSize);

    if (myErr != CUDA_SUCCESS) {
        // Errors will be put in error_log, per CU_JIT_ERROR_LOG_BUFFER option above.
        fprintf(stderr, "PTX Linker Error:\n%s\n", error_log);
        assert(0);
    }

    // Linker walltime and info_log were requested in options above.
    if (print_log)
        fprintf(stderr, "CUDA Link Completed in %fms. Linker Output:\n%s\n", walltime, info_log);

    // Load resulting cuBin into module
    checkCudaErrors(cuModuleLoadData(&cudaModule, cuOut));

    // Destroy the linker invocation
    checkCudaErrors(cuLinkDestroy(*lState));
    return cudaModule;
}
static void unloadModule(CUmodule &cudaModule) { checkCudaErrors(cuModuleUnload(cudaModule)); }

static CUfunction getFunction(CUmodule &cudaModule, const char *function) {
    CUfunction cudaFunction;
    checkCudaErrors(cuModuleGetFunction(&cudaFunction, cudaModule, function));
    return cudaFunction;
}

static CUdeviceptr deviceMalloc(const size_t size) {
    CUdeviceptr d_buf;
    checkCudaErrors(cuMemAlloc(&d_buf, size));
    return d_buf;
}
static void deviceFree(CUdeviceptr d_buf) { checkCudaErrors(cuMemFree(d_buf)); }
static void memcpyD2H(void *h_buf, CUdeviceptr d_buf, const size_t size) {
    checkCudaErrors(cuMemcpyDtoH(h_buf, d_buf, size));
}
static void memcpyH2D(CUdeviceptr d_buf, void *h_buf, const size_t size) {
    checkCudaErrors(cuMemcpyHtoD(d_buf, h_buf, size));
}
#define deviceLaunch(func, params)                                                                                     \
    checkCudaErrors(cuFuncSetCacheConfig((func), CU_FUNC_CACHE_PREFER_L1));                                            \
    checkCudaErrors(cuLaunchKernel((func), 1, 1, 1, 32, 1, 1, 0, NULL, (params), NULL));

typedef CUdeviceptr devicePtr;

/**************/
#include <vector>
static std::vector<char> readBinary(const char *filename, const bool print_size = false) {
    std::vector<char> buffer;
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "file %s not found\n", filename);
        assert(0);
    }
    fseek(fp, 0, SEEK_END);
    const unsigned long long size = ftell(fp); /*calc the size needed*/
    fseek(fp, 0, SEEK_SET);
    buffer.resize(size);

    if (fp == NULL) { /*ERROR detection if file == empty*/
        fprintf(stderr, "Error: There was an Error reading the file %s \n", filename);
        exit(1);
    } else if (fread(&buffer[0], sizeof(char), size, fp) !=
               size) { /* if count of read bytes != calculated size of .bin file -> ERROR*/
        fprintf(stderr, "Error: There was an Error reading the file %s \n", filename);
        exit(1);
    }
    if (print_size)
        fprintf(stderr, " read buffer of size= %d bytes \n", (int)buffer.size());
    return buffer;
}

static double CUDALaunch(void **handlePtr, const char *func_name, void **func_args, const bool print_log = true,
                         const int maxrregcount = 64, const char kernel_file[] = "__kernels.ptx",
                         const char cudadevrt_lib[] = "libcudadevrt.a", const int log_size = 32768) {
    fprintf(stderr, " launching kernel: %s \n", func_name);
    const std::vector<char> module_str = readBinary(kernel_file, print_log);
    const char *module = &module_str[0];
    CUmodule cudaModule = loadModule(module, maxrregcount, cudadevrt_lib, log_size, print_log);
    CUfunction cudaFunction = getFunction(cudaModule, func_name);
    deviceLaunch(cudaFunction, func_args);
    checkCudaErrors(cuStreamSynchronize(0));
    unloadModule(cudaModule);
    return 0.0;
}
/******************************/

extern "C" {
//    extern int width();
int width() { return 32; }
extern void f_v(float *result);
extern void f_f(float *result, float *a);
extern void f_fu(float *result, float *a, float b);
extern void f_fi(float *result, float *a, int *b);
extern void f_du(float *result, double *a, double b);
extern void f_duf(float *result, double *a, float b);
extern void f_di(float *result, double *a, int *b);
extern void result(float *val);
}

#if defined(_WIN32) || defined(_WIN64)
#define ALIGN
#else
#define ALIGN __attribute__((aligned(64)))
#endif

int main(int argc, char *argv[]) {
    int w = width();
    assert(w <= 64);

    float returned_result[64] ALIGN;
    float vfloat[64] ALIGN;
    double vdouble[64] ALIGN;
    int vint[64] ALIGN;
    int vint2[64] ALIGN;

    const int device = 0;
#if 0
    const bool verbose = true;
#else
    const bool verbose = false;
#endif

    /*******************/
    createContext(device, verbose);
    /*******************/

    devicePtr d_returned_result = deviceMalloc(64 * sizeof(float));
    devicePtr d_vfloat = deviceMalloc(64 * sizeof(float));
    devicePtr d_vdouble = deviceMalloc(64 * sizeof(double));
    devicePtr d_vint = deviceMalloc(64 * sizeof(int));
    devicePtr d_vint2 = deviceMalloc(64 * sizeof(int));

    for (int i = 0; i < 64; ++i) {
        returned_result[i] = -1e20;
        vfloat[i] = i + 1;
        vdouble[i] = i + 1;
        vint[i] = 2 * (i + 1);
        vint2[i] = i + 5;
    }

    memcpyH2D(d_returned_result, returned_result, 64 * sizeof(float));
    memcpyH2D(d_vfloat, vfloat, 64 * sizeof(float));
    memcpyH2D(d_vdouble, vdouble, 64 * sizeof(double));
    memcpyH2D(d_vint, vint, 64 * sizeof(int));
    memcpyH2D(d_vint2, vint2, 64 * sizeof(int));

    float b = 5.;

    const bool print_log = false;
    const int nreg = 64;
#if (TEST_SIG == 0)
    void *args[] = {&d_returned_result};
    CUDALaunch(NULL, "f_v", args, print_log, nreg);
#elif (TEST_SIG == 1)
    void *args[] = {&d_returned_result, &d_vfloat};
    CUDALaunch(NULL, "f_f", args, print_log, nreg);
#elif (TEST_SIG == 2)
    void *args[] = {&d_returned_result, &d_vfloat, &b};
    CUDALaunch(NULL, "f_fu", args, print_log, nreg);
#elif (TEST_SIG == 3)
    void *args[] = {&d_returned_result, &d_vfloat, &vint};
    CUDALaunch(NULL, "f_fi", args, print_log, nreg);
#elif (TEST_SIG == 4)
    int num = 5;
    void *args[] = {&d_returned_result, &d_vdouble, &num};
    CUDALaunch(NULL, "f_du", args, print_log, nreg);
#elif (TEST_SIG == 5)
    float num = 5.0f;
    void *args[] = {&d_returned_result, &d_vdouble, &num};
    CUDALaunch(NULL, "f_duf", args, print_log, nreg);
#elif (TEST_SIG == 6)
    void *args[] = {&d_returned_result, &d_vdouble, &v_int2};
    CUDALaunch(NULL, "f_di", args, print_log, nreg);
#else
#error "Unknown or unset TEST_SIG value"
#endif

    float expected_result[64];

    memset(expected_result, 0, 64 * sizeof(float));
    devicePtr d_expected_result = deviceMalloc(64 * sizeof(float));
    memcpyH2D(d_expected_result, expected_result, 64 * sizeof(float));
    void *res_args[] = {&d_expected_result};
    CUDALaunch(NULL, "result", res_args, print_log, nreg);
    memcpyD2H(expected_result, d_expected_result, 64 * sizeof(float));
    memcpyD2H(returned_result, d_returned_result, 64 * sizeof(float));

    deviceFree(d_returned_result);
    deviceFree(d_vfloat);
    deviceFree(d_vdouble);
    deviceFree(d_vint);
    deviceFree(d_vint2);
    deviceFree(d_expected_result);

    /*******************/
    destroyContext();
    /*******************/

    int errors = 0;
    for (int i = 0; i < w; ++i) {
        if (returned_result[i] != expected_result[i]) {
#ifdef EXPECT_FAILURE
            // bingo, failed
            return 1;
#else
            printf("%s: value %d disagrees: returned %f [%a], expected %f [%a]\n", argv[0], i, returned_result[i],
                   returned_result[i], expected_result[i], expected_result[i]);
            ++errors;
#endif // EXPECT_FAILURE
        }
    }

#ifdef EXPECT_FAILURE
    // Don't expect to get here
    return 0;
#else
    return errors > 0;
#endif
}
