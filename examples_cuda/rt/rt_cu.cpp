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
#include <math.h>
#include <algorithm>
#include <assert.h>
#include <string.h>
#include <sys/types.h>
#include "../timing.h"
#include "rt_ispc.h"

#include <sys/time.h>


/******************************/
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


using namespace ispc;

typedef unsigned int uint;


static void writeImage(int *idImage, float *depthImage, int width, int height,
    const char *filename) {
  FILE *f = fopen(filename, "wb");
  if (!f) {
    perror(filename);
    exit(1);
  }

  fprintf(f, "P6\n%d %d\n255\n", width, height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      // use the bits from the object id of the hit object to make a
      // random color
      int id = idImage[y * width + x];
      unsigned char r = 0, g = 0, b = 0;

      for (int i = 0; i < 8; ++i) {
        // extract bit 3*i for red, 3*i+1 for green, 3*i+2 for blue
        int rbit = (id & (1 << (3*i)))   >> (3*i);
        int gbit = (id & (1 << (3*i+1))) >> (3*i+1);
        int bbit = (id & (1 << (3*i+2))) >> (3*i+2);
        // and then set the bits of the colors starting from the
        // high bits...
        r |= rbit << (7-i);
        g |= gbit << (7-i);
        b |= bbit << (7-i);
      }
      fputc(r, f);
      fputc(g, f);
      fputc(b, f);
    }
  }            
  fclose(f);
  printf("Wrote image file %s\n", filename);
}


static void usage() {
  fprintf(stderr, "rt [--scale=<factor>] <scene name base>\n");
  exit(1);
}


int main(int argc, char *argv[]) {
  float scale = 1.f;
  const char *filename = NULL;
  for (int i = 1; i < argc; ++i) {
    if (strncmp(argv[i], "--scale=", 8) == 0) {
      scale = atof(argv[i] + 8);
      if (scale == 0.f)
        usage();
    }
    else if (filename != NULL)
      usage();
    else
      filename = argv[i];
  }
  if (filename == NULL)
    usage();

#define READ(var, n)                                            \
  if (fread(&(var), sizeof(var), n, f) != (unsigned int)n) {  \
    fprintf(stderr, "Unexpected EOF reading scene file\n"); \
    return 1;                                               \
  } else /* eat ; */                                                     

  //
  // Read the camera specification information from the camera file
  //
  char fnbuf[1024];
  sprintf(fnbuf, "%s.camera", filename);
  FILE *f = fopen(fnbuf, "rb");
  if (!f) {
    perror(fnbuf);
    return 1;
  }

  //
  // Nothing fancy, and trouble if we run on a big-endian system, just
  // fread in the bits
  //
  int baseWidth, baseHeight;
  float camera2world[4][4], raster2camera[4][4];
  READ(baseWidth, 1);
  READ(baseHeight, 1);
  READ(camera2world[0][0], 16);
  READ(raster2camera[0][0], 16);

  //
  // Read in the serialized BVH 
  //
  sprintf(fnbuf, "%s.bvh", filename);
  f = fopen(fnbuf, "rb");
  if (!f) {
    perror(fnbuf);
    return 1;
  }

  // The BVH file starts with an int that gives the total number of BVH
  // nodes
  uint nNodes;
  READ(nNodes, 1);

  LinearBVHNode *nodes = new LinearBVHNode[nNodes];
  for (unsigned int i = 0; i < nNodes; ++i) {
    // Each node is 6x floats for a boox, then an integer for an offset
    // to the second child node, then an integer that encodes the type
    // of node, the total number of int it if a leaf node, etc.
    float b[6];
    READ(b[0], 6);
    nodes[i].bounds[0][0] = b[0];
    nodes[i].bounds[0][1] = b[1];
    nodes[i].bounds[0][2] = b[2];
    nodes[i].bounds[1][0] = b[3];
    nodes[i].bounds[1][1] = b[4];
    nodes[i].bounds[1][2] = b[5];
    READ(nodes[i].offset, 1);
    READ(nodes[i].nPrimitives, 1);
    READ(nodes[i].splitAxis, 1);
    READ(nodes[i].pad, 1);
  }

  // And then read the triangles 
  uint nTris;
  READ(nTris, 1);
  Triangle *triangles = new Triangle[nTris];
  for (uint i = 0; i < nTris; ++i) {
    // 9x floats for the 3 vertices
    float v[9];
    READ(v[0], 9);
    float *vp = v;
    for (int j = 0; j < 3; ++j) {
      triangles[i].p[j][0] = *vp++;
      triangles[i].p[j][1] = *vp++;
      triangles[i].p[j][2] = *vp++;
    }
    // And create an object id
    triangles[i].id = i+1;
  }
  fclose(f);

  int height = int(baseHeight * scale);
  int width = int(baseWidth * scale);

  // allocate images; one to hold hit object ids, one to hold depth to
  // the first interseciton
  int *id = new int[width*height];
  float *image = new float[width*height];

  //
  // Run 3 iterations with ispc + 1 core, record the minimum time
  //
  double minTimeISPC = 1e30;
#if 0
  for (int i = 0; i < 3; ++i) {
    reset_and_start_timer();
    raytrace_ispc(width, height, baseWidth, baseHeight, raster2camera, 
        camera2world, image, id, nodes, triangles);
    double dt = get_elapsed_mcycles();
    minTimeISPC = std::min(dt, minTimeISPC);
  }
  printf("[rt ispc, 1 core]:\t\t[%.3f] million cycles for %d x %d image\n", 
      minTimeISPC, width, height);

  writeImage(id, image, width, height, "rt-ispc-1core.ppm");
#endif

  memset(id, 0, width*height*sizeof(int));
  memset(image, 0, width*height*sizeof(float));

  /*******************/
  createContext();
  /*******************/

  devicePtr d_raster2camera = deviceMalloc(4*4*sizeof(float));
  devicePtr d_camera2world  = deviceMalloc(4*4*sizeof(float));
  devicePtr d_nodes         = deviceMalloc(nNodes*sizeof(LinearBVHNode));
  devicePtr d_triangles     = deviceMalloc(nTris *sizeof(Triangle));
  devicePtr d_image         = deviceMalloc(width*height*sizeof(float));
  devicePtr d_id            = deviceMalloc(width*height*sizeof(int));

  memcpyH2D(d_raster2camera, raster2camera, 4*4*sizeof(float));
  memcpyH2D(d_camera2world,  camera2world,  4*4*sizeof(float));
  memcpyH2D(d_nodes,         nodes,         nNodes*sizeof(LinearBVHNode)); 
  memcpyH2D(d_triangles,     triangles,     nTris*sizeof(Triangle));
  memcpyH2D(d_image,         image,         width*height*sizeof(float));
  memcpyH2D(d_id,            id,            width*height*sizeof(int));


  //
  // Run 3 iterations with ispc + 1 core, record the minimum time
  //
  double minTimeISPCtasks = 1e30;
  for (int i = 0; i < 3; ++i) {
#if 0
    reset_and_start_timer();
    const double t0 = rtc();
    raytrace_ispc_tasks(
        width, 
        height, 
        baseWidth, 
        baseHeight, 
        (float(*)[4])d_raster2camera,
        (float(*)[4])d_camera2world, 
        (float*)d_image, 
        (int*)d_id, 
        (LinearBVHNode*)d_nodes, 
        (Triangle*)d_triangles);
    double dt = rtc() - t0; //get_elapsed_mcycles();
#else
    const char * func_name = "raytrace_ispc_tasks";
    void *func_args[] = {&width, &height, &baseWidth, &baseHeight,
      &d_raster2camera, &d_camera2world,
      &d_image, &d_id,
      &d_nodes, &d_triangles};
    const double dt = 1e3*CUDALaunch(NULL, func_name, func_args);
#endif
    minTimeISPCtasks = std::min(dt, minTimeISPCtasks);
  }
  printf("[rt ispc + tasks]:\t\t[%.3f] million cycles for %d x %d image\n", 
      minTimeISPCtasks, width, height);

  memcpyD2H(image, d_image, width*height*sizeof(float));
  memcpyD2H(id,    d_id,    width*height*sizeof(int));

  writeImage(id, image, width, height, "rt-cuda.ppm");

  /*******************/
  destroyContext();
  /*******************/


  memset(id, 0, width*height*sizeof(int));
  memset(image, 0, width*height*sizeof(float));

  //
  // And 3 iterations with the serial implementation, reporting the
  // minimum time.
  //
  double minTimeSerial = 1e30;
#if 0
  for (int i = 0; i < 3; ++i) {
    reset_and_start_timer();
    const double t0 = rtc();
    raytrace_serial(width, height, baseWidth, baseHeight, raster2camera, 
        camera2world, image, id, nodes, triangles);
    double dt = rtc() - t0; //get_elapsed_mcycles();
    minTimeSerial = std::min(dt, minTimeSerial);
  }
  printf("[rt serial]:\t\t\t[%.3f] million cycles for %d x %d image\n", 
      minTimeSerial, width, height);
  printf("\t\t\t\t(%.2fx speedup from ISPC, %.2fx speedup from ISPC + tasks)\n", 
      minTimeSerial / minTimeISPC, minTimeSerial / minTimeISPCtasks);

  writeImage(id, image, width, height, "rt-serial.ppm");
#endif

  return 0;
}
