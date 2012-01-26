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

#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#ifdef ISPC_IS_LINUX
#include <malloc.h>
#endif

extern "C" {
    extern int width();
    extern void f_v(float *result);
    extern void f_f(float *result, float *a);
    extern void f_fu(float *result, float *a, float b);
    extern void f_fi(float *result, float *a, int *b);
    extern void f_du(float *result, double *a, double b);
    extern void f_duf(float *result, double *a, float b);
    extern void f_di(float *result, double *a, int *b);
    extern void result(float *val);
    
    void ISPCLaunch(void **handlePtr, void *f, void *d, int);
    void ISPCSync(void *handle);
    void *ISPCAlloc(void **handlePtr, int64_t size, int32_t alignment);
}

void ISPCLaunch(void **handle, void *f, void *d, int count) {
    *handle = (void *)0xdeadbeef;
    typedef void (*TaskFuncType)(void *, int, int, int, int);
    TaskFuncType func = (TaskFuncType)f;
    for (int i = 0; i < count; ++i)
        func(d, 0, 1, i, count);
}

void ISPCSync(void *) {
}


void *ISPCAlloc(void **handle, int64_t size, int32_t alignment) {
    *handle = (void *)0xdeadbeef;
    // and now, we leak...
#ifdef ISPC_IS_WINDOWS
    return _aligned_malloc(size, alignment);
#endif
#ifdef ISPC_IS_LINUX
    return memalign(alignment, size);
#endif
#ifdef ISPC_IS_APPLE
    void *mem = malloc(size + (alignment-1) + sizeof(void*));
    char *amem = ((char*)mem) + sizeof(void*);
    amem = amem + uint32_t(alignment - (reinterpret_cast<uint64_t>(amem) &
                                        (alignment - 1)));
    ((void**)amem)[-1] = mem;
    return amem;
#endif
}



int main(int argc, char *argv[]) {
    int w = width();
    assert(w <= 16);

    float returned_result[16];
    for (int i = 0; i < 16; ++i)
        returned_result[i] = -1e20;
    float vfloat[16] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    double vdouble[16] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    int vint[16] = { 2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32 };
    int vint2[16] = { 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    float b = 5.;

#if (TEST_SIG == 0)
    f_v(returned_result);
#elif (TEST_SIG == 1)
    f_f(returned_result, vfloat);
#elif (TEST_SIG == 2)
    f_fu(returned_result, vfloat, b);
#elif (TEST_SIG == 3)
    f_fi(returned_result, vfloat, vint);
#elif (TEST_SIG == 4)
    f_du(returned_result, vdouble, 5.);
#elif (TEST_SIG == 5)
    f_duf(returned_result, vdouble, 5.f);
#elif (TEST_SIG == 6)
    f_di(returned_result, vdouble, vint2);
#else
#error "Unknown or unset TEST_SIG value"
#endif    

    float expected_result[16];
    memset(expected_result, 0, 16*sizeof(float));
    result(expected_result);

    int errors = 0;
    for (int i = 0; i < w; ++i) {
        if (returned_result[i] != expected_result[i]) {
#ifdef EXPECT_FAILURE
            // bingo, failed
            return 1;
#else
            printf("%s: value %d disagrees: returned %f [%a], expected %f [%a]\n",
                   argv[0], i, returned_result[i], returned_result[i], 
                   expected_result[i], expected_result[i]);
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
