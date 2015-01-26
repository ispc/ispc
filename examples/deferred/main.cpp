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
#define ISPC_IS_WINDOWS
#define NOMINMAX
#elif defined(__linux__)
#define ISPC_IS_LINUX
#elif defined(__APPLE__)
#define ISPC_IS_APPLE
#endif

#include <fcntl.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <stdint.h>
#include <algorithm>
#include <assert.h>
#include <vector>
#ifdef ISPC_IS_WINDOWS
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#endif
#include "deferred.h"
#include "kernels_ispc.h"
#include "../timing.h"

///////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("usage: deferred_shading <input_file (e.g. data/pp1280x720.bin)> [tasks iterations] [serial iterations]\n");
        return 1;
    }
    static unsigned int test_iterations[] = {5, 3, 500}; //last value is for nframes, it is scale.
    if (argc == 5) {
        for (int i = 0; i < 3; i++) {
            test_iterations[i] = atoi(argv[2 + i]);
        }
    }

    InputData *input = CreateInputDataFromFile(argv[1]);
    if (!input) {
        printf("Failed to load input file \"%s\"!\n", argv[1]);
        return 1;
    }

    Framebuffer framebuffer(input->header.framebufferWidth,
                            input->header.framebufferHeight);

    InitDynamicC(input);
#ifdef __cilk
    InitDynamicCilk(input);
#endif // __cilk

    int nframes = test_iterations[2];
    double ispcCycles = 1e30;
    for (unsigned int i = 0; i < test_iterations[0]; ++i) {
        framebuffer.clear();
        reset_and_start_timer();
        for (int j = 0; j < nframes; ++j)
            ispc::RenderStatic(input->header, input->arrays,
                               VISUALIZE_LIGHT_COUNT,
                               framebuffer.r, framebuffer.g, framebuffer.b);
        double mcycles = get_elapsed_mcycles() / nframes;
        printf("@time of ISPC + TASKS run:\t\t\t[%.3f] million cycles\n", mcycles);
        ispcCycles = std::min(ispcCycles, mcycles);
    }
    printf("[ispc static + tasks]:\t\t[%.3f] million cycles to render "
           "%d x %d image\n", ispcCycles,
           input->header.framebufferWidth, input->header.framebufferHeight);
    WriteFrame("deferred-ispc-static.ppm", input, framebuffer);

    nframes = 3;
#ifdef __cilk
    double dynamicCilkCycles = 1e30;
    for (int i = 0; i < test_iterations[1]; ++i) {
        framebuffer.clear();
        reset_and_start_timer();
        for (int j = 0; j < nframes; ++j)
            DispatchDynamicCilk(input, &framebuffer);
        double mcycles = get_elapsed_mcycles() / nframes;
        printf("@time of serial run:\t\t\t[%.3f] million cycles\n", mcycles);
        dynamicCilkCycles = std::min(dynamicCilkCycles, mcycles);
    }
    printf("[ispc + Cilk dynamic]:\t\t[%.3f] million cycles to render image\n", 
           dynamicCilkCycles);
    WriteFrame("deferred-ispc-dynamic.ppm", input, framebuffer);
#endif // __cilk

    double serialCycles = 1e30;
    for (unsigned int i = 0; i < test_iterations[1]; ++i) {
        framebuffer.clear();
        reset_and_start_timer();
        for (int j = 0; j < nframes; ++j)
            DispatchDynamicC(input, &framebuffer);
        double mcycles = get_elapsed_mcycles() / nframes;
        printf("@time of serial run:\t\t\t[%.3f] million cycles\n", mcycles);
        serialCycles = std::min(serialCycles, mcycles);
    }
    printf("[C++ serial dynamic, 1 core]:\t[%.3f] million cycles to render image\n", 
           serialCycles);
    WriteFrame("deferred-serial-dynamic.ppm", input, framebuffer);

#ifdef __cilk
    printf("\t\t\t\t(%.2fx speedup from static ISPC, %.2fx from Cilk+ISPC)\n", 
           serialCycles/ispcCycles, serialCycles/dynamicCilkCycles);
#else
    printf("\t\t\t\t(%.2fx speedup from ISPC + tasks)\n", serialCycles/ispcCycles);
#endif // __cilk

    DeleteInputData(input);

    return 0;
}
