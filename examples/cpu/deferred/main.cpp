/*
  Copyright (c) 2011-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#ifdef _MSC_VER
#define ISPC_IS_WINDOWS
#define NOMINMAX
#elif defined(__linux__)
#define ISPC_IS_LINUX
#elif defined(__APPLE__)
#define ISPC_IS_APPLE
#endif

#include <algorithm>
#include <assert.h>
#include <fcntl.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <vector>
#ifdef ISPC_IS_WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include "../../common/timing.h"
#include "deferred.h"
#include "kernels_ispc.h"

///////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
    if (argc < 2) {
        printf(
            "usage: deferred_shading <input_file (e.g. data/pp1280x720.bin)> [tasks iterations] [serial iterations]\n");
        return 1;
    }
    static unsigned int test_iterations[] = {5, 3, 500}; // last value is for nframes, it is scale.
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

    Framebuffer framebuffer(input->header.framebufferWidth, input->header.framebufferHeight);

    InitDynamicC(input);

    int nframes = test_iterations[2];
    double ispcCycles = 1e30;
    for (unsigned int i = 0; i < test_iterations[0]; ++i) {
        framebuffer.clear();
        reset_and_start_timer();
        for (int j = 0; j < nframes; ++j)
            ispc::RenderStatic(input->header, input->arrays, VISUALIZE_LIGHT_COUNT, framebuffer.r, framebuffer.g,
                               framebuffer.b);
        double mcycles = get_elapsed_mcycles() / nframes;
        printf("@time of ISPC + TASKS run:\t\t\t[%.3f] million cycles\n", mcycles);
        ispcCycles = std::min(ispcCycles, mcycles);
    }
    printf("[ispc static + tasks]:\t\t[%.3f] million cycles to render "
           "%d x %d image\n",
           ispcCycles, input->header.framebufferWidth, input->header.framebufferHeight);
    WriteFrame("deferred-ispc-static.ppm", input, framebuffer);

    nframes = 3;

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
    printf("[C++ serial dynamic, 1 core]:\t[%.3f] million cycles to render image\n", serialCycles);
    WriteFrame("deferred-serial-dynamic.ppm", input, framebuffer);

    printf("\t\t\t\t(%.2fx speedup from ISPC + tasks)\n", serialCycles / ispcCycles);

    DeleteInputData(input);

    return 0;
}
