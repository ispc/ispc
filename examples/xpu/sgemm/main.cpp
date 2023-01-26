/*
 * Copyright (c) 2019-2023, Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include <iostream>

#include "Matrix.h"
#include "sgemm.hpp"

constexpr int MAX_ITERS = 1000000;
constexpr int MAX_M = 4096;
constexpr int MAX_TH = 4096;

static void usage() {
    fprintf(stderr, "usage: sgemm [gemm block] [niterations] [group threads width] [group threads height]\n");
    fprintf(stderr, "\tmaximum size of gemm block: %d\n", MAX_M);
    fprintf(stderr, "\tmaximum number of iterations: %d\n", MAX_ITERS);
    fprintf(stderr, "\tmaximum value of group threads width/height: %d\n", MAX_TH);
}

int main(int argc, char *argv[]) {
    int m = GEMM_BLOCK;
    int niterations = 1;
    int gx = 2, gy = 1;

    if (argc >= 3) {
        m = atoi(argv[1]);
        niterations = atoi(argv[2]);
        if (argc == 5) {
            gx = atoi(argv[3]);
            gy = atoi(argv[4]);
        }
    }
    if (m < 1 || niterations < 1 || gx < 1 || gy < 1 || m > MAX_M || niterations > MAX_ITERS || gx > MAX_TH ||
        gy > MAX_TH) {
        usage();
        return -1;
    }

    std::cout << "Running test with " << niterations << " iterations on " << gx << " * " << gy << " threads."
              << std::endl;

    SGEMMApp app(true); // be verbose
    app.initialize();

    SGEMMApp::RunResult result;
    // validate only if number of iterations is set to 1
    try {
        app.run(result, m, niterations, gx, gy, niterations == 1);
    }
    catch (const std::exception &exc) {
        std::cerr << exc.what();
        return -1;
    }

    app.cleanup();

    return result.valid ? 0 : 1;
}
