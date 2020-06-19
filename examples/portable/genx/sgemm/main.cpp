/*
 * Copyright (c) 2019-2020, Intel Corporation
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

static void usage() {
    fprintf(stderr, "usage: sgemm [gemm block] [niterations] [group threads width] [group threads height]\n");
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
    if (m < 1 || niterations < 1 || gx < 1 || gy < 1) {
        usage();
        return -1;
    }

    std::cout << "Running test with " << niterations << " iterations on " << gx << " * " << gy << " threads."
              << std::endl;

    SGEMMApp app(true); // be verbose
    app.initialize();

    SGEMMApp::RunResult result;
    // validate only if number of iterations is set to 1
    app.run(result, m, niterations, gx, gy, niterations == 1);

    app.cleanup();

    return result.valid ? 0 : 1;
}
