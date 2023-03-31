/*
  Copyright (c) 2013, Durham University
  Copyright (c) 2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/* Author: Tomasz Koziara */

#include "../../common/timing.h"
#include "sort_ispc.h"
#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

using namespace ispc;

extern void sort_serial(int n, unsigned int code[], int order[]);

static void progressBar(const int x, const int n, const int width = 50) {
    assert(n > 1);
    assert(x >= 0 && x < n);
    assert(width > 10);
    const float f = static_cast<float>(x) / (n - 1);
    const int w = static_cast<int>(f * width);

    // print bar
    std::string bstr("[");
    for (int i = 0; i < width; i++)
        bstr += i < w ? '=' : ' ';
    bstr += "]";

    // print percentage
    std::stringstream pstr0;
    pstr0 << " " << static_cast<int>(f * 100.0) << " % ";
    const std::string pstr(pstr0.str());
    std::copy(pstr.begin(), pstr.end(), bstr.begin() + (width / 2 - 2));

    std::cout << bstr;
    std::cout << (x == n - 1 ? "\n" : "\r") << std::flush;
}

int main(int argc, char *argv[]) {
    int i, j, n = argc == 1 ? 1000000 : atoi(argv[1]), m = n < 100 ? 1 : 50, l = n < 100 ? n : RAND_MAX;
    double tISPC1 = 0.0, tISPC2 = 0.0, tSerial = 0.0;
    unsigned int *code = new unsigned int[n];
    int *order = new int[n];

    srand(0);

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++)
            code[j] = rand() % l;

        reset_and_start_timer();

        sort_ispc(n, code, order, 1);

        tISPC1 += get_elapsed_mcycles();

        if (argc != 3)
            progressBar(i, m);
    }

    printf("[sort ispc]:\t[%.3f] million cycles\n", tISPC1);

    srand(0);

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++)
            code[j] = rand() % l;

        reset_and_start_timer();

        sort_ispc(n, code, order, 0);

        tISPC2 += get_elapsed_mcycles();

        if (argc != 3)
            progressBar(i, m);
    }

    printf("[sort ispc + tasks]:\t[%.3f] million cycles\n", tISPC2);

    srand(0);

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++)
            code[j] = rand() % l;

        reset_and_start_timer();

        sort_serial(n, code, order);

        tSerial += get_elapsed_mcycles();

        if (argc != 3)
            progressBar(i, m);
    }

    printf("[sort serial]:\t\t[%.3f] million cycles\n", tSerial);

    printf("\t\t\t\t(%.2fx speedup from ISPC, %.2fx speedup from ISPC + tasks)\n", tSerial / tISPC1, tSerial / tISPC2);

    delete[] code;
    delete[] order;
    return 0;
}
