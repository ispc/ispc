/*
  Copyright (c)2022-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

struct Parameters {
    float *vec;
    int    count;
};

task void usm_mem(void *uniform _p) {
    Parameters *uniform p = (Parameters * uniform) _p;

    foreach (index = 0 ... p->count) {
        // Load the appropriate input value for this program instance.
        float v = p->vec[index];

        // Do an arbitrary little computation, but at least make the
        // computation dependent on the value being processed
        if (v < 3.)
            v = v * v;
        else
            v = sqrt(v);

        // And write the result to the output array.
        p->vec[index] = v;
    }
}

#include "ispcrt.isph"
DEFINE_CPU_ENTRY_POINT(usm_mem)
