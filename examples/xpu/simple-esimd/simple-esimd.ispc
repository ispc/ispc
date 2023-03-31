/*
  Copyright (c)2021-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

struct Parameters {
    float *vin;
    float *vout;
    int    count;
};

extern "C" void vmult(varying float * uniform a, varying float * uniform b, uniform int factor, uniform int n);

task void simple_ispc(void *uniform _p) {
    Parameters *uniform p = (Parameters * uniform) _p;

    foreach (index = 0 ... p->count) {
        // Load the appropriate input value for this program instance.
        float v = p->vin[index];
        float v_out = p->vout[index];

        // Do an arbitrary little computation, but at least make the
        // computation dependent on the value being processed
        if (v < 3.)
            v = v * v;
        else
            v = sqrt(v);

        vmult(&v, &v_out, 2, programCount);
        // And write the result to the output array.
        p->vout[index] = v_out;
    }
}


#include "ispcrt.isph"
DEFINE_CPU_ENTRY_POINT(simple_ispc)
