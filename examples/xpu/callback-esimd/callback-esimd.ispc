/*
  Copyright (c)2021-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

typedef uniform float (*uniform CallbackFn)(void *uniform, const uniform float x);

struct CallbackObject {
    CallbackFn function;
    float value;
};

struct Parameters {
    CallbackObject *uniform objects;
    float *output;
};

extern "C" uniform float runCallbackEsimd(CallbackObject *uniform object);

uniform float callbackFn(void *uniform _o, const uniform float x) {
    CallbackObject *uniform obj = (CallbackObject * uniform) _o;
    return obj->value * x;
}

task void make_objects(void *uniform _p) {
    Parameters *uniform p = (Parameters * uniform) _p;

    p->objects[0].function = callbackFn;
    p->objects[0].value = 1.f;

    p->objects[1].function = callbackFn;
    p->objects[1].value = 2.f;
}

task void call_objects(void *uniform _p) {
    Parameters *uniform p = (Parameters * uniform) _p;
    for (uniform int i = 0; i < 2; ++i) {
        p->output[i] = runCallbackEsimd(&p->objects[i]);
    }
}

#include "ispcrt.isph"
DEFINE_CPU_ENTRY_POINT(make_objects)
DEFINE_CPU_ENTRY_POINT(call_objects)

