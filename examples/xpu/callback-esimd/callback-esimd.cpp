/*
  Copyright (c) 2021-2023, Intel Corporation
*/

#include <iostream>
#include <CL/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

using namespace sycl::ext::intel::esimd;

typedef float (*CallbackFn)(void *, const float);

struct CallbackObject {
    CallbackFn function;
    float value;
};

extern "C" SYCL_EXTERNAL SYCL_ESIMD_FUNCTION float runCallbackEsimd(CallbackObject *object) {
    return object->function(object, -1.f);
}
