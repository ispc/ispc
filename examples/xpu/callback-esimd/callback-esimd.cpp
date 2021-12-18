/*
  Copyright (c) 2021, Intel Corporation
*/

#include <iostream>
#include <sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>

using namespace sycl::ext::intel::experimental::esimd;

typedef float (*CallbackFn)(void *, const float);

struct CallbackObject {
    CallbackFn function;
    float value;
};

extern "C" SYCL_EXTERNAL SYCL_ESIMD_FUNCTION float runCallbackEsimd(CallbackObject *object) {
    return object->function(object, -1.f);
}
