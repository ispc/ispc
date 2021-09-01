//==---------------- callback-esimd.cpp - DPC++ ESIMD calling ISPC function pointer test ---------==//

#include <CL/sycl.hpp>
#include <iostream>
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

