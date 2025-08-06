#include "test_static.isph"

template <typename T>
int clamp_test() {
    int errorCount = 0;

    // The implementation must perform a clamp and is expected to be 
    // precise for values <= 0.
    if(abs(srgb8_to_float((T)-30)) != 0.0)
        errorCount++;

    // The actual implementation might not be very precise 
    // for values >=255. That being said a clamp is required.
    if(abs(srgb8_to_float((T)300) - 1.0) > 1.0 / 254.0)
        errorCount++;

    return errorCount;
}

task void f_v(uniform float RET[]) {
    int errorCount = 0;

    for(uniform int i = 0; i < 256; ++i)
        if(float_to_srgb8(srgb8_to_float(i)) != i)
            errorCount++;

    errorCount += clamp_test<uniform int>();

    RET[programIndex] = errorCount;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
}

