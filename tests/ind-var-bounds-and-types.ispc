#include "../test_static.isph"
task void f_v(uniform float RET[]) {
    uniform int mismatches = 0;

    for (unsigned int8 a = 0; a < 255; ++a) {
        const bool mismatch = rotate((int8)a, 3) != a;
        mismatches += (uniform int)any(mismatch);
    }
    for (int8 a = -128; a < 127; ++a) {
        const bool mismatch = rotate((int8)a, 3) != a;
        mismatches += (uniform int)any(mismatch);
    }

    for (unsigned int16 a = 0; a < 0xffff; ++a) {
        const bool mismatch = rotate((int16)a, 3) != a;
        mismatches += (uniform int)any(mismatch);
    }
    for (int16 a = -32768; a < 32767; ++a) {
        const bool mismatch = rotate((int16)a, 3) != a;
        mismatches += (uniform int)any(mismatch);
    }

    for (unsigned int32 a = 0xffff0000; a < 0xffffffff; ++a) {
        const bool mismatch = rotate((int32)a, 3) != a;
        mismatches += (uniform int)any(mismatch);
    }
    for (int32 a = -2147483648; a < -2147483648 + 0xffff; ++a) {
        const bool mismatch = rotate((int32)a, 3) != a;
        mismatches += (uniform int)any(mismatch);
    }
    for (int32 a = 2147483647 - 0xffff; a < 2147483647; ++a) {
        const bool mismatch = rotate((int32)a, 3) != a;
        mismatches += (uniform int)any(mismatch);
    }

    for (unsigned int64 a = 0xffffffffffff0000; a < 0xffffffffffffffff; ++a) {
        const bool mismatch = rotate((int64)a, 3) != a;
        mismatches += (uniform int)any(mismatch);
    }
    for (int64 a = 9223372036854775807 - 0xffff; a < 9223372036854775807; ++a) {
        const bool mismatch = rotate((int64)a, 3) != a;
        mismatches += (uniform int)any(mismatch);
    }
    for (int64 a = -9223372036854775808; a < -9223372036854775808 + 0xffff; ++a) {
        const bool mismatch = rotate((int64)a, 3) != a;
        mismatches += (uniform int)any(mismatch);
    }

    RET[programIndex] = mismatches;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
