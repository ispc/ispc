#include "../test_static.isph"

template <typename T> inline T addT(T A, T B) { return A + B; }
template <typename T> inline T mulT(T A, T B) { return A * B; }
template <typename T> inline T fmaT(T A, T B, T C) { return addT<T>(mulT<T>(A, B), C); }


inline float add(float A, float B) { return A + B; }
inline float mul(float A, float B) { return A * B; }
inline float fma(float A, float B, float C) { return add(mul(A, B), C); }


task void f_fi(uniform float RET[], uniform float a[], uniform int b[]) {
    RET[programIndex] = 0;
    varying float val1 = a[programIndex];
    varying float val2 = b[programIndex];
    varying float val3 = val1 - val2;

    float template_val = fmaT<varying float>(val1, val2, val3);
    float func_val = fma(val1, val2, val3);
    if (template_val == func_val) {
        RET[programIndex] = 1;
    }
}


task void result(uniform float RET[4]) {
    RET[programIndex] = 1;
}
