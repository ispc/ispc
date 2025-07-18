#include "test_static.isph"
#include "short_vec.isph"
// rule: skip on arch=*

template <typename T, int N>
int test_select() {
    int errors = 0;

    uniform T<N> t;
    uniform T<N> f;
    uniform bool<N> cond;

    foreach(i = 0...N) {
        t[i] = 42 + i;
        f[i] = 815 + i;
        cond[i] = i % 2 == 0 || i % 5 == 3;
    }

    uniform T<N> rt = select<T, N>(true, t, f);
    uniform T<N> rf = select<T, N>(false, t, f);
    uniform T<N> rc1 = select<T, N>(cond, t, f);
    uniform T<N> rc2 = select<T, N>(cond, f, t);

    foreach(i = 0...N) {
        errors += rt[i] != t[i] ? 1 : 0;
        errors += rf[i] != f[i] ? 1 : 0;
        errors += (cond[i] == true && rc1[i] != t[i] || cond[i] == false && rc1[i] != f[i]) ? 1 : 0;
        errors += (cond[i] == true && rc2[i] != f[i] || cond[i] == false && rc2[i] != t[i]) ? 1 : 0;
    }

    return errors;
}

task void f_v(uniform float RET[]) {
    int errors = 0;
    // This is extraction from select-short-vec.ispc for easier reproduction of the issue
    errors += test_select<int8, 3>();
    errors += test_select<uint8, 3>();

    RET[programIndex] = errors;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
