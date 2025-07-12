#include "test_static.isph"
#include "short_vec.isph"

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

    errors += test_select<int32, 1>();
    errors += test_select<int32, 2>();
    errors += test_select<int32, 3>();
    errors += test_select<int32, 4>();
    errors += test_select<int32, 5>();
    errors += test_select<int32, 6>();
    errors += test_select<int32, 7>();
    errors += test_select<int32, 8>();
    errors += test_select<int32, 9>();
    errors += test_select<int32, 11>();
    errors += test_select<int32, 15>();
    errors += test_select<int32, 16>();
    errors += test_select<int32, 17>();

    errors += test_select<uint32, 1>();
    errors += test_select<uint32, 3>();
    errors += test_select<uint32, 4>();
    errors += test_select<uint32, 7>();

    errors += test_select<int8, 3>();
    errors += test_select<uint8, 3>();
    errors += test_select<int16, 3>();
    errors += test_select<uint16, 3>();
    errors += test_select<int64, 3>();
    errors += test_select<uint64, 3>();

    errors += test_select<float, 1>();
    errors += test_select<float, 2>();
    errors += test_select<float, 3>();
    errors += test_select<float, 4>();
    errors += test_select<float, 5>();
    errors += test_select<float, 6>();
    errors += test_select<float, 7>();
    errors += test_select<float, 8>();
    errors += test_select<float, 9>();

    errors += test_select<double, 1>();
    errors += test_select<double, 2>();
    errors += test_select<double, 3>();
    errors += test_select<double, 4>();
    errors += test_select<double, 5>();
    errors += test_select<double, 6>();
    errors += test_select<double, 7>();
    errors += test_select<double, 8>();
    errors += test_select<double, 9>();

    RET[programIndex] = errors;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
