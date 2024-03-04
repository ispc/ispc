// Check that compiler reports a error when non-type template parameters are used incorrectly.
// RUN: not %{ispc}  %s --target=host -DVARYING --nostdlib -o %t.o --nowrap 2>&1 | FileCheck %s -check-prefix=CHECK_VARYING
// RUN: not %{ispc}  %s --target=host -DNEGATIVE --nostdlib -o %t.o --nowrap 2>&1 | FileCheck %s -check-prefix=CHECK_NEGATIVE
// RUN: not %{ispc}  %s --target=host -DNOT_INTEGRAL --nostdlib -o %t.o --nowrap 2>&1 | FileCheck %s -check-prefix=CHECK_NOT_INTEGRAL
// RUN: not %{ispc}  %s --target=host -DPARAM_MODIF --nostdlib -o %t.o --nowrap 2>&1 | FileCheck %s -check-prefix=CHECK_PARAM_MODIF
// RUN: not %{ispc}  %s --target=host -DPARAM_REDECL --nostdlib -o %t.o --nowrap 2>&1 | FileCheck %s -check-prefix=CHECK_PARAM_REDECL

// CHECK-NOT: Please file a bug report

// CHECK_VARYING: Error: syntax error, unexpected 'varying'
// CHECK_NEGATIVE: Error: syntax error, unexpected '-'.
// CHECK_NOT_INTEGRAL: Error: syntax error, unexpected 'float'
// CHECK_PARAM_MODIF: Error: Can't assign to type "const uniform int32" on left-hand side of expression.
// CHECK_PARAM_REDECL:  Error: Ignoring redeclaration of symbol "x".

#ifdef VARYING
template<varying int x>
noinline int bar(int k) {
    return x + k;
}

int foo(int k) {
    return bar<2>(k);
}
#endif

#ifdef NEGATIVE
template<int x>
noinline int bar(int k) {
    return x + k;
}

int foo(int k) {
    return bar<-2>(k);
}
#endif

#ifdef NOT_INTEGRAL
template<float x>
noinline int bar(int k) {
    return x + k;
}

int foo(int k) {
    return bar<2>(k);
}
#endif

#ifdef PARAM_MODIF
template<int x>
noinline int bar(int k) {
    x = 2;
    return x + k;
}

int foo(int k) {
    return bar<2>(k);
}
#endif

#ifdef PARAM_REDECL
template<int x>
noinline int bar(int k, uniform int x) {
    return x * k;
}

int foo(int k) {
    return bar<4>(k);
}
#endif