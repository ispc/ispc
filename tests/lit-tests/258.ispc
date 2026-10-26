// Regression test for https://github.com/ispc/ispc/issues/258
// An anonymous struct defined in a function signature can't be named anywhere
// else, so it is an error, as in C++.  Named structs and typedefs of anonymous
// structs, or of pointers, references or arrays of them, are fine.

// RUN: not %{ispc} %s -DERRORS --target=host --nostdlib --nowrap -o %t.o 2>&1 | FileCheck %s --implicit-check-not=Error: --implicit-check-not=FATAL
// RUN: %{ispc} %s --target=host --nostdlib --nowrap -o %t.o 2>&1 | FileCheck %s --check-prefix=CHECK-OK --allow-empty

#ifdef ERRORS
// CHECK-DAG: 258.ispc:[[#@LINE+2]]:19: Error: Anonymous struct definition is illegal in function return type.
// CHECK-DAG: 258.ispc:[[#@LINE+1]]:53: Error: Anonymous struct definition is illegal in function parameter "s".
struct { int z; } foo(struct { int a, b; float c; } s);
// CHECK-DAG: 258.ispc:[[#@LINE+1]]:{{[0-9]+}}: Error: Anonymous struct definition is illegal in function parameter "p".
uniform int ptr(uniform struct { int x; } *uniform p);
// CHECK-DAG: 258.ispc:[[#@LINE+1]]:{{[0-9]+}}: Error: Anonymous struct definition is illegal in function parameter "a".
void arr(uniform struct { int x; } a[2]);
// CHECK-DAG: 258.ispc:[[#@LINE+1]]:{{[0-9]+}}: Error: Anonymous struct definition is illegal in function parameter "r".
void ref(uniform struct { int x; } &r);
// CHECK-DAG: 258.ispc:[[#@LINE+1]]:{{[0-9]+}}: Error: Anonymous struct definition is illegal in function parameter "p".
typedef void (*FP)(uniform struct { int x; } *uniform p);
// CHECK-DAG: 258.ispc:[[#@LINE+1]]:{{[0-9]+}}: Error: Anonymous struct definition is illegal in function return type.
uniform struct { int x; } *uniform ret();
// An unnamed parameter.
// CHECK-DAG: 258.ispc:[[#@LINE+1]]:{{[0-9]+}}: Error: Anonymous struct definition is illegal in function parameter.
void unnamed(uniform struct { int x; } *uniform);
// An abstract declarator, e.g. in sizeof.
// CHECK-DAG: 258.ispc:[[#@LINE+1]]:40: Error: Anonymous struct definition is illegal in function return type.
uniform int abstract() { return sizeof(uniform struct { int x; } (*uniform)()); }
#else
// CHECK-OK-NOT: Error
// CHECK-OK-NOT: FATAL
typedef struct {
    int x;
} S;
S good(S s, uniform S *uniform p);
typedef uniform struct {
    int x;
} *uniform P;
typedef uniform struct {
    int x;
} A2[2];
uniform int ptr(P p) { return p->x; }
uniform int arr(A2 a) { return a[1].x; }
P retPtr(P p) { return p; }
struct N {
    int y;
} named(struct N n);
uniform struct {
    int x;
} gv;
struct W {
    struct {
        int q;
    } inner;
};
uniform int useW(uniform W *uniform w) { return w->inner.q; }

// A typedef in a function body names the struct too.
uniform int local() {
    typedef uniform struct {
        int x;
    } LS;
    typedef uniform int (*FP)(LS s);
    typedef LS (*RP)();
    uniform LS ls = {1};
    // All declarators of a typedef name the same struct.
    typedef uniform struct {
        int y;
    } LA, LB, *uniform LP;
    typedef void (*FP2)(LA a, LB b, LP p);
    uniform LA la = {2};
    uniform LB lb = la;
    LP lp = &la;
    return ls.x + lb.y + lp->y;
}

uniform int abstract() { return sizeof(S(*uniform)()) + sizeof(uniform struct { int y; } * uniform); }

// Templates can still be instantiated over anonymous structs.
template <typename T> uniform int tf(T *uniform p) { return p->x; }
uniform int callTf() { return tf(&gv); }
#endif
