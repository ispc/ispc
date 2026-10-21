// Check that template instantiations over structs declared in a function body
// or over anonymous structs get internal linkage, so they are not merged with
// same-named instantiations from other translation units.
// See https://github.com/ispc/ispc/issues/3962

// RUN: %{ispc} %s --emit-llvm-text --target=host --nostdlib -O0 -o - | FileCheck %s

struct G {
    int x;
};
typedef struct {
    int x;
} GT;

template <typename U> noinline uniform int rd(U *uniform s) { return s[1].x; }
template <typename U> noinline uniform int rdRef(uniform U &s) { return s.x; }
template <typename U> noinline uniform int rdVal(uniform U s) { return s.x; }
template <typename U> noinline uniform int outer(U *uniform s) { return rd(s); }
template <typename U> noinline uniform bool isNull(U s) { return s == NULL; }
template <typename U> noinline uniform int first(uniform U a) { return a[0].x; }
template <typename U> noinline uniform int firstRef(uniform U &a) { return a[0].x; }

// Global struct: instantiations keep linkonce_odr linkage.
// CHECK-DAG: define linkonce_odr i32 @rd___s_5B_unG_5D___
export uniform int fglobal(uniform G *uniform v) { return rd(v); }

// Typedef of a global anonymous struct: the struct is named by the typedef.
// CHECK-DAG: define linkonce_odr i32 @rd___s_5B_unGT_5D___
export uniform int fglobalTypedef(uniform GT *uniform v) { return rd(v); }

// Local typedef of a global struct.
// CHECK-DAG: define linkonce_odr i32 @first___vy_3C_s_5B_unG_5D__3E___
export uniform int fglobalLocalTypedef(void *uniform v) {
    typedef G LG;
    return first((uniform LG * uniform) v);
}

// Local struct used through a pointer.
// CHECK-DAG: define internal i32 @rd___s_5B_unL1.1_5D___
export uniform int flocal1(void *uniform v) {
    struct L1 {
        int x;
    };
    return rd((uniform L1 * uniform) v);
}

// Local struct used through a reference.
// CHECK-DAG: define internal i32 @rdRef___s_5B_vyL2.1_5D___
export uniform int flocal2(void *uniform v) {
    struct L2 {
        int x;
    };
    return rdRef(*((uniform L2 * uniform) v));
}

// Local struct passed by value.
// CHECK-DAG: define internal i32 @rdVal___s_5B_vyL3.1_5D___
export uniform int flocal3(void *uniform v) {
    struct L3 {
        int x;
    };
    return rdVal(*((uniform L3 * uniform) v));
}

// Local struct in a nested scope.
// CHECK-DAG: define internal i32 @rd___s_5B_unL4.1_5D___
export uniform int flocal4(void *uniform v) {
    if (v != NULL) {
        struct L4 {
            int x;
        };
        return rd((uniform L4 * uniform) v);
    }
    return 0;
}

// Typedef of a local anonymous struct.
// CHECK-DAG: define internal i32 @"rd___s_5B_un$anon{{[0-9]+}}_5D___
export uniform int flocal5(void *uniform v) {
    typedef struct {
        int x;
    } T5;
    return rd((uniform T5 * uniform) v);
}

// Local struct passed through another template instantiation.
// CHECK-DAG: define internal i32 @outer___s_5B_unL6.1_5D___
// CHECK-DAG: define internal i32 @rd___s_5B_unL6.1_5D___
export uniform int flocal6(void *uniform v) {
    struct L6 {
        int x;
    };
    return outer((uniform L6 * uniform) v);
}

// Local struct that shadows the global one.
// CHECK-DAG: define internal i32 @rd___s_5B_unG.1_5D___
export uniform int flocal7(void *uniform v) {
    struct G {
        int64 x;
    };
    return rd((uniform G * uniform) v);
}

// Local struct that is only forward declared.
// CHECK-DAG: define internal i1 @isNull___un_3C_s_5B_unF8.1_5D__3E___
export uniform bool flocal8(void *uniform v) {
    struct F8;
    return isNull((uniform F8 * uniform) v);
}

// Local struct used through a function pointer type.
// CHECK-DAG: define internal i1 @isNull___un_3C____un_3C_s_5B_unF9.1_5D__3E__3E___
export uniform bool flocal9() {
    struct F9 {
        int x;
    };
    uniform int (*uniform fp)(uniform F9 * uniform) = NULL;
    return isNull(fp);
}

// Array of local structs, passed by reference so that it doesn't decay to a
// pointer.
// CHECK-DAG: define internal i32 @firstRef___s_5B_vyL10.1_5D__5B_2_5D____REFs_5B_unL10.1_5D__5B_2_5D_(
export uniform int flocal10() {
    struct L10 {
        int x;
    };
    uniform L10 a[2];
    a[0].x = 1;
    return firstRef(a);
}

// Anonymous struct at global scope.
uniform struct {
    int x;
} anonVar;
// CHECK-DAG: define internal i32 @"rdRef___s_5B_vy$anon{{[0-9]+}}_5D___
export uniform int fanon() { return rdRef(anonVar); }
