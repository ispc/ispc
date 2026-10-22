// Check that a struct definition or a "struct B;" declaration in a function
// body, where B is a global typedef of a struct with another name, declares
// a new local struct B, as C does for the struct tag, and leaves the global
// struct unchanged.  In the scope of the typedef, "struct B;" refers to the
// struct of the typedef, and a definition defines that struct if it is
// declared in that scope, too, and is a redefinition otherwise.

// RUN: %{ispc} %s --target=host --nostdlib --nowrap -O2 --emit-llvm-text -o - | FileCheck %s
// RUN: not %{ispc} %s -DERR --target=host --nostdlib --nowrap -o %t.o 2>&1 | FileCheck %s --check-prefix=CHECK-ERR

struct A {
    int a;
};
typedef struct A B;

// CHECK-LABEL: define {{.*}} @local_definition___
// CHECK: ret i64 164
uniform int64 local_definition() {
    struct B {
        double b;
        int c;
    };
    uniform B x;
    x.b = 2.0;
    uniform A y;
    y.a = 1;
    return sizeof(uniform B) * 10 + sizeof(uniform A);
}

// CHECK-LABEL: define {{.*}} @local_declaration___
// CHECK: ret i64 14
uniform int64 local_declaration(uniform A *uniform p) {
    struct B;
    uniform B *uniform q = (uniform B * uniform)p;
    struct B {
        int8 c;
    };
    return sizeof(*q) * 10 + sizeof(uniform A);
}

// CHECK-LABEL: define {{.*}} @global_unchanged___
// CHECK: ret i64 4
uniform int64 global_unchanged() {
    struct B;
    uniform B *uniform p = NULL;
    return sizeof(uniform A);
}

// "struct B;" in the scope of the typedef refers to A.
struct B;
// CHECK-LABEL: define {{.*}} @same_scope___
// CHECK: ret i32 3
uniform int same_scope() {
    uniform B b;
    b.a = 3;
    return b.a;
}

// A definition through a typedef in its scope defines the struct of the
// typedef if that is declared in the same scope.
struct C;
typedef C D;
struct D {
    double d;
};
// CHECK-LABEL: define {{.*}} @typedef_definition___
// CHECK: ret i64 8
uniform int64 typedef_definition() { return sizeof(uniform C); }

// Also in the declaration of a function template, whose structs are global.
struct G;
typedef G GT;
template <typename T> noinline uniform int tf(uniform struct GT { int8 g; } *uniform p, T t) { return p->g; }
// CHECK-LABEL: define {{.*}} @template_definition___
// CHECK: ret i64 1
uniform int64 template_definition() { return sizeof(uniform G); }

#ifdef ERR
// CHECK-ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Ignoring redefinition of type "A".
struct B {
    double b;
};

// CHECK-ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Ignoring redefinition of type "C".
struct D {
    double d;
};

struct E;
typedef E F;
struct E {
    int e;
};
// CHECK-ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Ignoring redefinition of type "E".
struct F {
    double f;
};

// Also in the declaration of a function template, whose structs are global.
// CHECK-ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Ignoring redefinition of type "E".
template <typename T> void tf(uniform struct F { double f; } *uniform p, T t);

// A typedef in a function body of a struct declared outside of it.
uniform int g() {
    typedef A LA;
    // CHECK-ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Ignoring redefinition of type "LA".
    struct LA {
        double b;
    };
    return 0;
}

struct H;
uniform int h() {
    typedef H LH;
    // CHECK-ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Ignoring redefinition of type "LH".
    struct LH {
        double b;
    };
    return 0;
}
#endif
