// Check that structs declared in the declaration of a function template are
// declared in the global scope, as for a regular function.  A struct first
// named there is the same struct as its later global definition, so calls
// with pointers to it match the template, while a struct with the same name
// defined in a function body is a different struct and doesn't match.  A
// struct defined there can be used after the template, unless its members
// depend on the template parameters, and so can an enum defined in it.

// RUN: %{ispc} %s --emit-llvm-text --target=host --nostdlib --nowrap -O0 -o - | FileCheck %s
// RUN: not %{ispc} %s -DLOCAL --target=host --nostdlib --nowrap -o %t.o 2>&1 | FileCheck %s --check-prefix=CHECK-LOCAL
// RUN: not %{ispc} %s -DENUM_REDEF --target=host --nostdlib --nowrap -o %t.o 2>&1 | FileCheck %s --check-prefix=CHECK-ENUM-REDEF
// RUN: not %{ispc} %s -DDEPENDENT --target=host --nostdlib --nowrap -o %t.o -h %t.h 2>&1 | FileCheck %s --check-prefix=CHECK-DEPENDENT

template <typename T> noinline uniform struct Foo *uniform ret(uniform struct Foo *uniform p, T t);
template <typename T> noinline uniform int def(uniform struct Bar {
    uniform struct Baz *uniform p;
    uniform struct Baz {
        uniform int x;
    } b;
} *uniform p, T t) {
    return p->p->x + p->b.x + t;
}
template <typename T> noinline uniform int en(uniform struct WithEnum {
    enum InStruct { IS0 = 5 } e;
} *uniform p, T t) {
    return IS0 + t;
}
template <typename T> noinline uniform int arg(uniform struct Foo *uniform p, T t) { return t; }
void fn(uniform struct Foo *uniform p);

struct Foo {
    uniform int x;
};

template <typename T> noinline uniform struct Foo *uniform ret(uniform struct Foo *uniform p, T t) { return p; }
void fn(uniform Foo *uniform p) { p->x = 1; }

// CHECK-DAG: define linkonce_odr ptr @ret___uni___un_3C_s_5B_unFoo_5D__3E_uni(
// CHECK-DAG: define linkonce_odr i32 @arg___uni___un_3C_s_5B_unFoo_5D__3E_uni(
// CHECK-DAG: define linkonce_odr i32 @def___uni___un_3C_s_5B_unBar_5D__3E_uni(
// CHECK-DAG: define linkonce_odr i32 @en___uni___un_3C_s_5B_unWithEnum_5D__3E_uni(
// CHECK-DAG: define i32 @testEnum___unenum_5B_InStruct_5D_(
uniform int testEnum(uniform InStruct e) {
    uniform WithEnum w;
    return en(&w, 1) + e + IS0;
}
// CHECK-DAG: define i32 @test___un_3C_s_5B_unFoo_5D__3E_un_3C_s_5B_unBar_5D__3E_(
uniform int test(uniform Foo *uniform p, uniform Bar *uniform q) {
    fn(p);
    uniform Baz z = {4};
    return ret(p, 3)->x + arg(p, 2) + def(q, 1) + z.x;
}

#ifdef LOCAL
// CHECK-LOCAL: Error: Ignoring redefinition of type "Baz"
struct Baz {
    uniform int y;
};

// CHECK-LOCAL: Error: Unable to find any matching overload for call to function "arg"
uniform int testLocal() {
    struct Foo {
        uniform int x;
    };
    uniform Foo f;
    return arg(&f, 1);
}
#endif

#ifdef DEPENDENT
// CHECK-DEPENDENT: Error: Member "x" of struct "D1" defined in the declaration of a function template can't depend on the template parameters.
template <typename T> void dep1(uniform struct D1 { uniform T *uniform x; } *uniform p, T t) {}
export void useD1(uniform D1 *uniform p) {}
// CHECK-DEPENDENT: Error: Member "x" of struct "D2" defined in the declaration of a function template can't depend on the template parameters.
template <int N> void dep2(uniform struct D2 { uniform int (*uniform x)[2][N]; } *uniform p) {}
// CHECK-DEPENDENT: Error: Member "i" of struct "D3" defined in the declaration of a function template can't depend on the template parameters.
template <typename T> void dep3(uniform struct D3 { struct { uniform T *uniform y; } i; } *uniform p, T t) {}
// CHECK-DEPENDENT: Error: Member operator "->" can't be applied to declared but not defined struct type "uniform struct D1".
export void incD1(uniform D1 *uniform p) { p->x++; }
// CHECK-DEPENDENT-NOT: FATAL
// CHECK-DEPENDENT-NOT: Assertion
#endif

#ifdef ENUM_REDEF
// An enum defined in the declaration of a function template is global, so
// another enum with the same name is an error, as for regular functions.
// CHECK-ENUM-REDEF: Error: syntax error, unexpected TOKEN_TYPE_NAME
// CHECK-ENUM-REDEF-NOT: FATAL
template <typename T> void redef1(uniform struct R1 { enum RE { R1A } e; } *uniform p, T t);
template <typename T> void redef2(uniform struct R2 { enum RE { R2A } e; } *uniform p, T t);
#endif
