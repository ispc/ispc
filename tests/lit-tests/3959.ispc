// Structs with the same name in different scopes are different types with
// their own layouts.
// See https://github.com/ispc/ispc/issues/3959

// RUN: %{ispc} %s --target=avx2-i32x8 --nostdlib --nowrap -O2 --emit-llvm-text -o %t.ll
// RUN: FileCheck %s < %t.ll
// RUN: FileCheck %s --check-prefix=CHECK_TYPES < %t.ll

// REQUIRES: X86_ENABLED

// Local structs with the same name in different functions.
// CHECK-LABEL: define {{.*}} @local_first(
// CHECK: getelementptr i8, ptr %v, i64 4
export uniform int local_first(void *uniform v) {
    struct L { int x; };
    uniform L *uniform p = (uniform L *uniform)v;
    return (p + 1)->x;
}

// CHECK-LABEL: define {{.*}} @local_second(
// CHECK: getelementptr i8, ptr %v, i64 8
export uniform int64 local_second(void *uniform v) {
    struct L { int64 y; };
    uniform L *uniform p = (uniform L *uniform)v;
    return (p + 1)->y;
}

// CHECK-LABEL: define {{.*}} @varying_first(
// CHECK: ret i64 32
export uniform int64 varying_first() {
    struct V { int x; };
    return sizeof(varying V);
}

// CHECK-LABEL: define {{.*}} @varying_second(
// CHECK: ret i64 128
export uniform int64 varying_second() {
    struct V { int64 y; double z; };
    return sizeof(varying V);
}

// A local struct that shadows a global one.
struct G { int8 a; };

// CHECK-LABEL: define {{.*}} @global_before(
// CHECK: ret i64 1
export uniform int64 global_before() { return sizeof(uniform G); }

// CHECK-LABEL: define {{.*}} @global_shadow(
// CHECK: ret i64 16
export uniform int64 global_shadow() {
    struct G { int64 a; int64 b; };
    return sizeof(uniform G);
}

// CHECK-LABEL: define {{.*}} @global_after(
// CHECK: ret i64 1
export uniform int64 global_after() { return sizeof(uniform G); }

// Only "struct G;" declares a new struct G, not a declaration with qualifiers
// or with a typedef name of G.
// CHECK-LABEL: define {{.*}} @global_qualified(
// CHECK: ret i64 1
export uniform int64 global_qualified() {
    struct G const;
    uniform struct G;
    typedef G GT;
    struct GT;
    return sizeof(uniform G);
}

// The members of a local struct that shadows a global one refer to the local
// struct.
struct Node { uniform Node *uniform next; };

// CHECK-LABEL: define {{.*}} @self_reference(
// CHECK: getelementptr i8, ptr %v, i64 16
export uniform int64 self_reference(void *uniform v) {
    struct Node { int64 value; uniform Node *uniform next; };
    uniform Node n;
    n.next = &n;
    uniform Node *uniform p = (uniform Node *uniform)v;
    return n.next == &n ? (p + 1)->value : 0;
}

// A local struct in a nested block that shadows a local struct.
// CHECK-LABEL: define {{.*}} @nested_block(
// CHECK: ret i64 20
export uniform int64 nested_block() {
    struct L { int a; };
    uniform int64 r = 0;
    {
        struct L { uniform L *uniform o; int64 b; };
        uniform L l;
        l.o = &l;
        r = l.o == &l ? sizeof(uniform L) : 0;
    }
    return r + sizeof(uniform L);
}

// "struct G;" in a function declares a new struct G that shadows the global
// one, and its definition in the function is the same struct.
// CHECK-LABEL: define {{.*}} @forward_shadow(
// CHECK: getelementptr i8, ptr %v, i64 16
export uniform int64 forward_shadow(void *uniform v) {
    struct G;
    uniform G *uniform p;
    struct G { int64 a; int64 b; };
    uniform G g;
    p = &g;
    uniform G *uniform q = (uniform G *uniform)v;
    return p == &g ? (q + 1)->a : 0;
}

// A local struct that shadows a global typedef of an anonymous struct.
typedef struct { int8 a; } T;

// CHECK-LABEL: define {{.*}} @typedef_global(
// CHECK: ret i64 1
export uniform int64 typedef_global() { return sizeof(uniform T); }

// CHECK-LABEL: define {{.*}} @typedef_shadow(
// CHECK: ret i64 24
export uniform int64 typedef_shadow() {
    struct T { int64 x; int64 y; int64 z; };
    return sizeof(uniform T);
}

// A local struct that is forward declared and defined in the same function.
// CHECK-LABEL: define {{.*}} @forward_local(
// CHECK: getelementptr i8, ptr %v, i64 6
export uniform int16 forward_local(void *uniform v) {
    struct L;
    uniform L *uniform p = (uniform L *uniform)v;
    struct L { int16 a; int16 b; int16 c; };
    uniform L *uniform q = (uniform L *uniform)v;
    return p == NULL ? 0 : (q + 1)->a;
}

// Using a pointer to a forward declared struct must not affect the layout of
// another struct with the same name.
// CHECK-LABEL: define {{.*}} @forward_compare(
export uniform bool forward_compare(void *uniform v) {
    struct F;
    uniform F *uniform p = (uniform F *uniform)v;
    struct F { int x; };
    return p != NULL;
}

// CHECK-LABEL: define {{.*}} @forward_other(
// CHECK: getelementptr i8, ptr %v, i64 8
export uniform int64 forward_other(void *uniform v) {
    struct F { int64 y; };
    uniform F *uniform p = (uniform F *uniform)v;
    return (p + 1)->y;
}

// A global forward declaration and its definition are the same struct, even
// if a local struct with the same name comes between them.
// CHECK_TYPES-DAG: %v8_uniform_R = type { i32 }
// CHECK_TYPES-DAG: define %v8_uniform_R @makeR___
// CHECK_TYPES-DAG: define {{.*}} @takeR___{{.*}}(%v8_uniform_R %r,
struct R;
noinline uniform R makeR(uniform int x);
uniform int64 localR() {
    struct R { int64 a; };
    uniform R r;
    r.a = 0;
    return r.a;
}
struct R { int x; };
noinline uniform int takeR(uniform R r) { return r.x; }
export uniform int forward_global(uniform int x) { return takeR(makeR(x)); }
noinline uniform R makeR(uniform int x) {
    uniform R r;
    r.x = x;
    return r;
}

// A definition through a typedef with qualifiers of a struct forward declared
// in the same scope defines that struct.
// CHECK-LABEL: define {{.*}} @typedef_definition(
// CHECK: store ptr
struct Q;
uniform Q *uniform globalQ;
typedef uniform Q QT;
struct QT { int x; };
export void typedef_definition(uniform Q *uniform q) { globalQ = q; }

// Same-named structs in different functions are different types, so a
// template gets a separate instantiation for each of them.
// CHECK_TYPES-DAG: %v8_uniform_S = type { i32 }
// CHECK_TYPES-DAG: %v8_uniform_S.1 = type { float }
// CHECK_TYPES-DAG: define {{.*}} @read___s_5B_unS.1_5D_{{.*}}(%v8_uniform_S %s{{[,)]}}
// CHECK_TYPES-DAG: define {{.*}} @read___s_5B_unS.2_5D_{{.*}}(%v8_uniform_S.1 %s{{[,)]}}
template <typename U> noinline uniform int read(U s) { return s.x; }

export uniform int template_int(uniform int x) {
    struct S { int x; };
    uniform S s = {x};
    return read(s);
}

export uniform int template_float(uniform int x) {
    struct S { float x; };
    uniform S s = {x};
    return read(s);
}
