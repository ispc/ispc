// Check that an enum or a struct declared in a function body can reuse the
// name of an outer enum, struct or typedef: it is a new type that hides the
// outer one.  Neither can hide a template parameter, as in C++.

// RUN: %{ispc} %s --target=host --nostdlib --nowrap -O0 --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=host --nostdlib --nowrap -O2 --emit-llvm-text -o - | FileCheck %s --check-prefix=CHECK-SELF
// RUN: %{ispc} %s --target=host --nostdlib --nowrap -o %t.o -h %t.h
// RUN: FileCheck %s --check-prefix=CHECK-HEADER < %t.h
// RUN: %{ispc} %s --target=host --nostdlib --nowrap -O0 -g --emit-llvm-text -o - | FileCheck %s --check-prefix=CHECK-DEBUG
// RUN: not %{ispc} %s -DERRORS --target=host --nostdlib --nowrap -o %t.o 2>&1 | FileCheck %s --check-prefix=CHECK-ERRORS --implicit-check-not=Error: --implicit-check-not=FATAL

#ifndef ERRORS
// CHECK-DEBUG-DAG: !DICompositeType(tag: DW_TAG_enumeration_type, name: "uniform enum LE", {{.*}}line: [[#@LINE+1]], {{.*}}identifier: "LE")
enum LE { A = 1 };
struct S {
    int x;
};
typedef int T;

template <typename E> noinline uniform int en(uniform E e) { return (uniform int)e; }

// CHECK-DAG: define linkonce_odr {{.*}}@en___vyenum_5B_LE_5D____unenum_5B_LE_5D_(
// CHECK-DAG: define internal {{.*}}@en___vyenum_5B_LE.1_5D____unenum_5B_LE.1_5D_(
// CHECK-DAG: define internal {{.*}}@en___vyenum_5B_S.1_5D____unenum_5B_S.1_5D_(
// CHECK-DAG: define internal {{.*}}@en___vyenum_5B_T.1_5D____unenum_5B_T.1_5D_(
uniform int global() {
    uniform LE e = A;
    return en(e);
}
uniform int local() {
    // The local enum has no identifier, through which LLVM would merge it with
    // the global one.
    // CHECK-DEBUG-DAG: !DICompositeType(tag: DW_TAG_enumeration_type, name: "uniform enum LE", {{.*}}line: [[#@LINE+1]], {{.*}}elements: ![[#]]){{$}}
    enum LE { B = 2 };
    uniform LE e = B;
    uniform enum LE e2 = B;
    return en(e) + e2;
}
uniform int otherNames() {
    enum S { C = 3 };
    enum T { D = 4 };
    uniform S s = C;
    uniform T t = D;
    return en(s) + en(t);
}

// The enum hides the outer struct already in its enumerators.
struct Big {
    int x[10];
};
// CHECK-SELF-LABEL: define {{.*}}@selfReference___(
// CHECK-SELF: ret i32 4
uniform int selfReference() {
    enum Big { BA = sizeof(uniform Big) };
    return BA;
}

// The variants of an enum made in its enumerators, here "uniform SelfRef",
// have all of its enumerators.
// CHECK-HEADER: enum SelfRef {
// CHECK-HEADER-NEXT: SR_A = 4,
// CHECK-HEADER-NEXT: SR_B = 5
// CHECK-DEBUG-DAG: !DIEnumerator(name: "SR_A", value: 4)
// CHECK-DEBUG-DAG: !DIEnumerator(name: "SR_B", value: 5)
enum SelfRef { SR_A = sizeof(uniform SelfRef), SR_B };
export uniform int selfRefExport(uniform SelfRef e) { return e == SR_B ? 1 : 0; }

// Structs hide outer types that aren't structs.
uniform int structs() {
    struct T {
        int a;
    };
    struct LE {
        int b;
    };
    uniform T t = {1};
    uniform LE le = {2};
    return t.a + le.b;
}

// A typedef of a template parameter can be hidden.
template <typename P> uniform int templateTypedef(P p) {
    typedef P Q;
    {
        enum Q { QA = 5 };
        uniform Q q = QA;
        return q;
    }
}
template <typename P> uniform int templateTypedef2(P p) {
    typedef P Q;
    {
        struct Q {
            int a;
        };
        uniform Q q = {6};
        return q.a;
    }
}
uniform int callTemplateTypedef() { return templateTypedef<uniform int>(0) + templateTypedef2<uniform int>(0); }

// The outer types are used again after the functions.
uniform int after() {
    uniform enum LE e = A;
    uniform T t = 3;
    return e + t;
}

#else
enum LE { A = 1 };
typedef int T;
struct S {
    int x;
};

// CHECK-ERRORS-DAG: local-type-shadow.ispc:[[#@LINE+1]]:6: Error: Ignoring redefinition of type "LE".
enum LE { B = 2 };
// CHECK-ERRORS-DAG: local-type-shadow.ispc:[[#@LINE+1]]:6: Error: Ignoring redefinition of type "T".
enum T { C = 3 };
// CHECK-ERRORS-DAG: local-type-shadow.ispc:[[#@LINE+1]]:6: Error: Ignoring redefinition of type "S".
enum S { D = 4 };

void sameScope() {
    enum L2 { E };
    // CHECK-ERRORS-DAG: local-type-shadow.ispc:[[#@LINE+1]]:10: Error: Ignoring redefinition of type "L2".
    enum L2 { F };
}

// An enum doesn't complete a forward-declared struct.
struct FS;
// CHECK-ERRORS-DAG: local-type-shadow.ispc:[[#@LINE+1]]:6: Error: Ignoring redefinition of type "FS".
enum FS { G };

// Type names can't be enumerators.
// CHECK-ERRORS-DAG: local-type-shadow.ispc:[[#@LINE+1]]:14: Error: syntax error, unexpected TOKEN_TYPE_NAME
enum EnumT { T = 1 };
// Nor can the name of the enum itself, which is declared before its
// enumerators.
// CHECK-ERRORS-DAG: local-type-shadow.ispc:[[#@LINE+1]]:11: Error: syntax error, unexpected TOKEN_TYPE_NAME
enum SE { SE };

template <typename P> uniform int templateParm(P p) {
    // CHECK-ERRORS-DAG: local-type-shadow.ispc:[[#@LINE+1]]:12: Error: Type "P" is not a struct type!
    struct P {
        int a;
    };
    // CHECK-ERRORS-DAG: local-type-shadow.ispc:[[#@LINE+1]]:10: Error: Enum "P" can't have the name of a template parameter.
    enum P { PA };
    return PA;
}

void notAnEnum() {
    // CHECK-ERRORS-DAG: local-type-shadow.ispc:[[#@LINE+1]]:18: Error: Type "T" is not an enum type
    uniform enum T x;
}
#endif
