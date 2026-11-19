// Regression test for https://github.com/ispc/ispc/issues/285
// "export Foo;" adds a single type to the generated header, like
// "export { Foo };" does.

// RUN: %{ispc} %s --target=host --nostdlib --nowrap -DISPC -o %t.o -h %t.h 2>&1 | FileCheck %s --check-prefix=WARN --allow-empty
// RUN: FileCheck --input-file=%t.h %s
// RUN: %{cxx} -x c++ -fsyntax-only %s --include %t.h
// RUN: not %{ispc} %s --target=host --nostdlib --nowrap -DERR -o %t.o 2>&1 | FileCheck %s --check-prefix=ERR --implicit-check-not=Error:
// RUN: not %{ispc} %s --target=host --nostdlib --nowrap -DERR_SYNTAX -o %t.o 2>&1 | FileCheck %s --check-prefix=ERR_SYNTAX --implicit-check-not=Error:

// WARN-NOT: Error
// WARN-NOT: Warning

// CHECK-DAG: struct Foo {
// CHECK-DAG: struct Bar {
// CHECK-DAG: struct Baz {
// CHECK-DAG: enum E {
// CHECK-DAG: struct float3 {
// CHECK-DAG: struct double2 {
// CHECK-DAG: struct Qux {
// CHECK-DAG: struct Quux {
// CHECK-DAG: struct Tr {
// CHECK-DAG: struct Fwd {
// CHECK-DAG: struct TA {

#ifdef ERR
typedef int I;

// ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Only struct, vector, and enum types, not "/*unbound*/ int32", are allowed in type export lists.
export I;

struct S {
    int a;
};

// ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Only a type name is allowed after "export" in a type export declaration.
export static S;
// ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Only a type name is allowed after "export" in a type export declaration.
export typedef S;
// ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Only a type name is allowed after "export" in a type export declaration.
export const S;
// ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Only a type name is allowed after "export" in a type export declaration.
export soa<4> S;
// ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Only a type name is allowed after "export" in a type export declaration.
export soa<0> S;
struct T {
    int b;
};
// ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Only a type name is allowed after "export" in a type export declaration.
export S T;
// ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Only a type name is allowed after "export" in a type export declaration.
export int<3>;
// ERR: :[[@LINE+2]]:{{[0-9]+}}: Error: Unknown identifier "N" is used to specify the size of a vector type.
// ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Only a type name is allowed after "export" in a type export declaration.
export int<N>;
// ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Only a type name is allowed after "export" in a type export declaration.
export uniform S;
// ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Only a type name is allowed after "export" in a type export declaration.
export varying S;
typedef uniform S US;
// ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Only a type name is allowed after "export" in a type export declaration.
export varying US;
// ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Only a type name is allowed after "export" in a type export declaration.
export uniform US;
// ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Only a type name is allowed after "export" in a type export declaration.
export US uniform;
// ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Only a type name is allowed after "export" in a type export declaration.
export __declspec(safe) S;
// ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Only a type name is allowed after "export" in a type export declaration.
export __attribute__((noescape)) S;
// ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Expected a type name after "export".
export;

// A type or declarator that is invalid gets only its own error.
// ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Type "I" is not a struct type! (/*unbound*/ int32)
export struct I;
// ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Enum type "Undef" unknown.
export enum Undef;
// ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Array dimension must be non-negative.
export int x[-1];

// A struct must be defined before it is exported, in both forms.
struct Later;
// ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Struct "Later" can't be exported before it is defined.
export struct Later;
// ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Struct "Later" can't be exported before it is defined.
export { Later };
struct Later {
    int l;
};

// Anonymous types can't be exported, in both forms.
// ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Anonymous struct types can't be exported; declare the type as "struct Name { ... }".
export struct { int c; };
// ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Anonymous enum types can't be exported; declare the type as "enum Name { ... }".
export enum { AE0, AE1 };
// ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Anonymous struct types can't be exported; declare the type as "struct Name { ... }".
export { struct { int d; } };
// ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Anonymous enum types can't be exported; declare the type as "enum Name { ... }".
export { enum { AE2, AE3 } };
typedef enum { TE0, TE1 } TE;
// ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Anonymous enum types can't be exported; declare the type as "enum Name { ... }".
export TE;

// Type exports are only allowed at global scope.
void f() {
    // ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Type export declarations are only allowed at global scope.
    export S;
    // ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Type export declarations are only allowed at global scope.
    export struct Local { int e; };
    // ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Type export declarations are only allowed at global scope.
    for (export S;;) {
    }
}
#elif defined(ERR_SYNTAX)
// A syntax error in an exported function prototype gets only its own error.
// ERR_SYNTAX: :[[@LINE+1]]:{{[0-9]+}}: Error: syntax error, unexpected identifier.
export void foo(bogus x);
#elif defined(ISPC)
struct Foo {
    int x;
};

struct Bar {
    float y;
};

enum E { A, B };

typedef float<3> float3;
typedef uniform Foo UFoo;

export Foo;
export UFoo;
export struct Bar;
export enum E;
export float3;

export struct Baz {
    int z;
};

// The emitted vector type is named after its element type and size, not
// after the typedef.
typedef double<2> MyVec;
export MyVec;

// The uniform/varying qualifier a typedef carries is ignored.
struct Qux {
    int q;
};
typedef uniform Qux UQux;
export UQux;
struct Quux {
    int q;
};
typedef varying Quux VQuux;
export VQuux;

// "export" may follow the type name.
struct Tr {
    int t;
};
Tr export;

// A typedef of a forward declared struct refers to the struct's definition.
struct Fwd;
typedef struct Fwd FwdT;
struct Fwd {
    int w;
};
export FwdT;

// A typedef at global scope names an anonymous struct.
typedef struct {
    int a;
} TA;
export TA;
#else
int g(ispc::Foo *f, ispc::Bar *b, ispc::Baz *z, ispc::float3 *v, ispc::double2 *d, ispc::Qux *q, ispc::Quux *qq,
      ispc::Tr *t, ispc::Fwd *w, ispc::TA *a) {
    return f->x + b->y + z->z + v->v[0] + d->v[0] + q->q + qq->q + t->t + w->w + a->a + ispc::B;
}
#endif
