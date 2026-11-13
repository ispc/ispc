// Regression test for https://github.com/ispc/ispc/issues/285
// "export Foo;" adds a single type to the generated header, like
// "export { Foo };" does.

// RUN: %{ispc} %s --target=host --nostdlib --nowrap -DISPC -o %t.o -h %t.h 2>&1 | FileCheck %s --check-prefix=WARN --allow-empty
// RUN: FileCheck --input-file=%t.h %s
// RUN: %{cxx} -x c++ -fsyntax-only %s --include %t.h
// RUN: not %{ispc} %s --target=host --nostdlib --nowrap -DERR -o %t.o 2>&1 | FileCheck %s --check-prefix=ERR --implicit-check-not=Error:

// WARN-NOT: Error
// WARN-NOT: Warning

// CHECK-DAG: struct Foo {
// CHECK-DAG: struct Bar {
// CHECK-DAG: struct Baz {
// CHECK-DAG: enum E {
// CHECK-DAG: struct float3 {

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
export int<3>;
// ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Only a type name is allowed after "export" in a type export declaration.
export uniform S;
// ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Only a type name is allowed after "export" in a type export declaration.
export varying S;
// ERR: :[[@LINE+1]]:{{[0-9]+}}: Error: Expected a type name after "export".
export;
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
#else
int g(ispc::Foo *f, ispc::Bar *b, ispc::Baz *z, ispc::float3 *v) { return f->x + b->y + z->z + v->v[0] + ispc::B; }
#endif
