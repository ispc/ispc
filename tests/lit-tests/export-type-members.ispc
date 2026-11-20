// A struct exported with "export { S };" or "export S;" gets the same
// declarations in the header as a struct used by an exported function: the
// header also declares the enum, vector, and struct types of its members.

// RUN: %{ispc} %s --target=avx2-i32x8 --nostdlib --nowrap -DISPC -DLIST -o %t.o -h %t_list.h
// RUN: FileCheck --input-file=%t_list.h %s
// RUN: %{cxx} -x c++ -fsyntax-only %s --include %t_list.h
// RUN: %{cc} -x c -fsyntax-only %s --include %t_list.h

// RUN: %{ispc} %s --target=avx2-i32x8 --nostdlib --nowrap -DISPC -DSINGLE -o %t.o -h %t_single.h
// RUN: FileCheck --input-file=%t_single.h %s
// RUN: %{cxx} -x c++ -fsyntax-only %s --include %t_single.h
// RUN: %{cc} -x c -fsyntax-only %s --include %t_single.h

// RUN: %{ispc} %s --target=avx2-i32x8 --nostdlib --nowrap -DISPC -DFUNC -o %t.o -h %t_func.h
// RUN: FileCheck --input-file=%t_func.h %s
// RUN: %{cxx} -x c++ -fsyntax-only %s --include %t_func.h
// RUN: %{cc} -x c -fsyntax-only %s --include %t_func.h

// The header for multiple targets declares them too.
// RUN: %{ispc} %s --target=sse4-i32x4,avx2-i32x8 --nostdlib --nowrap -DISPC -DLIST -o %t.o -h %t_multi.h
// RUN: FileCheck --input-file=%t_multi.h %s
// RUN: %{cxx} -x c++ -fsyntax-only %s --include %t_multi.h
// RUN: %{cc} -x c -fsyntax-only %s --include %t_multi.h

// REQUIRES: X86_ENABLED

// CHECK-DAG: struct float3 {
// CHECK-DAG: struct int32_t2 {
// CHECK-DAG: enum Kind {
// CHECK-DAG: enum Flag {
// CHECK-DAG: struct Inner {
// CHECK-DAG: struct Node {
// CHECK-DAG: struct S {

#ifdef ISPC
enum Kind { KA, KB };
enum Flag { F0 = 1 };
typedef float<3> float3;
typedef int<2> int2;

struct Inner {
    int2 i;
};

struct Node;

struct S {
    Kind k;
    float3 v;
    Inner inner;
    Flag flags[2];
    Node *uniform next;
};

struct Node {
    Kind nk;
};

#if defined(LIST)
export { S };
#elif defined(SINGLE)
export S;
#elif defined(FUNC)
export void f(uniform S *uniform s) {}
#endif
#elif defined(__cplusplus)
int g(ispc::S *s) { return s->k + s->v.v[0] + s->inner.i.v[1] + s->flags[0] + s->next->nk + ispc::KB + ispc::F0; }
#else
int g(struct S *s) { return s->k + s->v.v[0] + s->inner.i.v[1] + s->flags[0] + s->next->nk + KB + F0; }
#endif
