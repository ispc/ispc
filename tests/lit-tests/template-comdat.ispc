// Check that implicit and explicit template instantiations, which may be in
// several translation units, are in COMDATs, so the linker keeps only one of
// them; COFF linkers reject them as duplicates otherwise.  Specializations and
// instantiations with internal linkage are not, and neither are template
// instantiations of the standard library, which are internal in each module.

// RUN: %{ispc} %s --target=host --nostdlib --nowrap -O0 --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s -DSTDLIB --arch=x86-64 --target=avx512icl-x16 --nowrap --woff -O0 -o %t.o --dump-file=%t --debug-phase=first:first
// RUN: FileCheck %s --check-prefix=CHECK-STDLIB --input-file %t/ir_10_VerifierPass.ll --implicit-check-not="= comdat" --implicit-check-not=") comdat"

// REQUIRES: !MACOS_HOST && X86_ENABLED

// CHECK-STDLIB: define internal {{.*}}@__rotate_via_shuffle{{.*}} #{{[0-9]+}} {

// CHECK-DAG: $implicit___uni___uni = comdat any
// CHECK-DAG: $explicit___uni___uni = comdat any
// CHECK-DAG: define linkonce_odr {{.*}}@implicit___uni___uni({{.*}} comdat {
// CHECK-DAG: define weak_odr {{.*}}@explicit___uni___uni({{.*}} comdat {
// CHECK-DAG: define {{.*}}@special___uni___uni({{.*}} #{{[0-9]+}} {
// CHECK-DAG: define internal {{.*}}@local___s_5B_unL.1_5D____s_5B_unL.1_5D_({{.*}} #{{[0-9]+}} {

#ifdef STDLIB
export void rot(uniform int8 a[], uniform int n) { a[programIndex] = rotate(a[programIndex], n); }
#else
template <typename T> noinline uniform int implicit(T t) { return (uniform int)t; }
template <typename T> noinline uniform int explicit(T t) { return (uniform int)t; }
template uniform int explicit<uniform int>(uniform int t);
template <typename T> noinline uniform int special(T t) { return (uniform int)t; }
template <> noinline uniform int special<uniform int>(uniform int t) { return 1; }
template <typename T> noinline uniform int local(T t) { return 0; }

uniform int call() {
    struct L {
        int x;
    };
    uniform L l;
    return implicit<uniform int>(1) + explicit<uniform int>(2) + special<uniform int>(3) + local(l);
}
#endif
