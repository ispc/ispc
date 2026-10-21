// Check that on Xe, template instantiations over structs declared in a function
// body or over anonymous structs get internal linkage. Xe marks functions
// without "noinline" as alwaysinline but keeps their external linkage, so the
// templates here are not "noinline". The IR is checked before inlining.
// See https://github.com/ispc/ispc/issues/3962

// RUN: %{ispc} %s --emit-llvm-text --target=xehpg-x8 --nostdlib --nowrap -O0 --debug-phase=first:first -o /dev/null | FileCheck %s

// REQUIRES: XE_ENABLED

struct G {
    int x;
};

template <typename U> uniform int rd(U *uniform s) { return s[1].x; }
template <typename U> noinline uniform int rdNoInline(U *uniform s) { return s[1].x; }

// Global struct: Xe keeps external linkage.
// CHECK-DAG: define spir_func i32 @rd___s_5B_unG_5D___
uniform int fglobal(uniform G *uniform v) { return rd(v); }

// Local struct.
// CHECK-DAG: define internal spir_func i32 @rd___s_5B_unL.1_5D___
// CHECK-DAG: define internal spir_func i32 @rdNoInline___s_5B_unL.1_5D___
uniform int flocal(void *uniform v) {
    struct L {
        int x;
    };
    uniform L *uniform p = (uniform L * uniform) v;
    return rd(p) + rdNoInline(p);
}

// Global anonymous struct.
// CHECK-DAG: define internal spir_func i32 @"rd___s_5B_un$anon{{[0-9]+}}_5D___
uniform struct {
    int x;
} anon[2];
uniform int fanon() { return rd(anon); }
