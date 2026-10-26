// Guards two properties of memory operations inside `inline` helpers that a
// change of the inlining order can lose (for example, inlining the helpers
// after gathers and scatters are lowered, as PR #3808 did):
//
// 1. relax_edges: a 16-byte record read and written through large inline
//    helpers with an int32 index and a caller-provided base keeps dword
//    gathers and scatters. With the helpers inlined late, the index is
//    widened to 64 bits first, and the gathers become qword (twice as many).
// 2. sample_grid: an unmasked inline template helper that reads a local
//    uniform array declared inside `foreach` leaves no llvm.stacksave in the
//    final IR. Inlined late, the array becomes an alloca in the loop, wrapped
//    in llvm.stacksave/llvm.stackrestore.
//
// On avx512skx-x16 the late inlining leaves no llvm.stacksave either, so the
// second property is checked on avx2-i32x8 and sse4-i32x4.
// Ideal: = today for both.
//
// The helpers must stay large: LLVM inlines small helpers on cost before the
// memory operations are lowered, which hides the problem. The per-lane select
// loop in fetch_rec keeps fetch_rec large.

// RUN: %{ispc} %s --target=avx2-i32x8 --nowrap --emit-llvm-text -o - \
// RUN:   | FileCheck %s --check-prefixes=CHECK,GS -DGD=llvm.x86.avx2.gather.d.ps.256 -DGQ=llvm.x86.avx2.gather.q.ps.256
// RUN: %{ispc} %s --target=avx512skx-x16 --nowrap --emit-llvm-text -o - \
// RUN:   | FileCheck %s --check-prefixes=CHECK,GS,SKX -DGD=llvm.x86.avx512.mask.gather.dps.512 -DGQ=llvm.x86.avx512.mask.gather.qps.512
// RUN: %{ispc} %s --target=sse4-i32x4 --nowrap --emit-llvm-text -o - | FileCheck %s

// REQUIRES: X86_ENABLED

// Both copies of each function are checked: the internal one (`___` suffix,
// emitted first) and the exported one. The CHECK-NOT lines are repeated
// around every positive check, so that they cover the whole function.

// CHECK-LABEL: @relax_edges___
// GS-NOT: call {{.*}}@[[GQ]](
// SKX-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.qps.512(
// CHECK-NOT: call {{.*}}@llvm.stacksave
// GS: call {{.*}}@[[GD]](
// GS-NOT: call {{.*}}@[[GQ]](
// SKX-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.qps.512(
// CHECK-NOT: call {{.*}}@llvm.stacksave
// SKX: call {{.*}}@llvm.x86.avx512.mask.scatter.dps.512(
// GS-NOT: call {{.*}}@[[GQ]](
// SKX-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.qps.512(
// CHECK-NOT: call {{.*}}@llvm.stacksave

// CHECK-LABEL: @sample_grid___
// CHECK-NOT: call {{.*}}@llvm.stacksave
// CHECK: store <{{[0-9]+}} x float>
// CHECK-NOT: call {{.*}}@llvm.stacksave

// CHECK-LABEL: @relax_edges(
// GS-NOT: call {{.*}}@[[GQ]](
// SKX-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.qps.512(
// CHECK-NOT: call {{.*}}@llvm.stacksave
// GS: call {{.*}}@[[GD]](
// GS-NOT: call {{.*}}@[[GQ]](
// SKX-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.qps.512(
// CHECK-NOT: call {{.*}}@llvm.stacksave
// SKX: call {{.*}}@llvm.x86.avx512.mask.scatter.dps.512(
// GS-NOT: call {{.*}}@[[GQ]](
// SKX-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.qps.512(
// CHECK-NOT: call {{.*}}@llvm.stacksave

// CHECK-LABEL: @sample_grid(
// CHECK-NOT: call {{.*}}@llvm.stacksave
// CHECK: store <{{[0-9]+}} x float>
// CHECK-NOT: call {{.*}}@llvm.stacksave

struct rec4 {
    float c[4];
};

// All lanes on: per-lane uniform record loads, then a transpose.
inline rec4 fetch_rec(const uniform rec4 *varying p) {
    rec4 r;
    if (popcnt(lanemask()) == programCount) {
        uniform float tmp[4 * programCount];
        foreach_active (l) {
            const uniform rec4 *uniform q = (const uniform rec4 *uniform)extract((int64)p, l);
            for (uniform int k = 0; k < 4; ++k)
                tmp[4 * l + k] = q->c[k];
        }
        for (uniform int k = 0; k < 4; ++k) {
            float acc = 0;
            for (uniform int l = 0; l < programCount; ++l)
                acc = (programIndex == l) ? tmp[4 * l + k] : acc;
            r.c[k] = acc;
        }
    } else {
#pragma ignore warning(perf)
        r = *p;
    }
    return r;
}

inline void put_rec(uniform rec4 *varying p, const rec4 &r) {
    if (popcnt(lanemask()) == programCount) {
        uniform float tmp[4 * programCount];
        soa_to_aos4(r.c[0], r.c[1], r.c[2], r.c[3], tmp);
        foreach_active (l) {
            uniform rec4 *uniform q = (uniform rec4 *uniform)extract((int64)p, l);
            for (uniform int k = 0; k < 4; ++k)
                q->c[k] = tmp[4 * l + k];
        }
    } else {
#pragma ignore warning(perf)
        *p = r;
    }
}

export void relax_edges(uniform rec4 verts[], const uniform int32 ea[], const uniform int32 eb[], uniform float w,
                        uniform int n) {
    foreach (e = 0 ... n) {
        int32 a = ea[e], b = eb[e];
        rec4 va = fetch_rec(&verts[a]);
        rec4 vb = fetch_rec(&verts[b]);
        for (uniform int k = 0; k < 4; ++k) {
            float d = (vb.c[k] - va.c[k]) * w;
            va.c[k] += d;
            vb.c[k] -= d;
        }
        put_rec(&verts[a], va);
        put_rec(&verts[b], vb);
    }
}

template <typename T>
unmasked inline void deinterleave4(uniform T cell[], varying T *uniform x, varying T *uniform y, varying T *uniform z,
                                   varying T *uniform w) {
    const varying T a = cell[programIndex];
    const varying T b = cell[programCount + programIndex];
    const varying T c = cell[2 * programCount + programIndex];
    const varying T d = cell[3 * programCount + programIndex];
    const varying int lo = (programIndex >> 1) + ((programIndex & 1) ? programCount : 0);
    const varying int hi = lo + programCount / 2;
    const varying T p = shuffle(a, c, lo);
    const varying T q = shuffle(a, c, hi);
    const varying T r = shuffle(b, d, lo);
    const varying T s = shuffle(b, d, hi);
    *x = shuffle(p, r, lo);
    *y = shuffle(p, r, hi);
    *z = shuffle(q, s, lo);
    *w = shuffle(q, s, hi);
}

export void sample_grid(uniform int16 grid[], uniform float out[], uniform int n) {
    foreach (i = 0 ... n) {
        varying int16 x, y, z, w;
        uniform int16 cell[programCount * 4];
        for (uniform int k = 0; k < programCount * 4; ++k)
            cell[k] = grid[extract(i, 0) * 4 + k];
        deinterleave4(cell, &x, &y, &z, &w);
        out[i] = x + y + z + w;
    }
}
