#include "test_static.isph"
// Edge relaxation over 16-byte vertex records read and written through large
// inline helpers with an all-lanes-on fast path (the kernel of the
// inline-helper-gather lit test), compared with a scalar version. Every
// other gang-sized block of edges has all edges on, which runs the fast
// path; in the others every third edge is off, which runs the gathers and
// scatters of the partial-mask path. No vertex appears twice in one gang,
// so the result does not depend on the order of the lanes.

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

void relax_edges(uniform rec4 verts[], const uniform int32 ea[], const uniform int32 eb[], const uniform bool on[],
                 uniform float w, uniform int n) {
    foreach (e = 0 ... n) {
        if (on[e]) {
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
}

void relax_edges_scalar(uniform rec4 verts[], const uniform int32 ea[], const uniform int32 eb[],
                        const uniform bool on[], uniform float w, uniform int n) {
    for (uniform int e = 0; e < n; ++e) {
        if (!on[e])
            continue;
        uniform rec4 *uniform va = &verts[ea[e]];
        uniform rec4 *uniform vb = &verts[eb[e]];
        for (uniform int k = 0; k < 4; ++k) {
            uniform float d = (vb->c[k] - va->c[k]) * w;
            va->c[k] += d;
            vb->c[k] -= d;
        }
    }
}

#define NV 512 // vertices
#define NE 203 // edges; not a multiple of the gang size

task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform rec4 v[NV], ref[NV];
    uniform int32 ea[NE], eb[NE];
    uniform bool on[NE];
    for (uniform int i = 0; i < NV; ++i)
        for (uniform int k = 0; k < 4; ++k)
            v[i].c[k] = ref[i].c[k] = aFOO[(i + k) % 64] + i;
    // Edge e of gang block g = e / programCount connects vertices of that
    // block only: 2 * programCount distinct vertices per block.
    for (uniform int e = 0; e < NE; ++e) {
        uniform int g = e / programCount, l = e % programCount;
        uniform int lo = (2 * programCount * g) % NV;
        ea[e] = lo + l;
        eb[e] = lo + 2 * programCount - 1 - l;
        // Half the blocks all on (the fast path), in the others every
        // third edge off.
        on[e] = g % 2 == 0 || l % 3 != 0;
    }
    relax_edges(v, ea, eb, on, 0.25, NE);
    relax_edges_scalar(ref, ea, eb, on, 0.25, NE);
    uniform int err = 0;
    for (uniform int i = 0; i < NV; ++i)
        for (uniform int k = 0; k < 4; ++k)
            err += v[i].c[k] != ref[i].c[k] ? 1 : 0;
    RET[programIndex] = err;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
