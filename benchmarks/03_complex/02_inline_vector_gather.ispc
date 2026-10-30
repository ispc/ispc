// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// Edge relaxation over 16-byte vertex records, read and written through large inline helpers with an all-lanes-on
// fast path (per-lane uniform loads and stores with a transpose) and gathers and scatters otherwise. The int32 edge
// indices select records from a base pointer that the caller provides.
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

export void relax_edges(uniform rec4 verts[], const uniform int32 ea[], const uniform int32 eb[],
                        const uniform int8 on[], uniform float w, uniform int n) {
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
