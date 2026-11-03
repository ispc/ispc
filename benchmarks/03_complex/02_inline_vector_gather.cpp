// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include <benchmark/benchmark.h>
#include <cstdint>
#include <stdio.h>

#include "../common.h"
#include "02_inline_vector_gather_ispc.h"

static Docs docs("Edge relaxation over 16-byte vertex records read and written through inline helpers with an\n"
                 "all-lanes-on fast path (relax_edges). Every other block of 64 edges is all active (the fast path);\n"
                 "in the others about half of the edges are active, so the gathers and scatters of the partial-mask\n"
                 "path run. No vertex appears twice, so the result matches the scalar C++ version.\n"
                 "Expectation:\n"
                 " - No regressions\n"
                 " - dword gathers and scatters for the int32 indices, also when the helpers are inlined late\n");

WARM_UP_RUN();

// Number of edges; every edge has its own 2 vertices.
#define ARGS Arg(256)->Arg(4096)->Arg(16384)

static uint64_t next_random(uint64_t &s) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
}

// Edge e connects vertices that no other edge uses, in a random order. Half of the edges are in blocks that are all
// active, the others are active with probability 1/2.
static void init(ispc::rec4 *verts, int32_t *ea, int32_t *eb, int8_t *on, int edges) {
    uint64_t s = 0x9E3779B97F4A7C15ull;
    int vcount = 2 * edges;
    for (int i = 0; i < vcount; i++) {
        for (int k = 0; k < 4; k++) {
            verts[i].c[k] = static_cast<float>(next_random(s) % 1000) * 0.01f;
        }
    }
    int32_t *perm = new int32_t[vcount];
    for (int i = 0; i < vcount; i++) {
        perm[i] = i;
    }
    for (int i = vcount - 1; i > 0; i--) {
        int j = static_cast<int>(next_random(s) % static_cast<uint64_t>(i + 1));
        int32_t t = perm[i];
        perm[i] = perm[j];
        perm[j] = t;
    }
    for (int e = 0; e < edges; e++) {
        ea[e] = perm[2 * e];
        eb[e] = perm[2 * e + 1];
        // Every other block of 64 edges is all on, so that the all-lanes-on fast path runs at every gang size.
        on[e] = static_cast<int8_t>((e / 64) % 2 == 0 ? 1 : next_random(s) % 2);
    }
    delete[] perm;
}

static void relax_edges_ref(ispc::rec4 *verts, const int32_t *ea, const int32_t *eb, const int8_t *on, float w,
                            int edges) {
    for (int e = 0; e < edges; e++) {
        if (!on[e])
            continue;
        ispc::rec4 &va = verts[ea[e]];
        ispc::rec4 &vb = verts[eb[e]];
        for (int k = 0; k < 4; k++) {
            float d = (vb.c[k] - va.c[k]) * w;
            va.c[k] += d;
            vb.c[k] -= d;
        }
    }
}

static void relax_edges(benchmark::State &state) {
    int edges = static_cast<int>(state.range(0));
    int vcount = 2 * edges;
    ispc::rec4 *verts = static_cast<ispc::rec4 *>(aligned_alloc_helper(sizeof(ispc::rec4) * vcount));
    ispc::rec4 *ref = static_cast<ispc::rec4 *>(aligned_alloc_helper(sizeof(ispc::rec4) * vcount));
    int32_t *ea = static_cast<int32_t *>(aligned_alloc_helper(sizeof(int32_t) * edges));
    int32_t *eb = static_cast<int32_t *>(aligned_alloc_helper(sizeof(int32_t) * edges));
    int8_t *on = static_cast<int8_t *>(aligned_alloc_helper(edges));
    init(verts, ea, eb, on, edges);
    // The vertices change on every iteration, so the check runs one call on fresh data.
    for (int i = 0; i < vcount; i++) {
        ref[i] = verts[i];
    }
    // w is a power of 2, so (vb - va) * w is exact and the result does not depend on FMA contraction.
    for (auto _ : state) {
        ispc::relax_edges(verts, ea, eb, on, 0.25f, edges);
    }
    for (int i = 0; i < vcount; i++) {
        verts[i] = ref[i];
    }
    ispc::relax_edges(verts, ea, eb, on, 0.25f, edges);
    relax_edges_ref(ref, ea, eb, on, 0.25f, edges);
    for (int i = 0; i < vcount; i++) {
        for (int k = 0; k < 4; k++) {
            if (verts[i].c[k] != ref[i].c[k]) {
                printf("Error i=%d k=%d\n", i, k);
                fail_benchmark(state);
                i = vcount;
                break;
            }
        }
    }
    aligned_free_helper(verts);
    aligned_free_helper(ref);
    aligned_free_helper(ea);
    aligned_free_helper(eb);
    aligned_free_helper(on);
    state.SetComplexityN(state.range(0));
}
BENCHMARK(relax_edges)->ARGS;

CHECKED_BENCHMARK_MAIN();
