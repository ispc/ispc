// Pins the decisions of GatherCoalescePass through its performance warnings,
// which name the gathers of a group and the loads chosen for them. The pass
// only sees "factored" gathers, so it runs on targets without a gather
// instruction (sse4 here) or with --opt=disable-gathers, and only where the
// mask is known to be all on. The checks describe today's output; each case
// also gives the ideal output in an "Ideal:" comment, and the change that
// reaches it updates the checks of that case.
//
// Except pair_x64, the functions are not exported, so each has an all-on path
// (coalesced) and a partial-mask path (plain gathers, which also warn); only
// the "Coalesced" warnings are checked. Their order follows the pass, so they
// are matched in any order and counted separately.

// RUN: %{ispc} %s --target=sse4-i32x4 --nowrap -o /dev/null 2> %t.sse4
// RUN: FileCheck --input-file=%t.sse4 %s --check-prefix=WARN
// RUN: FileCheck --input-file=%t.sse4 %s --check-prefix=COUNT13
// RUN: %{ispc} %s --target=sse4-i32x4 --nowrap --opt=force-aligned-memory --emit-llvm-text -o - 2> /dev/null \
// RUN:   | FileCheck %s --check-prefix=ALIGN
// RUN: %{ispc} %s --target=avx512skx-x64 --opt=disable-gathers --nowrap -o /dev/null 2> %t.x64
// RUN: FileCheck --input-file=%t.x64 %s --check-prefix=X64

// REQUIRES: X86_ENABLED

// COUNT13-COUNT-13: Performance Warning: Coalesced
// COUNT13-NOT: Performance Warning: Coalesced

// Two gathers of neighbouring elements: one group, one 8-wide load.
// With --opt=force-aligned-memory the load gets the native vector alignment.
// Ideal: = today.
// ALIGN-LABEL: @pair___
// ALIGN: load <8 x i32>, ptr {{%.*}}, align 16
// ALIGN-LABEL: @pair_x64___
void pair(uniform float buf[], uniform float r[]) {
    // WARN-DAG: coalesce-gathers.ispc:[[@LINE+1]]:{{[0-9]+}}: Performance Warning: Coalesced 2 gathers starting here (other at line [[@LINE+2]]) into 1 load (1 x 8-wide).
    float a = buf[2 * programIndex];
    float b = buf[2 * programIndex + 1];
    r[programIndex] = a + b;
}

// The same at width 64 with --opt=disable-gathers (avx512skx-x64): this
// assembles 64-wide results. Exported, so that the mask is known to be all on
// (AVX-512 functions have no separate all-on path).
// On sse4 the external copy of the exported function is coalesced too, and its
// warning has lost the source position (the metadata of the pseudo calls is
// dropped, see the TODO in GatherCoalescePass).
// Ideal: one warning per source operation, with its position.
// WARN-DAG: {{^}}Performance Warning: Coalesced 2 gathers starting here (other at line ) into 1 load (1 x 8-wide).
// WARN-DAG: coalesce-gathers.ispc:[[@LINE+3]]:{{[0-9]+}}: Performance Warning: Coalesced 2 gathers starting here (other at line [[@LINE+4]]) into 1 load (1 x 8-wide).
export void pair_x64(uniform float buf[], uniform float r[]) {
    // X64: coalesce-gathers.ispc:[[@LINE+1]]:{{[0-9]+}}: Performance Warning: Coalesced 2 gathers starting here (other at line [[@LINE+2]]) into 16 loads (16 x 8-wide).
    float a = buf[2 * programIndex];
    float b = buf[2 * programIndex + 1];
    r[programIndex] = a + b;
}

// Four gathers of a 4-element record: one group of the maximum size (4), two
// 8-wide loads.
// Ideal: = today.
// WARN-DAG: coalesce-gathers.ispc:[[@LINE+2]]:{{[0-9]+}}: Performance Warning: Coalesced 4 gathers starting here (others at lines [[@LINE+3]], [[@LINE+4]], [[@LINE+5]]) into 2 loads (2 x 8-wide).
void quad(uniform float buf[], uniform float r[]) {
    float a = buf[4 * programIndex];
    float b = buf[4 * programIndex + 1];
    float c = buf[4 * programIndex + 2];
    float d = buf[4 * programIndex + 3];
    r[programIndex] = a + b + c + d;
}

// Neighbouring lanes swapped ({1, 0, 3, 2}): one 4-wide load and a shuffle.
// Ideal: = today.
// WARN-DAG: coalesce-gathers.ispc:[[@LINE+1]]:{{[0-9]+}}: Performance Warning: Coalesced gather into 1 load (1 x 4-wide).
void swap_pairs(uniform float buf[], uniform float r[]) { r[programIndex] = buf[programIndex ^ 1]; }

// Offsets {0, 0, 1, 1}: one 2-wide (64-bit) load.
// Ideal: = today.
// WARN-DAG: coalesce-gathers.ispc:[[@LINE+1]]:{{[0-9]+}}: Performance Warning: Coalesced gather into 1 load (1 x 2-wide).
void halves(uniform float buf[], uniform float r[]) { r[programIndex] = buf[programIndex / 2]; }

// Elements 0 and 2 of 4-element records (offsets 0, 2, 4, ... 14): elements
// 0-7 come from one 8-wide load. For 8, 10, 12 and 14 the 8-wide load runs out
// of offsets before it is full, the 4-wide load would use only half of its
// elements, and a 2-wide load is never used for a single needed element, so
// they are 4 scalar loads.
// Ideal: a second vector load for 8-14 where its footprint is known to be
// readable.
// WARN-DAG: coalesce-gathers.ispc:[[@LINE+2]]:{{[0-9]+}}: Performance Warning: Coalesced 2 gathers starting here (other at line [[@LINE+3]]) into 5 loads (4 x 1-wide, 1 x 8-wide).
void with_gap(uniform float buf[], uniform float r[]) {
    float a = buf[4 * programIndex];
    float c = buf[4 * programIndex + 2];
    r[programIndex] = a + c;
}

// A store between two gathers ends the group, so each gather is coalesced on
// its own (here the store writes the element the second gather reads).
// Ideal: = today.
// WARN-DAG: coalesce-gathers.ispc:[[@LINE+3]]:{{[0-9]+}}: Performance Warning: Coalesced gather into 4 loads (4 x 1-wide).
// WARN-DAG: coalesce-gathers.ispc:[[@LINE+4]]:{{[0-9]+}}: Performance Warning: Coalesced gather into 4 loads (4 x 1-wide).
void store_between(uniform float buf[], uniform float r[]) {
    float a = buf[4 * programIndex];
    buf[4 * programIndex + 1] = 0;
    float b = buf[4 * programIndex + 1];
    r[programIndex] = a + b;
}

// The same for an atomic, which may write any memory.
// Ideal: = today (cnt may alias buf).
// WARN-DAG: coalesce-gathers.ispc:[[@LINE+3]]:{{[0-9]+}}: Performance Warning: Coalesced gather into 4 loads (4 x 1-wide).
// WARN-DAG: coalesce-gathers.ispc:[[@LINE+4]]:{{[0-9]+}}: Performance Warning: Coalesced gather into 4 loads (4 x 1-wide).
void atomic_between(uniform float buf[], uniform int cnt[], uniform float r[]) {
    float a = buf[2 * programIndex];
    atomic_add_global(&cnt[0], 1);
    float b = buf[2 * programIndex + 1];
    r[programIndex] = a + b;
}

// Gathers from two different bases: only those with the same base are
// grouped, across the one in between.
// Ideal: = today.
// WARN-DAG: coalesce-gathers.ispc:[[@LINE+3]]:{{[0-9]+}}: Performance Warning: Coalesced 2 gathers starting here (other at line [[@LINE+5]]) into 1 load (1 x 8-wide).
// WARN-DAG: coalesce-gathers.ispc:[[@LINE+3]]:{{[0-9]+}}: Performance Warning: Coalesced gather into 4 loads (4 x 1-wide).
void two_bases(uniform float buf[], uniform float other[], uniform float r[]) {
    float a = buf[2 * programIndex];
    float b = other[2 * programIndex + 1];
    float c = buf[2 * programIndex + 1];
    r[programIndex] = a + b + c;
}
