// Checks that --dump-memory-plans prints a line for every decision of
// ImproveMemoryOps and GatherCoalescePass: the rule and the plan it returned,
// or keep. sse4 has no gather instruction, so its gathers take the factored
// form; avx2 takes the base-offsets form.

// RUN: %{ispc} %s --target=sse4-i32x4 --arch=x86-64 --nostdlib --nowrap -O2 --woff --dump-memory-plans -o %t.o 2>&1 | FileCheck %s --check-prefixes=CHECK,FACT --implicit-check-not="linear___{{.*}} -> masked-load"
// RUN: %{ispc} %s --target=avx2-i32x8 --arch=x86-64 --nostdlib --nowrap -O2 --woff --dump-memory-plans -o %t.o 2>&1 | FileCheck %s --check-prefixes=CHECK,NAT --implicit-check-not="linear___{{.*}} -> masked-load"

// REQUIRES: X86_ENABLED

// A linear gather becomes a based gather and then a vector load. Each step
// of the access is a line of the same pass stage and position, with the
// frontend's id of the access (the first access emitted in linear) and the
// next step. No pass rewrites the masked load
// (--implicit-check-not): the function's mask isn't known.
// FACT:      memory-plan: ImproveMemoryOps@[[STAGE:[0-9]+]] [[LIN:linear___[^ ]*]] [[POS:[^ ]*dump-memory-plans.ispc:]][[@LINE+10]]:[[COL:[0-9]+]] [[LIN]]:1#0 __pseudo_gather64_float -> gs-to-based: based __pseudo_gather_factored_base_offsets32_float scale=1 offsets=i32 delta=expr
// FACT-NEXT: memory-plan: ImproveMemoryOps@[[STAGE]] [[LIN]] [[POS]][[@LINE+9]]:[[COL]] [[LIN]]:1#1 __pseudo_gather_factored_base_offsets32_float -> more-const: based __pseudo_gather_factored_base_offsets32_float scale=1 offsets=i32 delta=expr
// FACT-NEXT: memory-plan: ImproveMemoryOps@[[STAGE]] [[LIN]] [[POS]][[@LINE+8]]:[[COL]] [[LIN]]:1#2 __pseudo_gather_factored_base_offsets32_float -> gs-to-load-store: load __masked_load_float offsets=i32 pointer=lane0
// FACT-NEXT: memory-plan: ImproveMemoryOps@[[STAGE]] [[LIN]] [[POS]][[@LINE+7]]:[[COL]] [[LIN]]:1#3 __masked_load_float -> keep
// NAT:       memory-plan: ImproveMemoryOps@[[STAGE:[0-9]+]] [[LIN:linear___[^ ]*]] [[POS:[^ ]*dump-memory-plans.ispc:]][[@LINE+6]]:[[COL:[0-9]+]] [[LIN]]:1#0 __pseudo_gather64_float -> gs-to-based: based __pseudo_gather_base_offsets32_float scale=1 offsets=i32{{$}}
// NAT-NEXT:  memory-plan: ImproveMemoryOps@[[STAGE]] [[LIN]] [[POS]][[@LINE+5]]:[[COL]] [[LIN]]:1#1 __pseudo_gather_base_offsets32_float -> gs-to-load-store: load __masked_load_float offsets=i32 pointer=lane0
// NAT-NEXT:  memory-plan: ImproveMemoryOps@[[STAGE]] [[LIN]] [[POS]][[@LINE+4]]:[[COL]] [[LIN]]:1#2 __masked_load_float -> keep
// The store of the result is the second access of linear, a masked store
// with the mask all on. The frontend doesn't give masked stores a position.
// CHECK-NEXT: memory-plan: ImproveMemoryOps@[[STAGE]] [[LIN]] ? [[LIN]]:2#0 __pseudo_masked_store_float -> masked-store: unmasked align=4
float linear(uniform float a[]) { return a[programIndex]; }

// A gather of decreasing addresses with the mask all on becomes a load of
// the lowest address and a reverse.
// FACT: memory-plan: ImproveMemoryOps@{{[0-9]+}} [[BWD:backward___[^ ]*]] {{[^ ]+}} [[BWD]]:1#1 __pseudo_gather_factored_base_offsets32_float -> gs-to-load-store: reverse-load __masked_load_float offsets=i32 pointer=last
// NAT:  memory-plan: ImproveMemoryOps@{{[0-9]+}} [[BWD:backward___[^ ]*]] {{[^ ]+}} [[BWD]]:1#1 __pseudo_gather_base_offsets32_float -> gs-to-load-store: reverse-load __masked_load_float offsets=i32 pointer=last
float backward(uniform float a[], uniform int n) {
    float r;
    unmasked { r = a[n - programIndex]; }
    return r;
}

// A gather of varying addresses stays a gather, and the next run of
// ImproveMemoryOps sees the same access again. The coalescer only looks at
// factored gathers.
// FACT:      memory-plan: ImproveMemoryOps@[[STAGE1:[0-9]+]] [[IDX:indexed___[^ ]*]] [[IPOS:[^ ]*dump-memory-plans.ispc:]][[@LINE+8]]:[[ICOL:[0-9]+]] [[IDX]]:1#0 __pseudo_gather64_float -> gs-to-based: based __pseudo_gather_factored_base_offsets32_float scale=1 offsets=i32 delta=expr
// FACT-NEXT: memory-plan: ImproveMemoryOps@[[STAGE1]] [[IDX]] [[IPOS]][[@LINE+7]]:[[ICOL]] [[IDX]]:1#1 __pseudo_gather_factored_base_offsets32_float -> keep
// FACT:      memory-plan: ImproveMemoryOps@{{[0-9]+}} [[IDX]] [[IPOS]][[@LINE+6]]:[[ICOL]] [[IDX]]:1#0 __pseudo_gather_factored_base_offsets32_float -> keep
// FACT:      memory-plan: GatherCoalesce@{{[0-9]+}} [[IDX]] [[IPOS]][[@LINE+5]]:[[ICOL]] [[IDX]]:1#0 __pseudo_gather_factored_base_offsets32_float -> keep: mask
// NAT:       memory-plan: ImproveMemoryOps@[[STAGE1:[0-9]+]] [[IDX:indexed___[^ ]*]] [[IPOS:[^ ]*dump-memory-plans.ispc:]][[@LINE+4]]:[[ICOL:[0-9]+]] [[IDX]]:1#0 __pseudo_gather64_float -> gs-to-based: based __pseudo_gather_base_offsets32_float scale=1 offsets=i32{{$}}
// NAT-NEXT:  memory-plan: ImproveMemoryOps@[[STAGE1]] [[IDX]] [[IPOS]][[@LINE+3]]:[[ICOL]] [[IDX]]:1#1 __pseudo_gather_base_offsets32_float -> keep
// NAT:       memory-plan: ImproveMemoryOps@{{[0-9]+}} [[IDX]] [[IPOS]][[@LINE+2]]:[[ICOL]] [[IDX]]:1#0 __pseudo_gather_base_offsets32_float -> keep
// NAT:       memory-plan: GatherCoalesce@{{[0-9]+}} [[IDX]] [[IPOS]][[@LINE+1]]:[[ICOL]] [[IDX]]:1#0 __pseudo_gather_base_offsets32_float -> keep: not a factored gather
float indexed(uniform float a[], int i) { return a[i]; }

// Two factored gathers with the same uniform offsets and constant lane
// offsets are coalesced into one load of 8 elements; lanes= gives the offset
// of every lane of each gather in elements. SimplifyCFG merges the copies
// of the gathers for the all on and the mixed mask and drops their ids and
// positions, so the first run of ImproveMemoryOps gives them the ids
// <function>@<stage>:<n> and <function>@<stage>:<n+1>.
// FACT:      memory-plan: GatherCoalesce@{{[0-9]+}} [[AOS:aos___[^ ]*]] {{[^ ]+}} [[AOS]]@[[ST:[0-9]+]]:[[#G1:]]#0 __pseudo_gather_factored_base_offsets32_float -> coalesce: gathers=2
// FACT-SAME: others=[[AOS]]@[[ST]]:[[#G1+1]]@{{[^ ]+}} element=4 lanes=0,2,4,6|1,3,5,7 loads=0:8
float aos(uniform float a[], uniform int u) {
    float r;
    unmasked { r = a[u * 8 + 2 * programIndex] + a[u * 8 + 2 * programIndex + 1]; }
    return r;
}
