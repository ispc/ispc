// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "test_static.isph"

struct Row {
    uniform int<4> lanes;
    uniform uint8 counter;
};

noinline constexpr uniform int digest(uniform int seed) {
    uniform Row table[2] = {{{seed, 2, 3, 4}, 1}, {{5, 6, 7, seed}, 255}};
    uniform Row copy = table[1];
    table[1].lanes.y += table[0].lanes.x++;
    uniform int before = table[1].lanes.w--;
    uniform int after = ++table[1].lanes.z;
    uniform uint8 count = ++table[1].counter;
    uniform int i = 0;
    table[i++].lanes[0] = table[i].lanes.y;
    // Return both the copied and updated values. The RHS of the compound
    // assignment observes i after the destination subscript is evaluated.
    table[i--].lanes[0] += (table[i].lanes[1] = seed);
    return copy.lanes.y * 10000000 + copy.counter * 10000 + before * 1000 + after * 100 + count * 10 +
           table[0].lanes[0] + table[1].lanes[0] + i;
}

constexpr uniform int first = digest(2);
constexpr uniform int second = digest(3);

noinline constexpr uniform bool indexed_updates(uniform int x) {
    uniform int a[3] = {x, x + 10, 0};
    uniform int i = 0;
    uniform int old = a[i++]++;
    uniform int next = ++a[i++];
    uniform int read_after_write = a[(a[2] = x + 20, 2)];
    return old == x && next == x + 11 && a[0] == x + 1 && i == 2 && read_after_write == x + 20;
}

constexpr uniform bool indexed_first = indexed_updates(2);
constexpr uniform bool indexed_second = indexed_updates(3);

noinline uniform int advance_reference(uniform int &value) { return value++; }

task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform bool positive = aFOO[0] > 0.f;
    uniform int seed = positive ? 2 : 3;
    uniform bool equal = digest(seed) == (positive ? first : second) &&
                         digest(5 - seed) == (positive ? second : first);
    uniform int from_ref = advance_reference(seed);
    uniform bool ref_ok = from_ref + 1 == seed;
    uniform bool indexed_ok = indexed_first && indexed_second && indexed_updates(from_ref);

    uniform int a[2 * programCount] = {0};
    for (uniform int n = 0; n < 2 * programCount; ++n) {
        a[n] = n;
    }
    bool active = (programIndex & 1) == 0;
    int i = 2 * programIndex;
    int old = -1;
    if (active) {
        old = a[i++]++;
    }
    bool masked_ok = old == (active ? 2 * programIndex : -1) &&
                     a[2 * programIndex] == 2 * programIndex + (int)active &&
                     i == 2 * programIndex + (int)active;
    RET[programIndex] = equal && indexed_ok && ref_ok && masked_ok ? 1.f : 0.f;
}

task void result(uniform float RET[]) { RET[programIndex] = 1.f; }
