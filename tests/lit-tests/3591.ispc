// Test for issue #3591: Internal error casting soa<> pointer to opaque pointer
// This test verifies that casting from an SOA pointer to a non-SOA pointer type
// produces a proper error message instead of crashing with "Illegal BitCast".
//
// RUN: not %{ispc} %s --target=host --nowrap -o %t.o 2>&1 | FileCheck %s

// CHECK: Error: Can't cast from pointer to SOA type
// CHECK-NOT: FATAL ERROR
struct point {
    int x, y;
};

struct foo;

__attribute__((external_only))
export foo* uniform foo_init() {
    soa<8> point* uniform points = uniform new soa<8> point[123];
    return (foo* uniform)points;
}
