// This test checks that the compiler reports errors when accessing `col.x`
// and assigning `reduce_col.w` and that this does not cause any crash 
// (fatal error).

// RUN: not %{ispc} --target=host --emit-llvm-text --nowrap -g %s -o - 2>&1 | FileCheck %s

// CHECK-COUNT-4: Error: Vector element identifier "w" is invalid for type
// CHECK-NOT: FATAL ERROR:

export void test(uniform int seq_len, uniform int tile_size) {
    varying float<3> col = {0, 0, 0};
    uniform float <3>* uniform reduce_col;
    reduce_col->x = reduce_add(col.x);
    reduce_col->y = reduce_add(col.y);
    reduce_col->z = reduce_add(col.z);
    reduce_col->w = reduce_add(col.w);
}

export void test2() {
    varying float<3> col = {0, 0, 0};
    col.w;
}

export void test3() {
    varying float<3> col = {0, 0, 0};
    sizeof(col.w);
}
