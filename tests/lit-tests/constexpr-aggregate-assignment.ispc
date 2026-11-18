// RUN: %{ispc} %s --target=host --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s

struct Row {
    uniform int<4> lanes;
    uniform uint8 tag;
};

struct Table {
    uniform Row rows[4];
};

constexpr uniform Table rotate(uniform int shift) {
    uniform Table table = {{{{0}, 0}}};
    for (uniform int row = 0; row < 4; ++row) {
        for (uniform int lane = 0; lane < 4; ++lane) {
            table.rows[row].lanes[lane] = (lane + row + shift) & 3;
        }
        table.rows[row].tag = (uniform uint8)row;
    }
    uniform Table copy = table;
    copy.rows[1] = table.rows[3];
    copy.rows[1].lanes.y += 10;
    copy.rows[1].lanes.z--;
    ++copy.rows[1].lanes.w;
    copy.rows[1].tag += (uniform uint8)4;
    // Neither the original table nor other rows of the copy share writes.
    return copy;
}

constexpr uniform Table rotations = rotate(1);
constexpr uniform int untouched = rotations.rows[3].lanes.y;
constexpr uniform int replaced = rotations.rows[1].lanes.y * 100 + rotations.rows[1].lanes.z * 10 +
                                rotations.rows[1].lanes.w;
constexpr uniform int updated_tag = rotations.rows[1].tag;

// CHECK-DAG: @untouched = {{.*}} i32 1
// CHECK-DAG: @replaced = {{.*}} i32 1114
// CHECK-DAG: @updated_tag = {{.*}} i32 7

constexpr uniform int assign_order() {
    uniform int a[3] = {10, 20, 30};
    uniform int i = 0;
    a[i++] = a[i];
    a[i++] += (a[1] = a[i]);
    return a[0] * 1000 + a[1] * 10 + i;
}

constexpr uniform int ordered = assign_order();

// CHECK-DAG: @ordered = {{.*}} i32 20602

constexpr uniform int indexed_updates() {
    uniform int a[3] = {10, 20, 30};
    uniform int i = 0;
    uniform int old = a[i++]++;
    uniform int next = ++a[i++];
    uniform int read_after_write = a[(a[2] = 8, 2)];
    return old * 100000 + next * 1000 + a[0] * 100 + i * 10 + read_after_write;
}

constexpr uniform int once = indexed_updates();

// CHECK-DAG: @once = {{.*}} i32 1022128
