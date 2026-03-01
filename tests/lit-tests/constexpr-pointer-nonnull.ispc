// RUN: %{ispc} %s --target=host --nostdlib --emit-llvm-text -o - | FileCheck %s

uniform int g0 = 1;

constexpr uniform int *uniform getp() { return &g0; }

constexpr uniform int *uniform PG = getp();
constexpr uniform int *uniform PG2 = PG + 2;

constexpr uniform int is_nonnull(uniform int *uniform p) {
    if (p) {
        return 1;
    }
    return 0;
}

constexpr uniform int PFlag = is_nonnull(PG);

uniform int use() { return PFlag; }

// CHECK-DAG: @PG = {{.*}} ptr @g0
// CHECK-DAG: @PG2 = {{.*}} getelementptr {{.*}} @g0
// CHECK-DAG: @PFlag = {{.*}} i32 1
