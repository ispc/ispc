// RUN: not %{ispc} %s --target=host --nostdlib --nowrap -o /dev/null 2>&1 | FileCheck %s

struct Aggregate {
    uniform int a[2];
};

uniform Aggregate global = {{0}};

constexpr uniform int write_global() {
    global.a[0] = 1;
    return 0;
}

constexpr uniform int increment_static() {
    static uniform Aggregate local = {{0}};
    return ++local.a[0];
}

constexpr uniform int write_pointer(uniform Aggregate * uniform p) {
    p->a[0] = 1;
    return 0;
}

constexpr uniform int read_uninitialized() {
    uniform Aggregate local;
    local.a[0] = 1;
    return local.a[1];
}

struct ReadOnlyElement {
    const uniform int value;
};

constexpr uniform int write_const_member() {
    uniform ReadOnlyElement local = {0};
    local.value = 1;
    return local.value;
}

// CHECK-DAG: Error: constexpr function "write_global" contains disallowed expression "assignment".
// CHECK-DAG: Error: constexpr function "increment_static" contains disallowed expression "increment/decrement".
// CHECK-DAG: Error: constexpr function "write_pointer" contains disallowed expression "assignment".
// CHECK-DAG: Error: constexpr function "read_uninitialized" requires initializer for local variable "local".
// CHECK-DAG: Error: Can't assign to type "const uniform int32" on left-hand side of expression.
