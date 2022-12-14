// Tests multiple instantiations of templates with same function type (no typenames as parameters or return type).
// This verifies temlate name mangling.

// RUN: %{ispc}  %s --emit-llvm-text --target=host --nostdlib -o - | FileCheck %s

// CHECK: define {{.*}} @addTemp___vyi___vyfvyf
// CHECK: define {{.*}} @addTemp___vyI___vyfvyf
template <typename T> noinline int addTemp(float targ0, float targ1) { return (int)((T)targ0 + (T)targ1); }

bool foo(float arg0, float arg1) {
    int v1 = addTemp<int>(arg0, arg1);
    int v2 = addTemp<int64>(arg0, arg1);
    if (v1 == v2)
        return true;
    return false;
}
