// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: No label named "label" found in current function.

void func(int x) {
completelydifferentlabel:
    goto label;
}

