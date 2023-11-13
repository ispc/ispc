// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Function can't have both "task" and "export" qualifiers 

export task void foo() {
}
