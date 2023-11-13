// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Illegal to provide soa<4> qualifier with non-struct type "void"

soa<4> void foo() { }
