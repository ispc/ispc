// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Arrays of "void" type are illegal

uniform void aFOO[] = { NULL };

