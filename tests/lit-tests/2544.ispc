// RUN: not %{ispc} --target=host --nowrap -o - %s 2>&1 | FileCheck %s

// CHECK: Error: syntax error
// CHECK-NOT: Please file a bug report
void sum(int x)
{
    print("sum: %", reduce_add(x:1:0));
}

