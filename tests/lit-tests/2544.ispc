// RUN: not %{ispc} --arch=x86-64 --target=avx2 --nowrap -o - %s 2>%t
// RUN: FileCheck --input-file=%t %s

// CHECK: Error: syntax error
// CHECK-NOT: Please file a bug report
void sum(int x)
{
    print("sum: %", reduce_add(x:1:0));
}

