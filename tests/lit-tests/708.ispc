// This test checks that generated code does not try to divide by zero.

// RUN: %{ispc} --pic --target=host %s -o %t.o
// RUN: %{cc} %t.o -o %t.c.bin
// RUN: %t.c.bin | FileCheck %s

// REQUIRES: !MACOS_HOST

// CHECK-NOT: Floating point exception

export void test(uniform int nb)
{
    foreach(i = 0 ... nb)
    {
        // foreach (j = nb+1 ... 1)
        float j = nb + 1 - i;
        float k;
        if (j > 0)
            k = nb / j;
        else
            k = -1;
        print("%\n", k);
    }
    foreach(i = 0 ... nb)
    {
        // foreach (j = nb+1 ... 1)
        int j = nb + 1 - i;
        int k;
        if (j > 0)
            k = nb / j;
        else
            k = -1;
        print("%\n", k);
    }
}

extern "C" uniform int main() {
    test(12);
    return 0;
}
