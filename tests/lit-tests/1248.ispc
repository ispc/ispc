// RUN: not %{ispc} --target=host --nostdlib --nowrap -o - --emit-llvm-text %s 2>&1 | FileCheck %s

// CHECK-NOT: FATAL ERROR:
struct Quaternion
{
    float x;
    float y;
    float z;
    float w;
};

export void QuaternionTestSOA(soa<32> Quaternion a[], uniform unsigned int count)
{
    foreach (i = 0 ... count)
    {
        a[i].x += 1.0;
    }
}
