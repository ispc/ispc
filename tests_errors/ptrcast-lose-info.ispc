// Pointer type cast of type "uniform int32 \* uniform" to integer type "uniform int32" may lose information.
//  rule: skip on arch=*
//  rule: run on arch=x86-64
//  rule: run on arch=aarch64

int32 foo(int * uniform x)  {
    return (int32) x;
}

