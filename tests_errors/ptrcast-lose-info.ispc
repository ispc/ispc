// Pointer type cast of type "uniform int32 \* uniform" to integer type "uniform int32" may lose information.
//  rule: run on arch=x86-64

int32 foo(int * uniform x)  {
    return (int32) x;
}

