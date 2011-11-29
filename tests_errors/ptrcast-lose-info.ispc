// Pointer type cast of type "int32 * uniform" to integer type "uniform int32" may lose information.

int32 foo(int * uniform x)  {
    return (int32) x;
}

