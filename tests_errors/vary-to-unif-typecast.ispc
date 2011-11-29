// Can't type cast from varying type "int32" to uniform type "uniform int32"

uniform int foo(int x)  {
    return (uniform int) x;
}

