// Can't type cast from type "varying int32" to type "uniform int32"

uniform int foo(int x)  {
    return (uniform int) x;
}

