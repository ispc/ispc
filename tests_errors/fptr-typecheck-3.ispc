// Too few parameter values provided in function call \(1 provided, 2 expected\).

float bar(float a, float b);

export uniform int foo(uniform int x[], uniform int i[]) {
    float (*fptr)(float, float) = bar;
    fptr(1.);
}
