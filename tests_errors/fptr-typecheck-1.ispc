// Too many parameter values provided in function call

float bar(float a, float b);

export uniform int foo(uniform int x[], uniform int i[]) {
    float (*fptr)(float, float) = bar;
//CO    bar(0,1,2);
    fptr(0., 1, 2);
}
