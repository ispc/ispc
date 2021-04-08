int sum_varying() { // should return 6
    int ia[] = {1, 2, 3};
    varying int *ptr = &ia[0];
    int sum = 0;
    for (uniform int i = 0; i < 3; i++)
        sum += ptr[i];
    return sum;
}

int sum_uniform() { // should return 60
    uniform int ia[] = {10, 20, 30};
    uniform int *ptr = &ia[0];
    int sum = 0;
    for (uniform int i = 0; i < 3; i++)
        sum += ptr[i];
    return sum;
}

export void f_v(uniform float RET[]) {
    RET[programIndex] = sum_varying() + sum_uniform(); // should be 66
}

export void result(uniform float RET[]) { RET[programIndex] = 66; }
