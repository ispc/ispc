export void print_uf(uniform float a) {
    uniform int intA = (int)a;
    uniform float  uniFloatVal   =  a + 4.75;

    print("Test uniform float: %\n", uniFloatVal);
}

export void print_result() {
    print("Test uniform float: 9.750000\n");
}
