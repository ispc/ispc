// rule: skip on cpu=TGLLP

export void print_uf(uniform float a) {
    uniform int intA = (int)a;
    uniform double uniDoubleVal  =  a + 13.25;

    print("Test uniform double: %\n", uniDoubleVal);
}

export void print_result() {
    print("Test uniform double: 18.250000\n");
}
