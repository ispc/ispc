export void print_uf(uniform float a) {
    uniform int intA = (uniform int)a;
    uniform          int64  uniLongVal    = -(uniform int64)intA - 10000000000;
    uniform unsigned int64  uniULongVal   =  (uniform int64)intA + 10000000000;

    print("Test uniform int64: %, %\n", uniLongVal, uniULongVal);
}

export void print_result() {
    print("Test uniform int64: -10000000005, 10000000005\n");
}
