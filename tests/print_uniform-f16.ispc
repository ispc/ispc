// rule: skip on arch=x86
// rule: skip on arch=x86-64
export void print_uf(uniform float a) {
    uniform float16 uniFloatVal = (float16)a + 4.75f16;
    print("Test uniform float: %\n", uniFloatVal);
}

export void print_result() { print("Test uniform float: 9.750000\n"); }
