// This test checks that the compiler supports `signed`, `unsigned` and `const`
// quialifiers in `new` expressions and template arguments. 
// It also check the compiler does not crash (fatal error).

// RUN: %{ispc} --target=host --emit-llvm-text --nostdlib --nowrap -g %s -o - 2>&1 | FileCheck %s

// CHECK-NOT: FATAL ERROR:

uniform uint* uniform test_uint() {
    return uniform new uniform unsigned int;
}

uniform int* uniform test_sint() {
    return uniform new uniform signed int; 
}

uniform const int* uniform test_const() {
    return uniform new uniform const int(815);
}

const unsigned uniform int* uniform test_mixed_unordered() {
    return uniform new unsigned const uniform int(815);
}

template <typename T> void test_template() {}

void test_template_instantiation()
{
    test_template<int>();
    test_template<uniform int>();
    test_template<varying int>();
    test_template<const int>();
    test_template<const signed int>();
    test_template<const unsigned int>();
    test_template<const unsigned uniform int>();
    test_template<uniform const unsigned int>();
}
