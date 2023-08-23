// RUN: %{ispc} --target=host --nowrap -o %t.o %s

void foo(unsigned int32 x) {
    unsigned int32 key_1 = x;
    uniform unsigned int32 u_key_1;
    uniform bool is_u_key_1 = reduce_equal(key_1, &u_key_1);
}
