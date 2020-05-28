// RUN: %{ispc} --target=avx2-i32x16 %s -o %t.o
// REQUIRES: X86_ENABLED
extern uniform bool arr_4[17];
export void test(uniform int64 var_1, uniform bool var_15) {
    foreach (i_0 = var_1... 16U)
        arr_4[i_0] = !(varying bool)reduce_min(180) ? 5923697687793782713ULL : (varying bool)var_15;
}
