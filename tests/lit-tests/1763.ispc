// RUN: %{ispc} %s --target=avx2-i64x4 --nowrap --opt=disable-handle-pseudo-memory-ops --emit-llvm-text -o - | FileCheck %s
// REQUIRES: X86_ENABLED
// CHECK: declare void @__pseudo_scatter_factored_base_offsets64_i8({{.*}} nocapture, <4 x i64>, i32, <4 x i64>, <4 x i8>, <4 x i64>)
extern uniform unsigned int64 var_1;
extern uniform bool var_11;
extern uniform unsigned int64 arr_253[2];
extern uniform bool arr_252[17][2];
extern uniform int16 arr_255[2];

void test() {
    if (-5174796161658191029LL)
        foreach (i_43 = 0LL...16LL)
            for (uniform bool i_52 = 0; i_52 < 1; i_52 += 1)
                arr_252[i_43][i_52] =
                    (uniform unsigned int)((var_11 ? 195 : 1153317864U) << (var_1 - 9494725476832750132ULL));
    for (uniform bool i_58 = 0; i_58 < 1; i_58 += 1)
        arr_255[i_58] = arr_253[i_58];
}
