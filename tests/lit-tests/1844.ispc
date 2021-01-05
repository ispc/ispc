// RUN: %{ispc} %s -o %t.o --arch=x86-64 --target=avx512skx-i8x64
// RUN: %{ispc} %s -o %t.o --arch=x86-64 --target=avx512skx-i16x32
// RUN: %{ispc} %s -o %t.o --arch=x86-64 --target=avx512skx-i32x16
// RUN: %{ispc} %s -o %t.o --arch=x86-64 --target=avx2-i8x32
// RUN: %{ispc} %s -o %t.o --target=host

// REQUIRES: X86_ENABLED

extern uniform unsigned int64 var_6;
extern uniform unsigned int16 var_11;
extern uniform bool arr_201 [1] [15] [20];
extern uniform unsigned int8 arr_202 [1] [15] [20];
extern uniform int64 arr_210 [1] [15] [20] [26] [11];

export void test() {
    foreach(i_53 = var_6...0L)
        for (uniform unsigned int16 i_54 = 0L; i_54 < 14L; i_54 += 4L)
            for (uniform unsigned int8 i_55 = 0; i_55 < 19; i_55 += 2)
                for (uniform unsigned int8 i_57 = 0L; i_57 < 25L; i_57 += 4)
                    for (uniform unsigned int8 i_58 = 0; i_58 < extract(((varying int32)(varying unsigned int16)var_11) ^ arr_201 [i_53] [i_54] [i_55], 0U) - 120L; i_58 += 4L)
                        if (arr_201 [i_53] [i_54] [i_55])
                            arr_210 [i_53] [i_54] [i_55] [i_57] [i_58] = -arr_202 [i_53] [i_54] [i_55];
}
