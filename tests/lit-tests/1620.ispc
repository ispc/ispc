// RUN: %{ispc} --target=sse4-i32x8 %s > %t 2>&1
// RUN: %{ispc} --target=sse4-i16x8 %s > %t 2>&1
// RUN: %{ispc} --target=sse4-i8x16 %s > %t 2>&1
// RUN: %{ispc} --target=sse4.1-i32x8 %s > %t 2>&1
// RUN: %{ispc} --target=sse4.1-i16x8 %s > %t 2>&1
// RUN: %{ispc} --target=sse4.1-i8x16 %s > %t 2>&1
// RUN: %{ispc} --target=avx1-i32x4 %s > %t 2>&1
// RUN: %{ispc} --target=avx2-i64x4 %s > %t 2>&1
// RUN: %{ispc} --target=avx512skx-x16 %s > %t 2>&1
// REQUIRES: X86_ENABLED
void test_crash()
{
#if TARGET_WIDTH == 4
    bool m = { 0, 1, 0, 1 };
#elif TARGET_WIDTH == 8
    bool m = { 0, 1, 0, 1, 0, 1, 1, 1 };
#elif TARGET_WIDTH == 16
    bool m = { 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1 };
#endif
    return ;
}
