//; RUN: not %{ispc} %s --target=sse2-i32x4 2>&1 | FileCheck %s -check-prefix=CHECK_4
//; RUN: not %{ispc} %s --target=avx1-i32x8 2>&1 | FileCheck %s -check-prefix=CHECK_8
//; CHECK_4: varying_init_error.ispc:13:14
//; CHECK_4: varying_init_error.ispc:16:23
//; CHECK_4: varying_init_error.ispc:18:14
//; CHECK_8: varying_init_error.ispc:21:14
//; CHECK_8: varying_init_error.ispc:24:35
//; CHECK_8: varying_init_error.ispc:26:14

export void foo() {
#if TARGET_WIDTH == 4
    // size check fail. Too many elements.
    int a = {1, 1, 1, 1, 1, 1, 1};
    varying int b_ = 3;
    // typecheck fail.
    int b = {0, 1, 1, b_};
    // size check fail. Too few elements.
    int c = {1, 1, 1};
#elif TARGET_WIDTH == 8
    // size check fail. Too many elements.
    int a = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    varying int b_ = 3;
    // typecheck fail.
    int b = {0, 1, 1, 1, 1, 1, 1, b_};
    // size check fail. Too few elements.
    int c = {1, 1, 1};
#endif
}
