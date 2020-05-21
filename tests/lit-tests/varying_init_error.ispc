//; RUN: not %{ispc} %s --target=sse2-i32x4 2>&1 | FileCheck %s -check-prefix=CHECK_4
//; RUN: not %{ispc} %s --target=avx1-i32x8 2>&1 | FileCheck %s -check-prefix=CHECK_8
//; CHECK_4: varying_init_error.ispc:15:14
//; CHECK_4: varying_init_error.ispc:18:23
//; CHECK_4: varying_init_error.ispc:20:14
//; CHECK_8: varying_init_error.ispc:23:14
//; CHECK_8: varying_init_error.ispc:26:35
//; CHECK_8: varying_init_error.ispc:28:14

//; REQUIRES: X86_ENABLED

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
