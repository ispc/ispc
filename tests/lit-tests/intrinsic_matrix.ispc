//; RUN: %{ispc} %s --target=sse4-i32x4 --enable-llvm-intrinsics
//; RUN: %{ispc} %s --target=avx2-i32x8 --enable-llvm-intrinsics
//; RUN: %{ispc} %s --target=avx512skx-x16 --enable-llvm-intrinsics

// REQUIRES: X86_ENABLED

 void f_f(uniform float RET[], uniform float aFOO[]) {

    RET[programIndex] = 0;
    float matrix = aFOO[programIndex];
    uniform float t_matriz[programCount];
#if TARGET_WIDTH == 4
    const uniform int row = 2;
    const uniform int column = 2;
    float transpose = @llvm.matrix.transpose.v8f32.i32.i32(matrix, row, column);
#elif TARGET_WIDTH == 8
    const uniform int row = 2;
    const uniform int column = 4;
    float transpose = @llvm.matrix.transpose.v8f32.i32.i32(matrix, row, column);
#elif TARGET_WIDTH == 16
    const uniform int row = 4;
    const uniform int column = 4;
    float transpose = @llvm.matrix.transpose.v8f32.i32.i32(matrix, row, column);
#endif
    t_matriz[programIndex] = transpose;

    for (uniform int r = 0; r < row; r++) {
        for (uniform int c = 0; c < column; c++) {
            if (t_matriz[r * row + c] != aFOO[c * row + r]) {
                RET[1] = 1;
            }
        }
    }

}

