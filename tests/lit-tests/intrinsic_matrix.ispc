//; RUN: %{ispc} %s --target=host --enable-intrinsic-call

// REQUIRES: X86_ENABLED

 void f_f(uniform float RET[], uniform float aFOO[]) {

    RET[programIndex] = 0;
    float matrix = aFOO[programIndex];
    uniform float t_matriz[programCount];
    uniform int row = 0;
    uniform int column = 0;
    float transpose = aFOO[programIndex];
#if TARGET_WIDTH == 4
    row = 2;
    column = 2;
    transpose = @llvm.matrix.transpose.v8f32.i32.i32(matrix, 2, 2);
#elif TARGET_WIDTH == 8
    row = 2;
    column = 4;
    transpose = @llvm.matrix.transpose.v8f32.i32.i32(matrix, 2, 4);
#elif TARGET_WIDTH == 16
    row = 4;
    column = 4;
    transpose = @llvm.matrix.transpose.v8f32.i32.i32(matrix, 4, 4);
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

