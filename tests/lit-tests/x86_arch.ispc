//; RUN: %{ispc} --arch=x86-64 --target=avx2-i32x8 %s
//; RUN: %{ispc} --arch=x86_64 --target=avx2-i32x8 %s
//; REQUIRES: X86_ENABLED
float foo(float b) {
    return b;
}
