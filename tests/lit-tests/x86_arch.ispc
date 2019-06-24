//; RUN: %{ispc} --arch=x86-64 %s
//; RUN: %{ispc} --arch=x86_64 %s
float foo(float b) {
    return b;
}
