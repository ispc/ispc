// RUN: %{ispc} --target=host %s > %t 2>&1
void test(uniform float &Val) { Val = Val * -(Val * Val); }
