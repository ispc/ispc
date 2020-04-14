// RUN: %{ispc} --target=avx2-i32x4 %s > %t 2>&1
bool test(bool &cond, bool arg1, uniform bool *uniform arg2) { return cond ? arg1 : *arg2; }
