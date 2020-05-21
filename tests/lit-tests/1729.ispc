// RUN: %{ispc} -O2 --target=host %s > %t 2>&1
void a(int8 b, int64 c[][2]) {
     c[programIndex][0] = extract((int)(varying bool)reduce_max(1) * b, 0);
}
