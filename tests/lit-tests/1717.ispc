// RUN: %{ispc} --target=avx2-i32x8 %s > %t 2>&1
static uniform int arr[256];

static void foo(uint32 r) { int *pEr = &arr[r]; }
