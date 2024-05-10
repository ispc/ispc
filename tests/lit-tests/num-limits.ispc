// RUN: %{ispc} -DISPC --pic --target=host -h %t.h %s -o %t.o
// RUN: %{cc} -x c -c %s -o %t.c.o --include %t.h
// RUN: %{cc} %t.o %t.c.o -o %t.c.bin
// RUN: %t.c.bin | FileCheck %s

// REQUIRES: !MACOS_HOST

// CHECK: int8_min: 1
// CHECK: int8_max: 1
// CHECK: uint8_max: 1
// CHECK: int16_min: 1
// CHECK: int16_max: 1
// CHECK: uint16_max: 1
// CHECK: int32_min: 1
// CHECK: int32_max: 1
// CHECK: uint32_max: 1
// CHECK: int64_min: 1
// CHECK: int64_max: 1
// CHECK: uint64_max: 1
// CHECK: float_min: 1
// CHECK: float_max: 1
// CHECK: double_min: 1
// CHECK: double_max: 1

#ifdef ISPC
export uniform int8 int8_min() { return INT8_MIN; }
export uniform int8 int8_max() { return INT8_MAX; }
export uniform uint8 uint8_max() { return UINT8_MAX; }
export uniform int16 int16_min() { return INT16_MIN; }
export uniform int16 int16_max() { return INT16_MAX; }
export uniform uint16 uint16_max() { return UINT16_MAX; }
export uniform int32 int32_min() { return INT32_MIN; }
export uniform int32 int32_max() { return INT32_MAX; }
export uniform uint32 uint32_max() { return UINT32_MAX; }
export uniform int64 int64_min() { return INT64_MIN; }
export uniform int64 int64_max() { return INT64_MAX; }
export uniform uint64 uint64_max() { return UINT64_MAX; }
export uniform float float_min() { return FLT_MIN; }
export uniform float float_max() { return FLT_MAX; }
export uniform double double_min() { return DBL_MIN; }
export uniform double double_max() { return DBL_MAX; }
#else
#include <stdint.h>
#include <float.h>
#include <stdio.h>
int main(int argc, char **argv) {
    printf("int8_min: %i\n", int8_min() == INT8_MIN);
    printf("int8_max: %i\n", int8_max() == INT8_MAX);
    printf("uint8_max: %i\n", uint8_max() == UINT8_MAX);
    printf("int16_min: %i\n", int16_min() == INT16_MIN);
    printf("int16_max: %i\n", int16_max() == INT16_MAX);
    printf("uint16_max: %i\n", uint16_max() == UINT16_MAX);
    printf("int32_min: %i\n", int32_min() == INT32_MIN);
    printf("int32_max: %i\n", int32_max() == INT32_MAX);
    printf("uint32_max: %i\n", uint32_max() == UINT32_MAX);
    printf("int64_min: %i\n", int64_min() == INT64_MIN);
    printf("int64_max: %i\n", int64_max() == INT64_MAX);
    printf("uint64_max: %i\n", uint64_max() == UINT64_MAX);
    printf("float_min: %i\n", float_min() == FLT_MIN);
    printf("float_max: %i\n", float_max() == FLT_MAX);
    printf("double_min: %i\n", double_min() == DBL_MIN);
    printf("double_max: %i\n", double_max() == DBL_MAX);
    return 0;
}
#endif // ISPC
