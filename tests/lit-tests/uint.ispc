// RUN: %{ispc} %s > %t 2>&1
// RUN: FileCheck --input-file=%t %s
// CHECK-NOT: Error
#ifndef ISPC_UINT_IS_DEFINED
typedef unsigned int64 uint64;
#endif
uint64 a;
