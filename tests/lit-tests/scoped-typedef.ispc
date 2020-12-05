//; RUN: not %{ispc} %s 2>&1 | FileCheck %s -check-prefix=CHECK
//; CHECK: scoped-typedef.ispc:6:19

void a() { typedef int64 BigInt; }

void b() { BigInt a; }
