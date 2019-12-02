//; RUN: cat %s | %{ispc} - --target=avx2-i32x8
//; RUN: cat %s | %{ispc} - --nocpp --target=avx2-i32x8
//; RUN: cat %s | not %{ispc} - -g --nowrap --target=avx2-i32x8 2>&1 | FileCheck %s -check-prefix=CHECK_ERROR

//; CHECK_ERROR: Error: Can't emit debugging information with no source file on disk.
int i;
