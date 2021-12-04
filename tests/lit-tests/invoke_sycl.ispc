// The test checks code generation and proper diagnostics for ISPC/DPC++ interop.

// RUN: %{ispc} %s --target=gen9-x16 --emit-llvm-text --nowrap -o %s.ll
// RUN: not %{ispc} %s --target=host --emit-llvm-text --nowrap -o %s.ll 2>&1 | FileCheck %s -check-prefix=CHECK_ERROR

// CHECK_ERROR: Error: 'extern "C"' function "extern_print" cannot be defined in ISPC
// REQUIRES: XE_ENABLED

extern "C" int extern_print(uniform float RET[], int b) {
  print("Hello world\n");
  return b*100;
}

