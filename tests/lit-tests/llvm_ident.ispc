// RUN: %{ispc} %s --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --emit-llvm-text --nostdlib -o - | FileCheck %s

// Check that version string embeded in !llvm.ident metadata is formed correctly.

// CHECK: Intel(r) Implicit SPMD Program Compiler (Intel(r) ISPC), {{1\.[[:digit:]]+\.[[:digit:]]+(dev)?}} (build{{( commit )?}}{{([[:xdigit:]]+)?}} @ {{([[:digit:]]{8})}}, LLVM [[LLVM_VERSION:[[:digit:]]+\.[[:digit:]]+\.[[:digit:]]+]])
// CHECK: LLVM version [[LLVM_VERSION]]{{(\( [[:graph:]]+ [[:graph:]]+\))?}}

int i;
