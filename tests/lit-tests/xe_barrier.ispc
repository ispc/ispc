// This test checks correct LLVM IR generation when memory_barrier is used on Xe target.

// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 --emit-llvm-text --nowrap -o - | FileCheck %s
// REQUIRES: XE_ENABLED

task void barrier() {
// CHECK: @llvm.genx.barrier()
  memory_barrier();
}