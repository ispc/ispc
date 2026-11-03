// #3983: a cast from integers to varying pointers disappears in the frontend.
// ImproveMemoryOps recovers origin as the base even though relative reaches
// destination. After the scatter becomes a masked store, LLVM incorrectly
// reuses the first load of destination. A GEP without inbounds still has this
// provenance problem. The second load must survive the store.
//
// The runtime oracle and passing configurations share the same kernel in
// address-provenance.inc. AVX2 executes the reproducer; the AVX-512 IR check
// needs no AVX-512 execution hardware.
// Remove XFAIL when the representation/lowering rewrite preserves the reload.

// RUN: %{ispc} %S/address-provenance.inc --target=avx512skx-x16 -O2 --nowrap --woff --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} --pic --target=avx2-i32x8 -O2 --woff -h %t.h %S/address-provenance.inc -o %t.o
// RUN: %{cc} -O2 -x c -c %S/address-provenance.inc --include %t.h -o %t.c.o
// RUN: %{cc} %t.o %t.c.o -o %t.bin
// RUN: %t.bin same_block
// RUN: %t.bin full_block
// Both runtime cases fail independently: same_block with n = 1..w active
// lanes and full_block, a full foreach iteration. The all-off same-block
// mask and the partial iterations are gated in address-provenance.ispc.
// Like 3985-1.ispc, the AVX2 execution assumes an AVX2-capable x86-64 host.
// REQUIRES: LINUX_HOST && X86_ENABLED && X86_64_HOST
// XFAIL: *

// CHECK-LABEL: define void @same_block(
// CHECK: load <16 x float>, ptr @destination
// CHECK-NOT: {{^}}}
// CHECK: load <16 x float>, ptr @destination
// CHECK: {{^}}}
