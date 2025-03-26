// This test checks that we emit warnings for misplaced attributes on struct.

// RUN: %{ispc} --target=host --nowrap %s --emit-llvm-text -o - 2>&1 | FileCheck %s

// CHECK: Warning: attribute ignored in declaration of 'struct S'. It must follow the 'struct' keyword.
__attribute__((aligned(8))) struct S { int8 x; int y; };

// CHECK: Warning: attribute ignored in declaration of 'struct S2'. It must follow the 'struct' keyword.
__attribute__((aligned(8))) struct S2;

// We forbid placing attributes before the struct keyword to avoid confusion
// for the following syntax. Here 32 is alignment for type but 16 for the variable.
// CHECK: @glob1 = local_unnamed_addr global %v{{.*}}_varying_S3 zeroinitializer, align 16
__attribute__((aligned(16))) struct S3 { int8 x; int y; } __attribute__((aligned(32))) glob1;
// CHECK: @glob2 = local_unnamed_addr global %v{{.*}}_varying_S3 zeroinitializer, align 32
struct S3 glob2;
