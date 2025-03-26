// This test checks that the alignment attribute is correctly applied to type
// definitions and can be applied to the particular variable declarations
// overriding the type definition.

// RUN: %{ispc} --target=host --emit-llvm-text -o - %s 2>&1 | FileCheck %s

// CHECK: @x = local_unnamed_addr global %v{{.*}}_varying_S zeroinitializer
struct S x;

struct Sy { int8 x; int y; } __attribute__((aligned(8)));

// CHECK: @y8 = local_unnamed_addr global %v{{.*}}_varying_Sy zeroinitializer, align 8
struct Sy y8;

// CHECK: @y32 = local_unnamed_addr global %v{{.*}}_varying_Sy zeroinitializer, align 32
__attribute__((aligned(32))) struct Sy y32;

// CHECK: @y8_2 = local_unnamed_addr global %v{{.*}}_varying_Sy zeroinitializer, align 8
struct Sy y8_2;

// CHECK: @z8 = local_unnamed_addr global %v{{.*}}_varying_Sz zeroinitializer, align 8
__attribute__((aligned(8))) struct Sz { int8 x; int y; } z8;

// CHECK: @z0 = local_unnamed_addr global %v{{.*}}_varying_Sz zeroinitializer
struct Sz z0;
