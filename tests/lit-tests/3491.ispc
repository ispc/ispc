// This test checks that ISPC generates the correct code for selects with bool
// condition for targets with i8 masks.

// RUN: %{ispc} --target=sse4.1-i8x16 --emit-llvm-text -O2 -o - %s 2>&1 | FileCheck %s
// RUN: %{ispc} --target=sse4.2-i8x16 --emit-llvm-text -O2 -o - %s 2>&1 | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK-LABEL: define <16 x i8> @embedded_select___unb_3C_3_3E_unT_3C_3_3E_unT_3C_3_3E_(
// CHECK-NEXT: allocas:
// CHECK:   {{%.*}} = {{shufflevector <16 x i1>|zext <16 x i1>}}
// CHECK:   {{%.*}} = {{zext <3 x i1>|shufflevector <16 x i8>}}
// CHECK:   [[CMP:%.*]] = icmp eq <16 x i8> {{%.*}}, zeroinitializer
// CHECK-NEXT:   [[SELECT:%.*]] = select <16 x i1> [[CMP]], <16 x i8> %f, <16 x i8> %t
// CHECK:   ret <16 x i8>

#define N 3

uniform uint8<N> embedded_select(uniform bool<N> cond, uniform uint8<N> t, uniform uint8<N> f) {
    uniform uint8<N> result;

    foreach (i = 0 ... N) {
        result[i] = select(cond[i], t[i], f[i]);
    }

    return result;
}
