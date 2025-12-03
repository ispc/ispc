// This test checks that ISPC generates the correct code for selects with bool
// condition for targets with i8 masks.

// RUN: %{ispc} --target=sse4.1-i8x16 --emit-llvm-text -O2 -o - %s 2>&1 | FileCheck %s
// RUN: %{ispc} --target=sse4.2-i8x16 --emit-llvm-text -O2 -o - %s 2>&1 | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK-LABEL: define <16 x i8> @embedded_select___unb_3C_3_3E_unT_3C_3_3E_unT_3C_3_3E_(
// CHECK-NEXT: allocas:
// CHECK-NEXT:   [[COND_STORAGE:%.*]] = zext <16 x i1> %cond to <16 x i8>
// CHECK-NEXT:   [[SHUFFLE4:%.*]] = shufflevector <16 x i8> [[COND_STORAGE]], <16 x i8> {{.*}}, <3 x i32> {{.*}}
// CHECK-NEXT:   [[SHUFFLE3:%.*]] = shufflevector <3 x i8> [[SHUFFLE4]], <3 x i8> {{.*}}, <4 x i32> {{.*}}
// CHECK-NEXT:   [[SHUFFLE2:%.*]] = shufflevector <4 x i8> [[SHUFFLE3]], <4 x i8> {{.*}}, <8 x i32> {{.*}}
// CHECK-NEXT:   [[SHUFFLE1:%.*]] = shufflevector <8 x i8> [[SHUFFLE2]], <8 x i8> {{.*}}, <16 x i32> {{.*}}
// CHECK-NEXT:   [[CMP:%.*]] = icmp eq <16 x i8> [[SHUFFLE1]], zeroinitializer
// CHECK-NEXT:   [[SELECT:%.*]] = select <16 x i1> [[CMP]], <16 x i8> %f, <16 x i8> %t
// CHECK-NEXT:   [[RESULT:%.*]] = shufflevector <16 x i8> [[SELECT]], <16 x i8> {{.*}}, <16 x i32> {{.*}}
// CHECK-NEXT:   ret <16 x i8> [[RESULT]]
// CHECK-NEXT: }

#define N 3

uniform uint8<N> embedded_select(uniform bool<N> cond, uniform uint8<N> t, uniform uint8<N> f) {
    uniform uint8<N> result;

    foreach (i = 0 ... N) {
        result[i] = select(cond[i], t[i], f[i]);
    }

    return result;
}
