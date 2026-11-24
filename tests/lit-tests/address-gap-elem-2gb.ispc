// An element of 2^31 bytes or more doesn't fit the 32-bit scale of 32-bit
// addressing. The frontend used to truncate the size to an int32 and
// multiply at 32 bits with nsw: p - 1 multiplied -2^31 by -1, which is
// poison, although its offset -2^31 fits in 32 bits. The index is now
// extended and multiplied at 64 bits. On a 32-bit target, where the address
// wraps at 2^32 anyway, the 32-bit product has no nsw. With
// --wrap-signed-int the 32-bit offset wraps for elements of any size, so two
// elements of 2^31 bytes are as far as four of 2^30 bytes: no offset.

// RUN: %{ispc} %s --target=avx2-i32x8 --arch=x86-64 --nostdlib --nowrap -O2 --emit-llvm-text --debug-phase=10:10 -o - | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %{ispc} %s --target=avx2-i32x8 --arch=x86 --nostdlib --nowrap -O2 --emit-llvm-text --debug-phase=10:10 -o - | FileCheck %s --check-prefixes=CHECK,X86
// RUN: %{ispc} %s --target=avx2-i32x8 --arch=x86-64 --nostdlib --nowrap -O2 --wrap-signed-int --emit-llvm-text -o - | FileCheck %s --check-prefix=WRAP
// RUN: %{ispc} %s --target=avx2-i32x8 --arch=x86-64 --nostdlib --nowrap -O2 --emit-llvm-text -o - | FileCheck %s --check-prefix=NOWRAP

// The 32-bit x86 run needs a host OS that supports it, as in 3985-4.ispc.
// REQUIRES: X86_ENABLED && !MACOS_HOST

struct Big {
    int8 a[0x40000000];
    int8 b[0x40000000];
};

// CHECK-LABEL: define {{.*}}@big_offset
// X64: [[T:%[^ ]+]] = trunc <8 x i64> %{{[^ ]+}} to <8 x i32>
// X64-NEXT: [[I:%[^ ]+]] = sext <8 x i32> [[T]] to <8 x i64>
// X64-NEXT: mul nsw <8 x i64> splat (i64 2147483648), [[I]]{{$}}
// X86: mul <8 x i32> splat (i32 -2147483648)
// CHECK-NOT: mul nsw <8 x i32> splat (i32 -2147483648)
varying int64 big_offset(uniform Big *uniform p, uniform int k) {
    varying int idx = k;
    uniform Big *varying q = p + idx;
    return (varying int64)q - (uniform int64)p;
}

// The difference of two such pointers is divided by the full size, at 64
// bits: (p - 1) - p is -1, not -2^31 / INT32_MIN = 1. With 32-bit pointers
// one element forward and one back are the same difference in bytes, which
// no division gets right for both.
// CHECK-LABEL: define {{.*}}@big_diff
// X64: sdiv <8 x i64> %{{.*}}, splat (i64 2147483648)
// X64: trunc <8 x i64> %{{.*}} to <8 x i32>
varying int big_diff(uniform Big *uniform p, uniform int k) {
    varying int idx = k;
    uniform Big *varying q = p + idx;
    uniform Big *varying r = p;
    return q - r;
}

struct Half {
    int8 a[0x40000000];
};

// WRAP-LABEL: define {{.*}}@wrap_same
// WRAP-NEXT: allocas:
// WRAP-NEXT: ret <8 x i32> splat (i32 -1)
varying bool wrap_same(uniform int8 *uniform base, uniform unsigned int k) {
    varying unsigned int i = k;
    uniform Big *varying p = (uniform Big * uniform) base + 2 * i;
    uniform Half *varying q = (uniform Half * uniform) base + 4 * i;
    return (uniform int8 * varying) p == (uniform int8 * varying) q;
}

// A component of a short vector in an soa<> struct, v.y, is a constant
// element offset, and v[1] is an index: both are 2^31 bytes after v.x. With
// --wrap-signed-int the 32-bit offset of the index wraps, and so does the
// constant offset, so the two always address the same memory.
struct V2 {
    int64<2> v;
};

// WRAP-LABEL: define {{.*}}soa_component
// WRAP: add <8 x i64> %p.fca.0.extract, splat (i64 -2147483648)
// NOWRAP-LABEL: define {{.*}}soa_component
// NOWRAP: add <8 x i64> %p.fca.0.extract, splat (i64 2147483648)
void soa_component(soa<268435456> V2 *varying p) { p->v.y = 1; }

// WRAP-LABEL: define {{.*}}soa_index
// WRAP: add <8 x i64> %p.fca.0.extract, splat (i64 -2147483648)
// NOWRAP-LABEL: define {{.*}}soa_index
// NOWRAP: add <8 x i64> %p.fca.0.extract, splat (i64 2147483648)
void soa_index(soa<268435456> V2 *varying p) { p->v[1] = 1; }
