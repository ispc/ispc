// Calls to the width-suffixed pdep/pext functions resolve to a unique overload
// without warnings for signed, unsigned, narrower, and wider operands, for
// untyped, u-suffixed, and 64-bit literals, and for mixed uniform/varying
// operands.
// RUN: %{ispc} %s --target=avx2-i32x8 --werror --nowrap --emit-llvm-text -o - | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK: define {{.*}} @uniform_args___
// CHECK: define {{.*}} @varying_args___

uniform uint64 uniform_args(uniform uint32 u32, uniform int32 i32, uniform int8 i8, uniform uint64 u64,
                            uniform int64 i64) {
    uniform uint64 r = 0;
    r += pdep32(u32, 0x55555555);
    r += pdep32(u32, 0x55555555u);
    r += pdep32(i32, 0x55555555);
    r += pdep32(i32, u32);
    r += pdep32(i8, i8);
    r += pdep32(u64, u32);
    r += pext32(5, 0x52);
    r += pext32(u32, i8);
    r += pdep64(u64, 0x5555555555555555);
    r += pdep64(u64, 0x5555555555555555ull);
    r += pdep64(i64, 0x5555555555555555);
    r += pdep64(u64, u32);
    r += pdep64(i32, i8);
    r += pext64(i8, i8);
    r += pext64(i64, u64);
    r += pext64(1, 0x8000000000000000);
    return r;
}

uint64 varying_args(uint32 u32, int32 i32, int8 i8, uint64 u64, int64 i64, uniform uint32 uu32, uniform uint64 uu64) {
    uint64 r = 0;
    r += pdep32(u32, 0x55555555);
    r += pdep32(i32, 0x55555555u);
    r += pdep32(i8, i8);
    r += pdep32(u32, uu32);
    r += pdep32(uu32, i32);
    r += pext32(u64, uu32);
    r += pext32(programIndex, 0xFF);
    r += pdep64(u64, 0x5555555555555555);
    r += pdep64(i64, uu64);
    r += pdep64(uu64, i8);
    r += pext64(i32, 0x5555555555555555ull);
    r += pext64(uu32, u64);
    return r;
}
