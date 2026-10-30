// Pins today's code for #2901 and #2899: a masked varying int8 store through
// a reference.
// Today: on sse4-i32x4 and avx2-i32x8, SetByte1 stores one byte per lane
// (extract and store), and no performance warning is printed (#2899).
// avx512skx-x16 uses a masked byte store. SetByte2 (explicit blend and
// unmasked store) is one load, a select and one vector store everywhere.
// Ideal: SetByte1 = SetByte2 (blend and store), or at least a warning.

// RUN: %{ispc} %s --target=sse4-i32x4 --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefixes=CHECK,BYTES
// RUN: %{ispc} %s --target=avx2-i32x8 --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefixes=CHECK,BYTES
// RUN: %{ispc} %s --target=avx512skx-x16 --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefixes=CHECK,SKX
// RUN: %{ispc} %s --target=sse4-i32x4 --nostdlib --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefix=NOWARN --allow-empty
// RUN: %{ispc} %s --target=avx2-i32x8 --nostdlib --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefix=NOWARN --allow-empty
// RUN: %{ispc} %s --target=avx512skx-x16 --nostdlib --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefix=NOWARN --allow-empty

// REQUIRES: X86_ENABLED

// NOWARN-NOT: Performance Warning

// CHECK-LABEL: @SetByte1___
// BYTES-NOT: store <{{[0-9]+}} x i8>
// BYTES: extractelement <{{[0-9]+}} x i8>
// BYTES-NOT: store <{{[0-9]+}} x i8>
// BYTES: store i8
// BYTES-NOT: store <{{[0-9]+}} x i8>
// SKX: call void @llvm.masked.store.v16i8.p0(
void SetByte1(int8& dest, int8 src)
{
    dest = src;
}

// CHECK-LABEL: @SetByte2___
// CHECK: load <[[N:[0-9]+]] x i8>
// CHECK: select <[[N]] x i1>
// CHECK: store <[[N]] x i8>
void SetByte2(int8& dest, int8 src) {
    varying bool mask = __mask;
    unmasked {
        dest = mask ? src : dest;
    }
}
