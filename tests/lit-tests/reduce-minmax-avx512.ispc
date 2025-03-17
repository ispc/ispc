// Check that vphminposuw is produced for reduce_min/max on int8 and int16 types with avx512 targets.
// RUN: %{ispc} %s --target=avx512skx-x16 --nowrap --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx512skx-x32 --nowrap --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx512skx-x64 --nowrap --emit-asm -o - | FileCheck %s
// REQUIRES: X86_ENABLED && !MACOS_HOST

// CHECK-LABEL: foo8___vytvyt
// CHECK-COUNT-2: vphminposuw
uniform int8 foo8(int8 a, int8 b) {
    if (programIndex < 3)
        return reduce_min(a);
    return reduce_max(b);
}

// CHECK-LABEL: foo16___vysvys
// CHECK-COUNT-2: vphminposuw
uniform int16 foo16(int16 a, int16 b) {
    if (programIndex < 3)
        return reduce_min(a);
    return reduce_max(b);
}