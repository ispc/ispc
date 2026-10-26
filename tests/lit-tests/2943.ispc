// Pins today's code for discussion #2943: interleaved RGBA8 pixels, 4
// contiguous bytes per lane (stride 4). Today every byte access is a gather
// or scatter: 8 gather and 4 scatter warnings on every x86 target, and the
// bytes are inserted and extracted one lane at a time.
// Ideal: the block is contiguous: vector loads, a masked store, no warnings.

// RUN: %{ispc} %s --target=sse4-i32x4 --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefix=IR
// RUN: %{ispc} %s --target=avx2-i32x8 --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefix=IR
// RUN: %{ispc} %s --target=avx512skx-x16 --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefix=IR
// RUN: %{ispc} %s --target=sse4-i32x4 --nostdlib --nowrap -o /dev/null 2> %t.sse4
// RUN: FileCheck %s --input-file=%t.sse4 --check-prefixes=WARN
// RUN: FileCheck %s --input-file=%t.sse4 --check-prefix=COUNT12
// RUN: %{ispc} %s --target=avx2-i32x8 --nostdlib --nowrap -o /dev/null 2> %t.avx2
// RUN: FileCheck %s --input-file=%t.avx2 --check-prefixes=WARN
// RUN: FileCheck %s --input-file=%t.avx2 --check-prefix=COUNT12
// RUN: %{ispc} %s --target=avx512skx-x16 --nostdlib --nowrap -o /dev/null 2> %t.avx512skx
// RUN: FileCheck %s --input-file=%t.avx512skx --check-prefixes=WARN
// RUN: FileCheck %s --input-file=%t.avx512skx --check-prefix=COUNT12

// REQUIRES: X86_ENABLED

// The warning checks match in any order; the COUNT prefix checks that
// there are no other warnings.
// COUNT12-COUNT-12: Performance Warning
// COUNT12-NOT: Performance Warning

// IR-LABEL: @difference_rgba32(
// IR-NOT: load <{{[0-9]+}} x i8>
// IR: insertelement <{{[0-9]+}} x i8>
// IR-NOT: load <{{[0-9]+}} x i8>

export void difference_rgba32(const uniform uint8 source[], const uniform uint8 destination[],
                              uniform uint8 output[], uniform uint32 count) {
    foreach (index = 0 ... count) {
        // WARN-DAG: 2943.ispc:[[#@LINE+12]]:13: Performance Warning: Gather required to load value.
        // WARN-DAG: 2943.ispc:[[#@LINE+11]]:38: Performance Warning: Gather required to load value.
        // WARN-DAG: 2943.ispc:[[#@LINE+11]]:13: Performance Warning: Gather required to load value.
        // WARN-DAG: 2943.ispc:[[#@LINE+10]]:38: Performance Warning: Gather required to load value.
        // WARN-DAG: 2943.ispc:[[#@LINE+10]]:13: Performance Warning: Gather required to load value.
        // WARN-DAG: 2943.ispc:[[#@LINE+9]]:38: Performance Warning: Gather required to load value.
        // WARN-DAG: 2943.ispc:[[#@LINE+9]]:13: Performance Warning: Gather required to load value.
        // WARN-DAG: 2943.ispc:[[#@LINE+8]]:38: Performance Warning: Gather required to load value.
        // WARN-DAG: 2943.ispc:[[#@LINE+8]]:13: Performance Warning: Scatter required to store value.
        // WARN-DAG: 2943.ispc:[[#@LINE+8]]:13: Performance Warning: Scatter required to store value.
        // WARN-DAG: 2943.ispc:[[#@LINE+8]]:13: Performance Warning: Scatter required to store value.
        // WARN-DAG: 2943.ispc:[[#@LINE+8]]:13: Performance Warning: Scatter required to store value.
        if (source[index * 4 + 0] != destination[index * 4 + 0] ||
            source[index * 4 + 1] != destination[index * 4 + 1] ||
            source[index * 4 + 2] != destination[index * 4 + 2] ||
            source[index * 4 + 3] != destination[index * 4 + 3]) {
            output[index * 4 + 0] = 255;
            output[index * 4 + 1] = 0;
            output[index * 4 + 2] = 255;
            output[index * 4 + 3] = 255;
        }
    }
}
