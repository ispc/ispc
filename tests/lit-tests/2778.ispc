// Pins today's code for #2778: copying an 8-byte struct of byte fields per
// lane is a gather and a scatter of single bytes, while the same copy through
// an int64 pointer is one 8-byte access per lane.
// Today: TestStruct and TestStructIdx load and store every byte separately
// (with a gather and a scatter warning each); TestInt64 is one vector load
// and store; TestInt64Idx uses 8-byte accesses per lane (also with a gather
// and a scatter warning).
// Ideal: TestStruct = TestInt64 and TestStructIdx = TestInt64Idx, with no
// warnings for the contiguous copy.

// RUN: %{ispc} %s --target=sse4-i32x4 --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefix=IR
// RUN: %{ispc} %s --target=avx2-i32x8 --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefix=IR
// RUN: %{ispc} %s --target=avx512skx-x16 --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefix=IR
// RUN: %{ispc} %s --target=sse4-i32x4 --nostdlib --nowrap -o /dev/null 2> %t.sse4
// RUN: FileCheck %s --input-file=%t.sse4 --check-prefixes=WARN
// RUN: FileCheck %s --input-file=%t.sse4 --check-prefix=COUNT6
// RUN: %{ispc} %s --target=avx2-i32x8 --nostdlib --nowrap -o /dev/null 2> %t.avx2
// RUN: FileCheck %s --input-file=%t.avx2 --check-prefixes=WARN
// RUN: FileCheck %s --input-file=%t.avx2 --check-prefix=COUNT6
// RUN: %{ispc} %s --target=avx512skx-x16 --nostdlib --nowrap -o /dev/null 2> %t.avx512skx
// RUN: FileCheck %s --input-file=%t.avx512skx --check-prefixes=WARN
// RUN: FileCheck %s --input-file=%t.avx512skx --check-prefix=COUNT6

// REQUIRES: X86_ENABLED

// The warning checks match in any order; the COUNT prefix checks that
// there are no other warnings.
// COUNT6-COUNT-6: Performance Warning
// COUNT6-NOT: Performance Warning

struct Foo { uint8 a, b, c, d, e, f, g, h; };

// IR-LABEL: @TestStruct___
// IR-NOT: x i64>
// IR: load i8
// IR-NOT: x i64>
// IR: store i8
// IR-NOT: x i64>
unmasked void TestStruct(uniform Foo Dest[], uniform Foo Src[])
{
    // WARN-DAG: 2778.ispc:[[#@LINE+2]]:5: Performance Warning: Scatter required to store value.
    // WARN-DAG: 2778.ispc:[[#@LINE+1]]:26: Performance Warning: Gather required to load value.
    Dest[programIndex] = Src[programIndex];
}

// IR-LABEL: @TestInt64___
// IR: load <[[N:[0-9]+]] x i64>
// IR: store <[[N]] x i64>
unmasked void TestInt64(uniform Foo Dest[], uniform Foo Src[])
{
    ((uniform int64 * uniform)Dest)[programIndex] = ((uniform int64 * uniform)Src)[programIndex];
}

// Non-contiguous variant from the issue comment.
// IR-LABEL: @TestStructIdx___
// IR-NOT: {{load|store}} i64
// IR-NOT: x i64>
// IR: load i8
// IR-NOT: {{load|store}} i64
// IR-NOT: x i64>
// IR: store i8
// IR-NOT: {{load|store}} i64
// IR-NOT: x i64>
unmasked void TestStructIdx(uniform Foo Dest[], uniform Foo Src[], varying int i)
{
    // WARN-DAG: 2778.ispc:[[#@LINE+2]]:5: Performance Warning: Scatter required to store value.
    // WARN-DAG: 2778.ispc:[[#@LINE+1]]:15: Performance Warning: Gather required to load value.
    Dest[i] = Src[i];
}

// IR-LABEL: @TestInt64Idx___
// IR: {{load i64|call <[0-9]+ x i64> @llvm.x86}}
unmasked void TestInt64Idx(uniform Foo Dest[], uniform Foo Src[], varying int i)
{
    // WARN-DAG: 2778.ispc:[[#@LINE+2]]:5: Performance Warning: Scatter required to store value.
    // WARN-DAG: 2778.ispc:[[#@LINE+1]]:42: Performance Warning: Gather required to load value.
    ((uniform int64 * uniform)Dest)[i] = ((uniform int64 * uniform)Src)[i];
}
