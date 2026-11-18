// RUN: %{ispc} %s %{generic_i32x4} --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s
// REQUIRES: GENERIC_I32X4

constexpr varying int Lane = programIndex;
constexpr varying int Lane2 = Lane * 2 + 1;

varying int foo() { return Lane2; }

// CHECK: ret <4 x i32> <i32 1, i32 3, i32 5, i32 7>

constexpr varying uint16 NegativeLane = -(varying uint16)(programIndex + 1);

varying uint16 negative_lane() { return NegativeLane; }

// CHECK: ret <4 x i16> <i16 -1, i16 -2, i16 -3, i16 -4>
