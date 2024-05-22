// RUN: not %{ispc} --nowrap --nostdlib --target=host %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Enum type "Ab" unknown. Did you mean "Aa"?

enum Aa { X = 1, };
enum Ab;
