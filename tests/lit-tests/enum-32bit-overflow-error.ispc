// RUN: not %{ispc} --nowrap --nostdlib --target=host %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Enumerator value must be representable with a 32-bit integer.

enum TooBig {
    TOO_BIG = 0x100000000
};
