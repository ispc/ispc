// Invalid combinations of the rvv target, the zvl<N>b attribute and the device.

// RUN: not %{ispc} --target=rvv --attr=zvl100b --arch=riscv64 --target-os=linux --nowrap %s -o %t.o 2>&1 | FileCheck %s --check-prefix=BAD-WIDTH
// RUN: not %{ispc} --target=rvv --attr=zvlxb --arch=riscv64 --target-os=linux --nowrap %s -o %t.o 2>&1 | FileCheck %s --check-prefix=BAD-ATTR
// RUN: not %{ispc} --target=rvv --attr=zvl256b,zvl512b --arch=riscv64 --target-os=linux --nowrap %s -o %t.o 2>&1 | FileCheck %s --check-prefix=TWO
// RUN: not %{ispc} --target=rvv-256b --attr=zvl128b --arch=riscv64 --target-os=linux --nowrap %s -o %t.o 2>&1 | FileCheck %s --check-prefix=ALIAS
// RUN: not %{ispc} --target=rvv --attr=zvl512b --cpu=spacemit-x60 --arch=riscv64 --target-os=linux --nowrap %s -o %t.o 2>&1 | FileCheck %s --check-prefix=DEVICE
// RUN: not %{ispc} --target=rvv-512b --cpu=spacemit-x60 --arch=riscv64 --target-os=linux --nowrap %s -o %t.o 2>&1 | FileCheck %s --check-prefix=DEVICE
// RUN: not %{ispc} --target=rvv --attr=zvl4096b --arch=riscv64 --target-os=linux --nowrap %s -o %t.o 2>&1 | FileCheck %s --check-prefix=TOO-WIDE
// RUN: not %{ispc} --target=generic-i1x4 --attr=zvl256b --arch=riscv64 --target-os=linux --nowrap %s -o %t.o 2>&1 | FileCheck %s --check-prefix=OTHER

// REQUIRES: RISCV_ENABLED && LINUX_ENABLED

// BAD-WIDTH: Error: Invalid attribute "zvl100b": the vector register width must be a power of two of at least 128 bits.
// BAD-ATTR: Error: Invalid attribute "zvlxb".
// TWO: Error: Only one zvl<N>b attribute may be given.
// ALIAS: Error: Attribute zvl128b contradicts target rvv-256b.
// DEVICE: Error: Device "spacemit-x60" has 256-bit vector registers, but the target needs 512 bits.
// TOO-WIDE: Error: Vector registers wider than 2048 bits are not supported (attribute zvl4096b).
// OTHER: Error: Target attributes (--attr) are only supported for the rvv target.

export void inc(uniform int v[], uniform int count) {
    foreach (i = 0 ... count) {
        v[i] += 1;
    }
}
