// This test checks that dispatch function (in case of multi-target compilation) and application function (in case of single target compilation) have
// "dllexport" storage type when ISPC "--dllexport" switch is used.

// RUN: %{ispc} %s  --dllexport --target=sse2 --target-os=windows --emit-llvm-text -o - | FileCheck %s --check-prefixes=CHECK_DLLEXPORT,CHECK_DLLEXPORT_SINGLE_TARGET
// RUN: %{ispc} %s  --dllexport --target=sse2 --target-os=windows --emit-llvm-text -o - | FileCheck %s --check-prefix=CHECK_DLLEXPORT

// RUN: %{ispc} %s  --target=sse2 --target-os=windows --emit-llvm-text -o - | FileCheck %s --check-prefix=CHECK_NODLLEXPORT
// RUN: %{ispc} %s  --target=sse2 --target-os=windows --emit-llvm-text -o - | FileCheck %s --check-prefix=CHECK_NODLLEXPORT

// REQUIRES: WINDOWS_ENABLED

// CHECK_DLLEXPORT_SINGLE_TARGET: define dllexport float @foo_extern___
// CHECK_DLLEXPORT_SINGLE_TARGET: define dllexport float @foo_export___

// CHECK_DLLEXPORT: define dllexport float @foo_export

// CHECK_NODLLEXPORT-NOT: define dllexport

static uniform float foo_static(uniform float a, uniform float b) {
    return a + b;
}

uniform float foo_extern(uniform float a, uniform float b) {
    return a + b;
}

export uniform float foo_export(uniform float a, uniform float b) {
    return a + b;
}
