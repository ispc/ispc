// The test checks that the triple contain minimum OS version if provided.

// RUN: %{ispc} %s --nostdlib --target-os=macos --target=avx2 --arch=x86-64 --emit-llvm-text --nowrap -o - | FileCheck %s --check-prefix=CHECK-MACOS-DEFAULT
// RUN: %{ispc} %s --nostdlib --target-os=macos --target=avx2 --arch=x86-64 --emit-llvm-text --nowrap --darwin-version-min=15.0 -o - | FileCheck %s --check-prefix=CHECK-MACOS-VER
// RUN: %{ispc} %s --nostdlib --target-os=macos --target=avx2 --arch=x86-64 --emit-llvm-text --nowrap --darwin-version-min="" -o - | FileCheck %s --check-prefix=CHECK-MACOS-VER-UNSET
// RUN: not %{ispc} %s --nostdlib --target-os=macos --arch=x86-64 --nowrap --target=avx2 --darwin-version-min=a.b -o - 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR-VER

// RUN: %{ispc} %s --nostdlib --target-os=macos --target=host -h %t.h -o %t.o
// RUN: %{cc} -x c -c %s --include %t.h -o %t.cpp.o
// RUN: %{cxx} %t.o %t.cpp.o -o %t.bin -mmacosx-version-min=13.0 2>&1 | FileCheck %s --check-prefix=CHECK-WARNING --allow-empty
// RUN: otool -l %t.bin | FileCheck %s --check-prefix=CHECK-MINOS

// REQUIRES: MACOS_HOST && X86_ENABLED

// CHECK-MACOS-DEFAULT: target triple = "x86_64-apple-macosx10.12"
// CHECK-MACOS-VER: target triple = "x86_64-apple-macosx15.0"
// CHECK-MACOS-VER-UNSET: target triple = "x86_64-apple-macosx"

// CHECK-ERROR-VER: Error: Invalid version format: "a.b". Use <major_ver.minor_ver>.

// CHECK-WARNING-NOT: no platform load command found

// CHECK-MINOS: minos 13.0

#ifdef ISPC
export void set(uniform int &result) {
    result = programCount;
}

#else
int main(int argc, char **argv) {
    int r = 0;
    set(&r);
    return 0;
}
#endif
