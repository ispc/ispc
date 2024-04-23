//; RUN: %{ispc} %s --nostdlib --target=avx2-i32x4 --target-os=windows --emit-llvm-text -o - | FileCheck %s

// REQUIRES: X86_ENABLED && WINDOWS_ENABLED

template<typename TYPE>
TYPE bar() {
    return 42;
}

// CHECK-COUNT-2: ; Function Attrs: {{.*}} uwtable
export uniform int foo() {
    return bar<uniform int>();
}