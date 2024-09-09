// This test checks that __memory_alignment is correctly set based on the user
// provided command line option --force-alignment
// __memory_alignment value is defined in the core.isph file without the initialization
// when ISPC compiles stdlib bitcode files during ISPC build. Later, this value
// is initialized with the value provided by the user via macro definition in
// ISPC_MEMORY_ALIGNMENT_VAL defined in module.cpp file (where preprocessor is
// initialized). This value is then used in the new functions to allocate memory.

// RUN: %{ispc} %s --target=host --force-alignment=42 2>&1 --debug-phase=2:2 -o t.o | FileCheck %s --check-prefix=CHECK-CONSTANT
// RUN: %{ispc} %s --target=host --target-os=linux --force-alignment=1024 2>&1 --emit-llvm-text -o - | FileCheck %s

// REQUIRES: LINUX_ENABLED

// CHECK-CONSTANT: @__memory_alignment = constant i32 42

// CHECK: call {{.*}} @posix_memalign({{.*}}, {{.*}} 1024, {{.*}}

uniform int* foo() {
    return uniform new int[10];
}
