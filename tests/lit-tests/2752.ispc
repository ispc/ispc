// RUN: %{ispc} --target=host --nowrap --nostdlib -O2 --emit-llvm-text %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %{ispc} --target=host --nowrap --nostdlib -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s --check-prefix=ASM

// REQUIRES: X86_ENABLED && !ARM_ENABLED

// LLVM-LABEL: @set_ref(
// LLVM-NEXT: allocas:
// LLVM-NEXT: store i8 1, {{.*}} %result
// LLVM-NEXT: ret void

// ASM-LABEL: set_ref:
// ASM-NEXT: # %bb.0:
// ASM-NEXT:        mov     byte ptr [r{{.*}}], 1
// ASM-NEXT:        ret

export void set_ref(uniform bool &result) {
    result = true;
}

// LLVM-LABEL: @set_ptr(
// LLVM-NEXT: allocas:
// LLVM-NEXT: store i8 1, {{.*}} %result
// LLVM-NEXT: ret void

// ASM-LABEL: set_ptr:
// ASM-NEXT: # %bb.0:
// ASM-NEXT:        mov     byte ptr [r{{.*}}], 1
// ASM-NEXT:        ret

export void set_ptr(uniform bool * uniform result) {
    *result = true;
}
