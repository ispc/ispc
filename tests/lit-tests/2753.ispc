// RUN: %{ispc} --target=host --nowrap --nostdlib -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s --check-prefix=O2
// RUN: %{ispc} --target=host --nowrap --nostdlib -O0 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s --check-prefix=O0

// REQUIRES: X86_ENABLED && !ARM_ENABLED

// O2-LABEL: set:
// O2-NEXT:# %bb.0:
// O2-NEXT:        mov     al, 1
// O2-NEXT:        ret

// O0-LABEL: set:
// O0:        mov     byte ptr [rsp {{[-+]}} {{[0-9]+}}], 1
// O0:        mov     al, byte ptr [rsp {{[-+]}}  {{[0-9]+}}]
// O0:        ret
export uniform bool set() {
    return true;
}
