// The dispatch function is compiled with APX disabled when the target it is
// compiled for (the lowest one) enables APX.

// RUN: %{ispc} %s -O0 --target=avx10.2nvl-x8,avx10.2dmr-x16 --emit-llvm-text -o %t.ll --nostdlib
// RUN: FileCheck --input-file=%t.ll %s
// RUN: %{ispc} %s -O2 --target=avx10.2nvl-x8,avx10.2dmr-x16 --emit-asm -o %t.s
// RUN: FileCheck --input-file=%t.s %s --check-prefix=ASM

// REQUIRES: X86_ENABLED && LLVM_22_0+ && !MACOS_HOST

// CHECK: define i32 @which() #[[ATTR:[0-9]+]]
// CHECK: attributes #[[ATTR]] = { {{.*}}"target-features"="-egpr,-ndd,-push2pop2,-ppx,-ccmp,-cf,-nf,-zu{{(,-jmpabs)?}}" }

// ASM:     which:
// ASM-NOT: %r{{(1[6-9]|2[0-9]|3[01])}}
// ASM-NOT: {nf}
// ASM-NOT: {{push2|pop2|pushp|popp|ccmp|ctest|jmpabs}}
// ASM:     {{\.cfi_endproc|\.seh_endproc}}

export uniform int which() { return 0; }
