// This test case check for proper functionality of the `--dump-file` option,
//   with and without the optional destination directory specified.

// RUN: %{ispc} %s --target=host --nostdlib -h %t.h --emit-llvm-text --dump-file=%t --debug-phase=220,300 -o /dev/null
// RUN: FileCheck %s --input-file %t/ir_220_DCEPass.ll -check-prefixes=CHECK_ALL,CHECK_PRE
// RUN: FileCheck %s --input-file %t/ir_300_LoopUnrollPass.ll -check-prefixes=CHECK_ALL,CHECK_POST

// RUN: %{ispc} %s --target=host --nostdlib -h %t.h --emit-llvm-text --dump-file --debug-phase=220,300 -o /dev/null
// RUN: FileCheck %s --input-file ir_220_DCEPass.ll -check-prefixes=CHECK_ALL,CHECK_PRE
// RUN: FileCheck %s --input-file ir_300_LoopUnrollPass.ll -check-prefixes=CHECK_ALL,CHECK_POST

// RUN: %{ispc} %s --target=host --nostdlib -h %t.h --emit-llvm-text --dump-file=./ --debug-phase=220,300 -o /dev/null
// RUN: FileCheck %s --input-file ./ir_220_DCEPass.ll -check-prefixes=CHECK_ALL,CHECK_PRE
// RUN: FileCheck %s --input-file ./ir_300_LoopUnrollPass.ll -check-prefixes=CHECK_ALL,CHECK_POST

#define NUM 1024

// CHECK_ALL-LABEL: @foo(

// CHECK_PRE:   getelementptr
// CHECK_PRE:   load

// CHECK_POST:  getelementptr
// CHECK_POST:  load
// CHECK_POST:  getelementptr
// CHECK_POST:  load
// CHECK_POST:  getelementptr
// CHECK_POST:  load
// CHECK_POST:  getelementptr
// CHECK_POST:  load

export void foo(uniform int * uniform _in, uniform int * uniform _out, uniform int k)
{
    uniform int array[NUM];
    #pragma unroll 4
    for (uniform int i = 0; i < NUM; i++) {
        array[i] = _in[i] * k;
    }
    _out[programIndex] = array[programIndex];
}
