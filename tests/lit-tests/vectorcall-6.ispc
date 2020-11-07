// This test confirms ispc emits an error when '__vectorcall' is used on 'non-extern' functions.
//; RUN: not %{ispc} %s -o %t.o --nowrap --vectorcall --target=host --target-os=windows 2>&1 | FileCheck %s

// REQUIRES: X86_ENABLED

// REQUIRES: WINDOWS_ENABLED

//; CHECK: Error: Illegal to use "__vectorcall" qualifier on non-extern function "CopyFloat".

__vectorcall uniform float CopyFloat(const uniform float Source) {
    return Source;
}

export void CopyFloat_ispc(uniform float Result[], const uniform float Source[], const uniform int Iterations) {
    for(uniform int i = 0; i < Iterations; i++) {
        Result[i] = CopyFloat(Source[i]);
    }
}
