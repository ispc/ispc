// Test for --sample-profiling-debug-info flag functionality
//
// RUN: %{ispc} --target=host --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s --check-prefix=CHECK_NO_DEBUG
// RUN: %{ispc} --sample-profiling-debug-info --target=host --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s --check-prefix=CHECK_SAMPLE_PROF
// RUN: %{ispc} --sample-profiling-debug-info --dwarf-version=5 --target=host --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s --check-prefix=CHECK_SAMPLE_PROF_DWARF5
// RUN: %{ispc} -g --target=host --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s --check-prefix=CHECK_FULL_DEBUG

// CHECK_NO_DEBUG-NOT: llvm.dbg.cu
// CHECK_NO_DEBUG-NOT: DICompileUnit

// CHECK_SAMPLE_PROF: DICompileUnit
// CHECK_SAMPLE_PROF: emissionKind: LineTablesOnly
// CHECK_SAMPLE_PROF: debugInfoForProfiling: true
// CHECK_SAMPLE_PROF-NOT: DILocalVariable
// CHECK_SAMPLE_PROF: DISubprogram
// CHECK_SAMPLE_PROF: DISPFlagDefinition
// CHECK_SAMPLE_PROF-NOT: retainedNodes

// CHECK_SAMPLE_PROF_DWARF5: DICompileUnit
// CHECK_SAMPLE_PROF_DWARF5: emissionKind: LineTablesOnly
// CHECK_SAMPLE_PROF_DWARF5: debugInfoForProfiling: true
// CHECK_SAMPLE_PROF_DWARF5-NOT: DILocalVariable
// CHECK_SAMPLE_PROF_DWARF5: DISubprogram
// CHECK_SAMPLE_PROF_DWARF5: DISPFlagDefinition
// CHECK_SAMPLE_PROF_DWARF5-NOT: retainedNodes

// CHECK_FULL_DEBUG: DICompileUnit
// CHECK_FULL_DEBUG: emissionKind: FullDebug
// CHECK_FULL_DEBUG-NOT: debugInfoForProfiling: true
// CHECK_FULL_DEBUG: DILocalVariable
// CHECK_FULL_DEBUG: DISubprogram
// CHECK_FULL_DEBUG: DISPFlagDefinition
// CHECK_FULL_DEBUG: retainedNodes

static uniform float internal_helper(uniform float value) {
    uniform float result = value * 2.0f;
    if (result > 10.0f) {
        result = result / 2.0f;
    }
    return result;
}

export void process_array(uniform float input[], uniform float output[], uniform int count) {
    for (uniform int i = 0; i < count; i++) {
        uniform float temp = internal_helper(input[i]);
        if (temp > 5.0f) {
            output[i] = temp * 1.5f;
        } else {
            output[i] = temp + 1.0f;
        }
    }
}