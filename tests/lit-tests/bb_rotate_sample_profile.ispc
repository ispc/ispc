// This test verifies that sample profile data generates branch weights for basic block reordering

// RUN: %{ispc} %s --target=host -O2 --nostdlib --emit-llvm-text --sample-profiling-debug-info --profile-sample-use=%S/bb_rotate_sample_profile.prof -o - | FileCheck %s --check-prefix=CHECK-PROFILE
// RUN: %{ispc} %s --target=host -O2 --nostdlib --emit-llvm-text --sample-profiling-debug-info -o - | FileCheck %s --check-prefix=CHECK-NO-PROFILE
// RUN: %{ispc} %s --target=host -O2 --nostdlib --emit-asm --sample-profiling-debug-info --profile-sample-use=%S/bb_rotate_sample_profile.prof -o - | FileCheck %s --check-prefix=CHECK-ASM-PROFILE
// RUN: %{ispc} %s --target=host -O2 --nostdlib --emit-asm --sample-profiling-debug-info -o - | FileCheck %s --check-prefix=CHECK-ASM-NO-PROFILE

// REQUIRES: LINUX_HOST

noinline static uniform float cold_path_computation(uniform float x) {
    return x * x + 10.0f;
}

noinline static uniform float hot_path_computation(uniform float x) {
    return x + 1.0f;
}

export void block_reorder_demo(uniform float input[], uniform float output[], uniform int count) {
    for (uniform int i = 0; i < count; i++) {
        uniform float value = input[i];

        // First branch decision point
        if (value > 0.5f) {
            // Hot path according to profile (90%)
            goto hot_branch;
        } else {
            goto cold_branch;
        }

    hot_branch:
        // Nested decision in hot path
        if (value > 2.0f) {
            // Very hot path (80% of total)
            output[i] = hot_path_computation(value);
            goto exit;
        } else {
            // Moderately hot path (10% of total)
            output[i] = value * 3.0f;
            goto exit;
        }

    cold_branch:
        // Cold path processing
        if (value < -1.0f) {
            // Very cold path (5% of total)
            output[i] = cold_path_computation(value);
            goto exit;
        } else {
            // Cold path (5% of total)
            output[i] = value * 0.5f;
            goto exit;
        }

    exit:
        continue;
    }
}

// Check that profile generates function entry count and branch weights
// CHECK-PROFILE: define void @block_reorder_demo
// CHECK-PROFILE-DAG: !{!"function_entry_count", i64 {{[0-9]+}}}
// CHECK-PROFILE-DAG: !{!"branch_weights", i32 {{[0-9]+}}, i32 {{[0-9]+}}}

// Check that without profile, no metadata is generated
// CHECK-NO-PROFILE: define void @block_reorder_demo
// CHECK-NO-PROFILE-NOT: function_entry_count
// CHECK-NO-PROFILE-NOT: branch_weights

// CHECK-ASM-NO-PROFILE-NOT: .section{{.*}}.text.hot.

// Check that with profile, function is placed in hot section
// CHECK-ASM-PROFILE: .section{{.*}}.text.hot.
// CHECK-ASM-PROFILE: block_reorder_demo:
