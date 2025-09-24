// Test to verify sample profile metadata is correctly loaded and embedded
//
// Test without profile - should have no profile metadata:
// RUN: %{ispc} -O2 --target=host --nostdlib --nowrap --emit-llvm-text -o - %s | FileCheck %s --check-prefix=CHECK_NO_PROFILE
//
// Test with profile - should have complete profile metadata:
// RUN: %{ispc} -O2 --profile-sample-use=%S/test_profile_metadata.prof --target=host --nostdlib --nowrap --emit-llvm-text -o - %s | FileCheck %s --check-prefix=CHECK_WITH_PROFILE
//
// Test error handling:
// RUN: not %{ispc} --profile-sample-use=nonexistent_file.prof --target=host --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s --check-prefix=CHECK_ERROR

// WITHOUT PROFILE - should have no profile metadata
// CHECK_NO_PROFILE-NOT: ProfileSummary
// CHECK_NO_PROFILE-NOT: SampleProfile

// WITH PROFILE - should have complete profile metadata
// CHECK_WITH_PROFILE: !{{[0-9]+}} = !{i32 1, !"ProfileSummary", !{{[0-9]+}}}
// CHECK_WITH_PROFILE: !{{[0-9]+}} = !{!"ProfileFormat", !"SampleProfile"}
// CHECK_WITH_PROFILE: !{{[0-9]+}} = !{!"TotalCount", i64 {{[0-9]+}}}
// CHECK_WITH_PROFILE: !{{[0-9]+}} = !{!"MaxCount", i64 {{[0-9]+}}}
// CHECK_WITH_PROFILE: !{{[0-9]+}} = !{!"MaxFunctionCount", i64 {{[0-9]+}}}
// CHECK_WITH_PROFILE: !{{[0-9]+}} = !{!"NumCounts", i64 {{[0-9]+}}}
// CHECK_WITH_PROFILE: !{{[0-9]+}} = !{!"NumFunctions", i64 3}
// CHECK_WITH_PROFILE: !{{[0-9]+}} = !{!"IsPartialProfile", i64 0}
// CHECK_WITH_PROFILE: !{{[0-9]+}} = !{!"DetailedSummary", !{{[0-9]+}}}

// ERROR HANDLING - should report file not found
// CHECK_ERROR: error: nonexistent_file.prof: Could not open profile

static uniform float hot_helper(uniform float x) {
    return x * 2.0f + 1.0f;
}

static uniform float cold_helper(uniform float x) {
    uniform float result = x;
    for (uniform int i = 0; i < 5; i++) {
        result = result * 0.9f + 0.1f;
    }
    return result;
}

export void profile_metadata_test(uniform float input[], uniform float output[], uniform int count) {
    for (uniform int i = 0; i < count; i++) {
        uniform float value = input[i];

        // Hot path - 90% according to profile
        if (value > 0.5f) {
            output[i] = hot_helper(value);
        }
        // Cold path - 10% according to profile
        else {
            output[i] = cold_helper(value);
        }
    }
}
