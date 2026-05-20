// Test error handling for missing profile file.
// LLVM 23+ silently ignores missing profile files, so this test is only for older versions.
//
// REQUIRES: !LLVM_23_0+
// RUN: not %{ispc} --profile-sample-use=nonexistent_file.prof --target=host --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s

// CHECK: error: nonexistent_file.prof: Could not open profile

export void dummy() {}
