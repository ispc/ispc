// The test checks that cpu definitions (including all synonyms) are successfully consumed by compiler.

//; RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --cpu=cortex-a9
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --cpu=cortex-a15
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --cpu=cortex-a35
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --cpu=cortex-a53
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --cpu=cortex-a57
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --cpu=apple-a7
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --cpu=apple-a10
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --cpu=apple-a11
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --cpu=apple-a12
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --cpu=apple-a13
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --cpu=apple-a14

// ARM must be enabled in order to test it.
// REQUIRES: ARM_ENABLED
// There are a few exceptions though.
// We do not support ARM targets on Windows.
// REQUIRES: !WINDOWS_HOST
// Not every ISPC build on macOS supports ARM targets.
// They are supported only if macOS SDK supports them.
// REQUIRES: !MACOS_HOST || MACOS_ARM_ENABLED

uniform int i;

void foo() {}
