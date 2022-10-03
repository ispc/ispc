// The test checks that GPU device/target definitions (including all synonyms) are successfully consumed by compiler.

// Gen9
// RUN: %{ispc} %s -o %t.spv --nostdlib --target=gen9-x8 --device=skl
// RUN: %{ispc} %s -o %t.spv --nostdlib --target=gen9-x8 --device=tgllp
// RUN: %{ispc} %s -o %t.spv --nostdlib --target=gen9-x8 --device=dg1
// RUN: %{ispc} %s -o %t.spv --nostdlib --target=gen9-x16 --device=skl
// RUN: %{ispc} %s -o %t.spv --nostdlib --target=gen9-x16 --device=tgllp
// RUN: %{ispc} %s -o %t.spv --nostdlib --target=gen9-x16 --device=dg1

// XeLP
// RUN: not %{ispc} %s -o %t.spv --nostdlib --target=xelp-x8 --device=skl
// RUN: %{ispc} %s -o %t.spv --nostdlib --target=xelp-x8 --device=tgllp
// RUN: %{ispc} %s -o %t.spv --nostdlib --target=xelp-x8 --device=dg1
// RUN: not %{ispc} %s -o %t.spv --nostdlib --target=xelp-x16 --device=skl
// RUN: %{ispc} %s -o %t.spv --nostdlib --target=xelp-x16 --device=tgllp
// RUN: %{ispc} %s -o %t.spv --nostdlib --target=xelp-x16 --device=dg1

// XeHPG
// RUN: not %{ispc} %s -o %t.spv --nostdlib --target=xehpg-x8 --device=skl
// RUN: not %{ispc} %s -o %t.spv --nostdlib --target=xehpg-x8 --device=tgllp
// RUN: not %{ispc} %s -o %t.spv --nostdlib --target=xehpg-x8 --device=dg1
// RUN: %{ispc} %s -o %t.spv --nostdlib --target=xehpg-x8 --device=dg2
// RUN: not %{ispc} %s -o %t.spv --nostdlib --target=xehpg-x16 --device=skl
// RUN: not %{ispc} %s -o %t.spv --nostdlib --target=xehpg-x16 --device=tgllp
// RUN: not %{ispc} %s -o %t.spv --nostdlib --target=xehpg-x16 --device=dg1
// RUN: %{ispc} %s -o %t.spv --nostdlib --target=xehpg-x16 --device=dg2

// REQUIRES: XE_ENABLED
// REQUIRES: LINUX_HOST

uniform int i;

task void foo() {}
