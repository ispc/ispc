// The test checks that GPU device/target definitions (including all synonyms) are successfully consumed by compiler.

// Gen9
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=gen9-x8 --device=skl
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=gen9-x8 --device=tgllp
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=gen9-x8 --device=dg1
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=gen9-x8 --device=acm-g10
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=gen9-x8 --device=acm-g11
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=gen9-x8 --device=acm-g12
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=gen9-x16 --device=skl
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=gen9-x16 --device=tgllp
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=gen9-x16 --device=dg1
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=gen9-x16 --device=acm-g10
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=gen9-x16 --device=acm-g11
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=gen9-x16 --device=acm-g12

// XeLP
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelp-x8 --device=skl
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelp-x8 --device=tgllp
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelp-x8 --device=dg1
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelp-x8 --device=acm-g10
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelp-x8 --device=acm-g11
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelp-x8 --device=acm-g12
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelp-x16 --device=skl
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelp-x16 --device=tgllp
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelp-x16 --device=dg1
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelp-x16 --device=acm-g10
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelp-x16 --device=acm-g11
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelp-x16 --device=acm-g12

// XeHPG
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xehpg-x8 --device=skl
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xehpg-x8 --device=tgllp
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xehpg-x8 --device=dg1
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xehpg-x8 --device=acm-g10
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xehpg-x8 --device=acm-g11
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xehpg-x8 --device=acm-g12
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xehpg-x16 --device=skl
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xehpg-x16 --device=tgllp
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xehpg-x16 --device=dg1
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xehpg-x16 --device=acm-g10
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xehpg-x16 --device=acm-g11
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xehpg-x16 --device=acm-g12

// XeHPC
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xehpc-x16 --device=skl
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xehpc-x16 --device=tgllp
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xehpc-x16 --device=dg1
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xehpc-x16 --device=acm-g10
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xehpc-x16 --device=acm-g11
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xehpc-x16 --device=acm-g12
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xehpc-x16 --device=pvc

// XeLPG
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelpg-x8 --device=skl
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelpg-x8 --device=tgllp
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelpg-x8 --device=dg1
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelpg-x8 --device=acm-g10
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelpg-x8 --device=acm-g11
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelpg-x8 --device=acm-g12
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelpg-x8 --device=mtl-m
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelpg-x8 --device=mtl-p
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelpg-x16 --device=skl
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelpg-x16 --device=tgllp
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelpg-x16 --device=dg1
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelpg-x16 --device=acm-g10
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelpg-x16 --device=acm-g11
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelpg-x16 --device=acm-g12
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelpg-x16 --device=mtl-m
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelpg-x16 --device=mtl-p

// REQUIRES: XE_ENABLED
// REQUIRES: LINUX_HOST
// REQUIRES: OCLOC_INSTALLED

uniform int i;

task void foo() {}
