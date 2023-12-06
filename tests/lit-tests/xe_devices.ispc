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
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelpg-x8 --device=mtl-u
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelpg-x8 --device=mtl-h
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelpg-x16 --device=skl
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelpg-x16 --device=tgllp
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelpg-x16 --device=dg1
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelpg-x16 --device=acm-g10
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelpg-x16 --device=acm-g11
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelpg-x16 --device=acm-g12
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelpg-x16 --device=mtl-u
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xelpg-x16 --device=mtl-h

// Xe2HPG
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2hpg-x16 --device=skl
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2hpg-x16 --device=tgllp
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2hpg-x16 --device=dg1
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2hpg-x16 --device=acm-g10
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2hpg-x16 --device=acm-g11
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2hpg-x16 --device=acm-g12
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2hpg-x16 --device=mtl-u
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2hpg-x16 --device=mtl-h
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2hpg-x16 --device=bmg-g21
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2hpg-x32 --device=skl
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2hpg-x32 --device=tgllp
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2hpg-x32 --device=dg1
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2hpg-x32 --device=acm-g10
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2hpg-x32 --device=acm-g11
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2hpg-x32 --device=acm-g12
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2hpg-x32 --device=mtl-u
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2hpg-x32 --device=mtl-h
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2hpg-x32 --device=bmg-g21

// Xe2LPG
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2lpg-x16 --device=skl
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2lpg-x16 --device=tgllp
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2lpg-x16 --device=dg1
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2lpg-x16 --device=acm-g10
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2lpg-x16 --device=acm-g11
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2lpg-x16 --device=acm-g12
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2lpg-x16 --device=mtl-u
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2lpg-x16 --device=mtl-h
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2lpg-x16 --device=bmg-g21
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2lpg-x16 --device=lnl-m
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2lpg-x32 --device=skl
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2lpg-x32 --device=tgllp
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2lpg-x32 --device=dg1
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2lpg-x32 --device=acm-g10
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2lpg-x32 --device=acm-g11
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2lpg-x32 --device=acm-g12
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2lpg-x32 --device=mtl-u
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2lpg-x32 --device=mtl-h
// RUN: not %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2lpg-x32 --device=bmg-g21
// RUN: %{ispc} %s -o %t.bin --emit-zebin --nostdlib --target=xe2lpg-x32 --device=lnl-m

// REQUIRES: XE_ENABLED
// REQUIRES: LINUX_HOST
// REQUIRES: OCLOC_INSTALLED

uniform int i;

task void foo() {}
