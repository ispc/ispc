// RUN: %{ispc} %s --target=xehpg-x16 --emit-llvm-text -o - | FileCheck %s

// REQUIRES: XE_ENABLED

// CHECK-NOT:   [[V:%.*]] = add <16 x i64> %base.i, <i64 {{.*}}, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison>

struct vec4f { float x, y, z, w; };
struct FB { vec4f *buf; };
unmasked void foo(FB *uniform p, const int index) {
  const vec4f color = { 0, 1, 2, 3, };
  p->buf[index] = color;
}
