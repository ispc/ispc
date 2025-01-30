// RUN: %{ispc} --target=host --nowrap --nostdlib %s --emit-llvm-text -o - | FileCheck %s

// CHECK: define void @foo___un_3C_uni_3E_({{.*}} {{nocapture|captures\(none\)}} %ptr, {{.*}}
void foo(__attribute__((noescape)) uniform int *uniform ptr) { return; }
