// RUN: not %{ispc} --target=host --nostdlib --nowrap -o - --emit-llvm-text %s 2>&1 | FileCheck %s

// CHECK-NOT: LLVM ERROR
inline float gather(varying float& ptr, const uniform size_t stride, const size_t pid, const size_t rid) {
return ((uniform float_)(((uniform int8_ uniform)&ptr) + pid*stride))[rid];
}
