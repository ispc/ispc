// RUN: not %{ispc} --target=sse4-x2 --nostdlib --nowrap -o - --emit-llvm-text %s 2>&1 | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK-NOT: LLVM ERROR
export void crash(float input[], float output[]) {output[0] = round(input[0]);}
