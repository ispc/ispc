// RUN: %{ispc} -O2 --target=avx2-i64x4 --nowrap %s > %t 2>&1
// RUN: FileCheck --input-file=%t %s
// REQUIRES: X86_ENABLED
// CHECK-NOT: LLVM ERROR: Instruction Combining seems stuck in an infinite loop after 1000 iterations.
int8 a[1][3][6];
void b(uniform int32 c, uniform int8 d[][3][6]) {
    foreach (e = 0...c)
        for (int8 f = 0;;)
            for (int g = 0;;)
                a[e][f][g] = min(~d[e][f][g], 0);
}
