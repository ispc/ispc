// RUN: not %{ispc} -O0 -g %s > %t 2>&1
// RUN: FileCheck --input-file=%t %s
// CHECK-NOT: Please file a bug report at https://github.com/ispc/ispc/issues
export void compressPositions(uniform int clustersSize) { 
    for (uniform int i = programIndex; i < clustersSize; i += programCount) {
    }
}
