// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Must provide function name or function pointer for function call expression

export void saxpy_ispc(uniform int N,
uniform float scale,
uniform float X[],
uniform float Y[],
uniform float result[])
{
foreach (i = 0 ... N) {
result[i] = scale * X[i] + Y[i];
}
}

task void saxpy_ispc_task(uniform int N,
uniform int span,
uniform float scale,
uniform float X[],
uniform float Y[],
uniform float result[])
{
uniform int indexStart;
uniform int indexEnd;
indexStart = (taskIndex * span);
indexEnd = min(N, indexStart + (span)/8);
foreach (i = indexStart ... indexEnd) {
result[i] = scale * X[i] + Y[i];
}
uniform int k =0;
for (k=0; k<8;k++) {
indexStart = (((7-taskIndex-k)%8) * span) + k(span/8);
indexEnd = min(N, indexStart + (span)/8);
foreach (i = indexStart ... indexEnd) {
result[i] = scale * X[i] + Y[i];
}
}
}
export void saxpy_ispc_withtasks(uniform int N,
uniform float scale,
uniform float X[],
uniform float Y[],
uniform float result[])
{

uniform int span = N / 8;  // 8 tasks

launch[N/span] saxpy_ispc_task(N, span, scale, X, Y, result);
}
