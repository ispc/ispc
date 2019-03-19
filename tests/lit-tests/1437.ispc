//; RUN: %{ispc}  %s
typedef float<2> float2;

float2 BitonicMergeKernel(float2 &b) {
    float2 bs = b.xy;
    return bs;
}
