// Test that compiler does not crash when using sample profiling with gather operations
// that get transformed to gep/load by ImproveMemoryOpsPass.
//
// REQUIRES: X86_ENABLED
// RUN: %{ispc} -O2 --sample-profiling-debug-info --profile-sample-use=%S/prof_metadata.prof --target=host --nowrap -o /dev/null %s

noinline varying float LoadValue(const uniform float Values[], const varying int Index, const bool IsConstant)
{
    return IsConstant ? Values[0] : Values[Index];
}

unmasked void ComputeResult(const uniform float InputA[], const uniform float InputB[],
                          const bool IsConstA, const bool IsConstB,
                          uniform float Output[], const uniform int Count) {
    foreach (i = 0 ... Count) {
        const varying float ValA = LoadValue(InputA, i, IsConstA);
        const varying float ValB = LoadValue(InputB, i, IsConstB);

        float Result;

        if (ValA > 0.0f) {
            const float Scale = ValA + 1.0f;
            Result = (ValA - ValB) * Scale;
        }

        Output[i] = Result;
    }
}
