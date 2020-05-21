// RUN: %{ispc} --target=host %s > %t 2>&1
struct FVector8
{
	float<8> V;
};

unmasked void Add8(
    uniform FVector8 &VResult,
	const uniform FVector8& A,
	const uniform FVector8& B)
{
    varying float S0, S1, Result;
	*((uniform FVector8 *uniform)&S0) = *((uniform FVector8 *uniform)&A);
	*((uniform FVector8 *uniform)&S1) = *((uniform FVector8 *uniform)&B);

    Result = S0 + S1;

    *((uniform FVector8 *uniform)&VResult) = *((uniform FVector8 *uniform)&Result);
}

