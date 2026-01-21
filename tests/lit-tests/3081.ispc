// RUN: %{ispc} --emit-llvm-text -O2 --opt=disable-loop-unroll --target=sse4 %s -o - | FileCheck %s -check-prefix=CHECK_SSE4
// REQUIRE: X86_ENABLED && LLVM_20_0+

extern "C" void use(uniform int);

// CHECK_SSE4:  fptosi <4 x float>
// CHECK_SSE4:  br label %[[FOR_LOOP:.*]]
// CHECK_SSE4:  [[FOR_LOOP]]:
// CHECK_SSE4:  extractelement <4 x i32>
unmasked void draw_rects(
    float min_x)
{
        int min_xi = floor(min_x);
        for (uniform int i = 0; i < 4; ++i)
        {
            uniform int x0 = extract(min_xi, i);
            use(x0);
        }
}
