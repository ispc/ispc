// This test checks code generation for the __regcall functions. This calling
// convention makes return values and function arguments passed through registers
// in most cases. ISPC follows this convention by default so the same code
// generation will happen for the functions not marked with __regcall keyword as well.

// RUN: %{ispc} %s --target=gen9-x16 --emit-llvm-text --no-discard-value-names --nowrap -o - | FileCheck %s

// REQUIRES: XE_ENABLED

//CHECK: %v16_uniform_C_Struct = type { %v16_uniform_C, {{(i32\*)|ptr}}, <4 x float> }
//CHECK: %v16_uniform_C = type { float, float }
//CHECK: %v16_uniform_C1 = type { %v16_uniform_B1 }
//CHECK: %v16_uniform_B1 = type { %v16_uniform_A1 }
//CHECK: %v16_uniform_A1 = type { i8 }
//CHECK: %v16_varying_Float = type { <16 x float> }
//CHECK: %v16_uniform_StillPassThroughRegisters = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
//CHECK: %v16_uniform_EndsWithFlexArray = type { i32, [0 x i32] }

// CHECK: declare x86_regcallcc <16 x i32> @__regcall3__foo1(
// CHECK: call x86_regcallcc <16 x i32> @__regcall3__foo1(

extern "C" __regcall int foo1(float b);

extern "C" __regcall int foo2(float b) {
  foo1(b);
}

struct C {
  float x, y;
};

struct C_Struct {
  C a;
  int *b;
  float<3> c;
};

// CHECK: define x86_regcallcc %v16_uniform_C_Struct @__regcall3__foo3()
extern "C" __regcall uniform C_Struct foo3() {
  uniform C_Struct s;
  s.a.x += 1;
  return s;
}

struct A1 { int8 x; };
struct B1 { A1 a; };
struct C1 {
  B1 b;
};

// CHECK: define x86_regcallcc %v16_uniform_C1 @__regcall3__foo4
extern "C" __regcall uniform C1 foo4(int b) {
  uniform C1 s;
  s.b.a.x = 1;
  return s;
}

struct Float {
  float a;
};

// CHECK: define x86_regcallcc %v16_varying_Float @__regcall3__foo5
extern "C" __regcall Float foo5(int b) {
  Float f;
  f.a = b;
  return f;
}

struct StillPassThroughRegisters {
  // 130 total:
  int a, a01, a02, a03, a04, a05, a06, a07, a08, a09,
    a10, a11, a12, a13, a14, a15, a16, a17, a18, a19,
    a20, a21, a22, a23, a24, a25, a26, a27, a28, a29,
    a30, a31, a32, a33, a34, a35, a36, a37, a38, a39,
    a40, a41, a42, a43, a44, a45, a46, a47, a48, a49,
    a50, a51, a52, a53, a54, a55, a56, a57, a58, a59,
    a60, a61, a62, a63, a64, a65, a66, a67, a68, a69,
    a70, a71, a72, a73, a74, a75, a76, a77, a78, a79,
    a80, a81, a82, a83, a84, a85, a86, a87, a88, a89,
    a90, a91, a92, a93, a94, a95, a96, a97, a98, a99,
    aa0, aa1, aa2, aa3, aa4, aa5, aa6, aa7, aa8, aa9,
    ab0, ab1, ab2, ab3, ab4, ab5, ab6, ab7, ab8, ab9,
    ac0, ac1, ac2, ac3, ac4, ac5, ac6, ac7, ac8, ac9;

};

// CHECK: define x86_regcallcc %v16_uniform_StillPassThroughRegisters @__regcall3__foo6
extern "C" __regcall uniform StillPassThroughRegisters foo6(uniform int b) {
  uniform StillPassThroughRegisters f;
  f.a = b;
  return f;
}

struct EndsWithFlexArray {
  int a;
  int x[];
};

// CHECK: define x86_regcallcc %v16_uniform_EndsWithFlexArray @__regcall3__foo7
extern "C" __regcall uniform EndsWithFlexArray foo7(uniform int b) {
  uniform EndsWithFlexArray f;
  f.a = b;
  return f;
}

// CHECK: define x86_regcallcc <16 x float> @__regcall3__foo8({{(float addrspace\(4\)\*)|(ptr addrspace\(4\))}} noalias nocapture {{(readonly)?}} %A, <16 x float> %b, i32 %I, <16 x i1> %mask)
extern "C" __regcall float foo8(uniform float A[], float b, uniform int I, bool mask) {
  return A[programIndex+I];
}

struct Callback {
 float (*uniform callback)(float, float, uniform float);
};

// CHECK: define x86_regcallcc <16 x float> @__regcall3__foo9({{(%v16_uniform_Callback addrspace\(4\)\*)|(ptr addrspace\(4\))}} noalias nocapture readnone %callback)
extern "C" __regcall float foo9(Callback * uniform callback) {}

// CHECK: define x86_regcallcc <16 x float> @__regcall3__foo10(<16 x i64> %ptr)
extern "C" __regcall float foo10(uniform float* ptr) {}

// CHECK: define x86_regcallcc <16 x float> @__regcall3__foo11({{(float addrspace\(4\)\*)|(ptr addrspace\(4\))}} noalias nocapture readnone %ptr)
extern "C" __regcall float foo11(uniform float* uniform ptr) {}
