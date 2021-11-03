// COM: Xe Gather Coalescing is disabled to not perform redundant optimizations on gathers.
// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 -DUNIFORM --emit-llvm-text --nowrap -o %t.ll --opt=disable-xe-gather-coalescing
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_XE_16 %s
// RUN: %{ispc} %s --target=gen9-x8 --arch=xe64 -DUNIFORM --emit-llvm-text --nowrap -o %t.ll --opt=disable-xe-gather-coalescing
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_XE_8 %s

// REQUIRES: XE_ENABLED
struct FVector
{
    float V[3];
};

struct FHalfVector
{
    int16 V[3];
};

struct GenerateHalfFloatParameters
{
    int NumInstances;
    FVector *Data;
    FHalfVector *DataHalf;
};

inline FHalfVector FloatToHalf(FVector v)
{
    FHalfVector hv;
    hv.V[0] = (int16)float_to_half(v.V[0]);
    hv.V[1] = (int16)float_to_half(v.V[1]);
    hv.V[2] = (int16)float_to_half(v.V[2]);
    return hv;
}

inline FVector HalfToFloat(FHalfVector v)
{
    FVector hv;
    hv.V[0] = (float)half_to_float(v.V[0]);
    hv.V[1] = (float)half_to_float(v.V[1]);
    hv.V[2] = (float)half_to_float(v.V[2]);
    return hv;
}

task void GenerateHalfData(void *uniform _p)
{
    GenerateHalfFloatParameters *uniform p = (GenerateHalfFloatParameters * uniform) _p;
  // CHECK_XE_8: <16 x i16> @llvm.genx.svm.gather.v16i16.v8i1.v8i64(<8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, i32 1
  // CHECK_XE_8: <8 x i16> @llvm.genx.rdregioni.v8i16.v16i16.i16
  // CHECK_XE_8: <16 x i16> @llvm.genx.svm.gather.v16i16.v8i1.v8i64(<8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, i32 1
  // CHECK_XE_8: <8 x i16> @llvm.genx.rdregioni.v8i16.v16i16.i16
  // CHECK_XE_8: <16 x i16> @llvm.genx.svm.gather.v16i16.v8i1.v8i64(<8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, i32 1
  // CHECK_XE_8: <8 x i16> @llvm.genx.rdregioni.v8i16.v16i16.i16

  // CHECK_XE_16: <32 x i16> @llvm.genx.svm.gather.v32i16.v16i1.v16i64(<16 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, i32 1
  // CHECK_XE_16: <16 x i16> @llvm.genx.rdregioni.v16i16.v32i16.i16
  // CHECK_XE_16: <32 x i16> @llvm.genx.svm.gather.v32i16.v16i1.v16i64(<16 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, i32 1
  // CHECK_XE_16: @llvm.genx.rdregioni.v16i16.v32i16.i16
  // CHECK_XE_16: <32 x i16> @llvm.genx.svm.gather.v32i16.v16i1.v16i64(<16 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, i32 1
  // CHECK_XE_16: <16 x i16> @llvm.genx.rdregioni.v16i16.v32i16.i16
      p->Data[programIndex] = HalfToFloat(p->DataHalf[programIndex]);
}

task void GenerateFloatData(void *uniform _p)
{
    GenerateHalfFloatParameters *uniform p = (GenerateHalfFloatParameters * uniform) _p;
// CHECK_XE_8: <16 x i16> @llvm.genx.wrregioni.v16i16.v8i16.i16.v8i1
// CHECK_XE_8: void @llvm.genx.svm.scatter.v8i1.v8i64.v16i16(<8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, i32 1
// CHECK_XE_8: @llvm.genx.wrregioni.v16i16.v8i16.i16.v8i1
// CHECK_XE_8: void @llvm.genx.svm.scatter.v8i1.v8i64.v16i16(<8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, i32 1
// CHECK_XE_8: <16 x i16> @llvm.genx.wrregioni.v16i16.v8i16.i16.v8i1
// CHECK_XE_8: void @llvm.genx.svm.scatter.v8i1.v8i64.v16i16(<8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, i32 1

// CHECK_XE_16: <32 x i16> @llvm.genx.wrregioni.v32i16.v16i16.i16.v16i1
// CHECK_XE_16: void @llvm.genx.svm.scatter.v16i1.v16i64.v32i16(<16 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, i32 1
// CHECK_XE_16: <32 x i16> @llvm.genx.wrregioni.v32i16.v16i16.i16.v16i1
// CHECK_XE_16: void @llvm.genx.svm.scatter.v16i1.v16i64.v32i16(<16 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, i32 1
// CHECK_XE_16:  <32 x i16> @llvm.genx.wrregioni.v32i16.v16i16.i16.v16i1
// CHECK_XE_16: void @llvm.genx.svm.scatter.v16i1.v16i64.v32i16(<16 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, i32 1

    p->DataHalf[programIndex] = FloatToHalf(p->Data[programIndex]);
}
