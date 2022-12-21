; ModuleID = 'float16_uniform.ispc'
source_filename = "float16_uniform.ispc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nofree nosync nounwind readnone
declare <8 x i16> @llvm.x86.vcvtps2ph.256(<8 x float>, i32 immarg) #0

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone willreturn
define half @foo0___unh(half %arg0, <4 x i32> %__mask) local_unnamed_addr #1 {
allocas:
  %add_arg0_load_ = fadd half %arg0, 0xH011D
  ret half %add_arg0_load_
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn
define void @foo1___REFunhun_3C_unh_3E_(half* noalias nocapture %arg0, half* noalias nocapture %arg1, <4 x i32> %__mask) local_unnamed_addr #2 {
allocas:
  store half 0xH4252, half* %arg0, align 2
  %arg1_load7_load = load half, half* %arg1, align 2
  %mul_arg1_load7_load_arg0_load10_load = fmul half %arg1_load7_load, 0xH4252
  store half %mul_arg1_load7_load_arg0_load10_load, half* %arg1, align 2
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone willreturn
define half @foo2___unhunh(half %arg0, half %arg1, <4 x i32> %__mask) local_unnamed_addr #1 {
allocas:
  %greater_arg0_load_arg1_load = fcmp ogt half %arg0, %arg1
  %arg0.arg1 = select i1 %greater_arg0_load_arg1_load, half %arg0, half %arg1
  ret half %arg0.arg1
}

; Function Attrs: mustprogress nofree noinline nosync nounwind readnone willreturn
define half @goo3___unh(half %arg0, <4 x i32> %__mask) local_unnamed_addr #3 {
allocas:
  %r.i.i.i = fpext half %arg0 to float
  %add_arg0_load_to_float_ = fadd float %r.i.i.i, 0x401F9999A0000000
  %v1.i.i.i = bitcast float %add_arg0_load_to_float_ to <1 x float>
  %vv.i.i.i = shufflevector <1 x float> %v1.i.i.i, <1 x float> undef, <8 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rv.i.i.i = tail call <8 x i16> @llvm.x86.vcvtps2ph.256(<8 x float> %vv.i.i.i, i32 0) #5
  %bc = bitcast <8 x i16> %rv.i.i.i to <8 x half>
  %int16_to_float16_bitcast.i.i.i = extractelement <8 x half> %bc, i32 0
  ret half %int16_to_float16_bitcast.i.i.i
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define half @foo3___unh(half %arg0, <4 x i32> %__mask) local_unnamed_addr #4 {
allocas:
  %calltmp = tail call half @goo3___unh(half %arg0, <4 x i32> undef)
  ret half %calltmp
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone willreturn
define half @foo4___uni(i32 %arg0, <4 x i32> %__mask) local_unnamed_addr #1 {
allocas:
  %arg0_load_to_float16 = sitofp i32 %arg0 to half
  ret half %arg0_load_to_float16
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone willreturn
define double @foo5___unh(half %arg0, <4 x i32> %__mask) local_unnamed_addr #1 {
allocas:
  %arg0_load_to_double = fpext half %arg0 to double
  ret double %arg0_load_to_double
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone willreturn
define void @foo6___(<4 x i32> %__mask) local_unnamed_addr #1 {
allocas:
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone willreturn
define i8 @foo7___unfunh(float %arg0, half %arg1, <4 x i32> %__mask) local_unnamed_addr #1 {
allocas:
  %r.i.i.i = fpext half %arg1 to float
  %add_arg0_load_arg1_load_to_float = fadd float %r.i.i.i, %arg0
  %add_arg0_load_arg1_load_to_float_to_int8 = fptosi float %add_arg0_load_arg1_load_to_float to i8
  ret i8 %add_arg0_load_arg1_load_to_float_to_int8
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone willreturn
define i8 @foo8___unsunh(i16 %arg0, half %arg1, <4 x i32> %__mask) local_unnamed_addr #1 {
allocas:
  %arg0_load_to_float16 = sitofp i16 %arg0 to half
  %add_arg0_load_to_float16_arg1_load = fadd half %arg0_load_to_float16, %arg1
  %add_arg0_load_to_float16_arg1_load_to_int8 = fptosi half %add_arg0_load_to_float16_arg1_load to i8
  ret i8 %add_arg0_load_to_float16_arg1_load_to_int8
}

attributes #0 = { nofree nosync nounwind readnone }
attributes #1 = { mustprogress nofree norecurse nosync nounwind readnone willreturn }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn }
attributes #3 = { mustprogress nofree noinline nosync nounwind readnone willreturn }
attributes #4 = { mustprogress nofree nosync nounwind readnone willreturn }
attributes #5 = { nounwind }

!llvm.ident = !{!0, !1}
!llvm.module.flags = !{!2, !3, !4, !5}

!0 = !{!"Intel(r) Implicit SPMD Program Compiler (Intel(r) ISPC), 1.19.0dev (build commit 427565bfaf3018c7 @ 20230104, LLVM 13.0.1)"}
!1 = !{!"LLVM version 13.0.1 (https://github.com/llvm/llvm-project.git 75e33f71c2dae584b13a7d1186ae0a038ba98838)"}
!2 = !{i32 1, !"wchar_size", i32 4}
!3 = !{i32 7, !"PIC Level", i32 2}
!4 = !{i32 7, !"uwtable", i32 1}
!5 = !{i32 7, !"frame-pointer", i32 2}
