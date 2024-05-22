;;  Copyright (c) 2011-2024, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

;; This file defines various functions that are used when generating the
;; the "dispatch" object/assembly file that has entrypoints for each
;; exported function in a module that dispatch to the best available
;; variant of that function that will run on the system's CPU.

;; Stores the best target ISA that the system on which we're actually
;; running supports.  -1 represents "uninitialized", otherwise this value
;; should correspond to one of the enumerant values of Target::ISA from
;; ispc.h.

@__system_best_isa = internal global i32 -1

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

define(`PTR_OP_ARGS',
  `$1 , $1 *'
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; Function Attrs: nounwind uwtable
define dso_local i32 @__get_system_isa() local_unnamed_addr {
entry:
  %0 = tail call { i32, i32, i32, i32 } asm sideeffect "cpuid", "={ax},={bx},={cx},={dx},0,~{dirflag},~{fpsr},~{flags}"(i32 1) nounwind
  %asmresult5.i = extractvalue { i32, i32, i32, i32 } %0, 2
  %asmresult6.i = extractvalue { i32, i32, i32, i32 } %0, 3
  %1 = tail call { i32, i32, i32, i32 } asm sideeffect "cpuid", "={ax},={bx},={cx},={dx},0,2,~{dirflag},~{fpsr},~{flags}"(i32 7, i32 0) nounwind
  %.fr = freeze { i32, i32, i32, i32 } %1
  %asmresult.i300 = extractvalue { i32, i32, i32, i32 } %.fr, 0
  %asmresult4.i301 = extractvalue { i32, i32, i32, i32 } %.fr, 1
  %asmresult5.i302 = extractvalue { i32, i32, i32, i32 } %.fr, 2
  %asmresult6.i303 = extractvalue { i32, i32, i32, i32 } %.fr, 3
  %cmp = icmp sgt i32 %asmresult.i300, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %2 = tail call { i32, i32, i32, i32 } asm sideeffect "cpuid", "={ax},={bx},={cx},={dx},0,2,~{dirflag},~{fpsr},~{flags}"(i32 7, i32 1) nounwind
  %asmresult.i307 = extractvalue { i32, i32, i32, i32 } %2, 0
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %info3.sroa.0.0 = phi i32 [ %asmresult.i307, %if.then ], [ 0, %entry ]
  %and = and i32 %asmresult6.i, 67108864
  %cmp4.not = icmp ne i32 %and, 0
  %and10 = and i32 %asmresult5.i, 1048576
  %cmp11.not.not = icmp eq i32 %and10, 0
  %and18 = and i32 %asmresult4.i301, 32
  %cmp19.not = icmp eq i32 %and18, 0
  %and22 = and i32 %info3.sroa.0.0, 16
  %cmp23.not = icmp ne i32 %and22, 0
  %and34 = and i32 %asmresult5.i, 134217728
  %cmp35.not = icmp eq i32 %and34, 0
  %3 = and i32 %asmresult4.i301, 65568
  %4 = icmp ne i32 %3, 65568
  %brmerge271 = or i1 %cmp35.not, %4
  br i1 %brmerge271, label %if.end220, label %land.lhs.true44

land.lhs.true44:                                  ; preds = %if.end
  %5 = tail call { i32, i32 } asm sideeffect ".byte 0x0f, 0x01, 0xd0", "={ax},={dx},{cx},~{dirflag},~{fpsr},~{flags}"(i32 0) nounwind
  %asmresult.i311 = extractvalue { i32, i32 } %5, 0
  %and.i = and i32 %asmresult.i311, 230
  %cmp.i.not = icmp eq i32 %and.i, 230
  br i1 %cmp.i.not, label %if.then46, label %if.end220

if.then46:                                        ; preds = %land.lhs.true44
  %6 = insertelement <4 x i32> poison, i32 %asmresult5.i302, i64 0
  %7 = shufflevector <4 x i32> %6, <4 x i32> poison, <4 x i32> zeroinitializer
  %.fr327 = freeze <4 x i32> %7
  %8 = and <4 x i32> %.fr327, <i32 2048, i32 64, i32 256, i32 512>
  %and84 = and i32 %asmresult5.i302, 1024
  %cmp85.not = icmp ne i32 %and84, 0
  %9 = icmp eq <4 x i32> %8, zeroinitializer
  %10 = insertelement <4 x i32> poison, i32 %info3.sroa.0.0, i64 0
  %11 = insertelement <4 x i32> %10, i32 %asmresult6.i303, i64 1
  %12 = shufflevector <4 x i32> %11, <4 x i32> poison, <4 x i32> <i32 0, i32 1, i32 1, i32 1>
  %.fr329 = freeze <4 x i32> %12
  %13 = and <4 x i32> %.fr329, <i32 32, i32 4194304, i32 16777216, i32 33554432>
  %14 = icmp eq <4 x i32> %13, zeroinitializer
  %and120 = and i32 %asmresult6.i303, 8388608
  %cmp121 = icmp ne i32 %and120, 0
  %15 = and i32 %asmresult4.i301, 469762048
  %spec.select288 = icmp eq i32 %15, 469762048
  %16 = and i32 %asmresult4.i301, 268566528
  %brmerge272.not = icmp eq i32 %16, 268566528
  %spec.select = icmp ugt i32 %asmresult4.i301, -1073741825
  %spec.select321 = and i1 %brmerge272.not, %spec.select
  %17 = and i32 %asmresult5.i302, 20480
  %spec.select277 = icmp eq i32 %17, 20480
  %18 = bitcast <4 x i1> %9 to i4
  %19 = icmp eq i4 %18, 0
  %op.rdx324 = and i1 %19, %cmp85.not
  %op.rdx325 = and i1 %spec.select321, %spec.select277
  %op.rdx326 = and i1 %op.rdx324, %op.rdx325
  %20 = bitcast <4 x i1> %14 to i4
  %21 = icmp eq i4 %20, 0
  %op.rdx = and i1 %21, %cmp121
  %22 = and i1 %op.rdx, %op.rdx326
  %op.rdx323 = select i1 %22, i1 %cmp23.not, i1 false
  %.mux = select i1 %op.rdx326, i32 8, i32 7
  %.mux.mux = select i1 %op.rdx323, i32 9, i32 %.mux
  %retval.0 = select i1 %spec.select321, i32 %.mux.mux, i32 6
  %cond.not = or i1 %spec.select321, %spec.select288
  br i1 %cond.not, label %cleanup248, label %if.end220

if.end220:                                        ; preds = %if.end, %if.then46, %land.lhs.true44
  %23 = and i32 %asmresult5.i, 402653184
  %brmerge285.not = icmp eq i32 %23, 402653184
  br i1 %brmerge285.not, label %land.lhs.true224, label %if.else238

land.lhs.true224:                                 ; preds = %if.end220
  %24 = tail call { i32, i32 } asm sideeffect ".byte 0x0f, 0x01, 0xd0", "={ax},={dx},{cx},~{dirflag},~{fpsr},~{flags}"(i32 0) nounwind
  %asmresult.i312 = extractvalue { i32, i32 } %24, 0
  %and.i313 = and i32 %asmresult.i312, 6
  %cmp.i314.not = icmp eq i32 %and.i313, 6
  br i1 %cmp.i314.not, label %if.then227, label %if.else238

if.then227:                                       ; preds = %land.lhs.true224
  br i1 %cmp23.not, label %cleanup248, label %if.end230

if.end230:                                        ; preds = %if.then227
  %25 = and i32 %asmresult5.i, 1610612736
  %brmerge286 = icmp ne i32 %25, 1610612736
  %brmerge287 = or i1 %brmerge286, %cmp19.not
  %spec.select292 = select i1 %brmerge287, i32 3, i32 4
  br label %cleanup248

if.else238:                                       ; preds = %if.end220, %land.lhs.true224
  %26 = and i32 %asmresult5.i, 1572864
  %brmerge293 = icmp ne i32 %26, 0
  %.mux294 = select i1 %cmp11.not.not, i32 1, i32 2
  %brmerge295 = select i1 %brmerge293, i1 true, i1 %cmp4.not
  %.mux294.mux = select i1 %brmerge293, i32 %.mux294, i32 0
  br i1 %brmerge295, label %cleanup248, label %if.else247

if.else247:                                       ; preds = %if.else238
  tail call void @abort() #3
  unreachable

cleanup248:                                       ; preds = %if.end230, %if.else238, %if.then227, %if.then46
  %retval.1 = phi i32 [ %retval.0, %if.then46 ], [ 5, %if.then227 ], [ %.mux294.mux, %if.else238 ], [ %spec.select292, %if.end230 ]
  ret i32 %retval.1
}

declare void @abort() local_unnamed_addr noreturn nounwind

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; This function is called by each of the dispatch functions we generate;
;; it sets @__system_best_isa if it is unset.

define void @__set_system_isa() {
entry:
  %bi = load PTR_OP_ARGS(`i32 ')  @__system_best_isa
  %unset = icmp eq i32 %bi, -1
  br i1 %unset, label %set_system_isa, label %done

set_system_isa:
  %bival = call i32 @__get_system_isa()
  store i32 %bival, i32* @__system_best_isa
  ret void

done:
  ret void
}

