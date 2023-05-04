;;  Copyright (c) 2011-2023, Intel Corporation
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

define i32 @__get_system_isa() local_unnamed_addr nounwind uwtable {
entry:
  %0 = tail call { i32, i32, i32, i32 } asm sideeffect "cpuid", "={ax},={bx},={cx},={dx},0,~{dirflag},~{fpsr},~{flags}"(i32 1) nounwind
  %asmresult5.i = extractvalue { i32, i32, i32, i32 } %0, 2
  %asmresult6.i = extractvalue { i32, i32, i32, i32 } %0, 3
  %1 = tail call { i32, i32, i32, i32 } asm sideeffect "cpuid", "={ax},={bx},={cx},={dx},0,2,~{dirflag},~{fpsr},~{flags}"(i32 7, i32 0) nounwind
  %asmresult.i288 = extractvalue { i32, i32, i32, i32 } %1, 0
  %asmresult4.i289 = extractvalue { i32, i32, i32, i32 } %1, 1
  %asmresult5.i290 = extractvalue { i32, i32, i32, i32 } %1, 2
  %asmresult6.i291 = extractvalue { i32, i32, i32, i32 } %1, 3
  %and = and i32 %asmresult6.i, 67108864
  %cmp.not = icmp ne i32 %and, 0
  %and7 = and i32 %asmresult5.i, 1048576
  %cmp8.not.not = icmp eq i32 %and7, 0
  %and15 = and i32 %asmresult4.i289, 32
  %cmp16.not = icmp eq i32 %and15, 0
  %and27 = and i32 %asmresult5.i, 134217728
  %cmp28.not = icmp eq i32 %and27, 0
  %and31 = and i32 %asmresult4.i289, 65536
  %cmp32.not = icmp eq i32 %and31, 0
  %brmerge = select i1 %cmp28.not, i1 true, i1 %cmp16.not
  %brmerge262 = select i1 %brmerge, i1 true, i1 %cmp32.not
  br i1 %brmerge262, label %if.end219, label %land.lhs.true37

land.lhs.true37:                                  ; preds = %entry
  %2 = tail call { i32, i32 } asm sideeffect ".byte 0x0f, 0x01, 0xd0", "={ax},={dx},{cx},~{dirflag},~{fpsr},~{flags}"(i32 0) nounwind
  %asmresult.i292 = extractvalue { i32, i32 } %2, 0
  %and.i = and i32 %asmresult.i292, 230
  %cmp.i.not = icmp eq i32 %and.i, 230
  br i1 %cmp.i.not, label %if.then, label %if.end219

if.then:                                          ; preds = %land.lhs.true37
  %cmp40 = icmp sgt i32 %asmresult.i288, 0
  br i1 %cmp40, label %if.then41, label %if.end

if.then41:                                        ; preds = %if.then
  %3 = tail call { i32, i32, i32, i32 } asm sideeffect "cpuid", "={ax},={bx},={cx},={dx},0,2,~{dirflag},~{fpsr},~{flags}"(i32 7, i32 1) nounwind
  %asmresult.i296 = extractvalue { i32, i32, i32, i32 } %3, 0
  br label %if.end

if.end:                                           ; preds = %if.then41, %if.then
  %info3.sroa.0.0 = phi i32 [ %asmresult.i296, %if.then41 ], [ 0, %if.then ]
  %and96 = and i32 %info3.sroa.0.0, 16
  %cmp97.not = icmp ne i32 %and96, 0
  %and100 = and i32 %info3.sroa.0.0, 32
  %cmp101.not = icmp ne i32 %and100, 0
  %and108 = and i32 %asmresult6.i291, 4194304
  %cmp109.not = icmp ne i32 %and108, 0
  %and112 = and i32 %asmresult6.i291, 16777216
  %cmp113.not = icmp ne i32 %and112, 0
  %and116 = and i32 %asmresult6.i291, 33554432
  %cmp117.not = icmp ne i32 %and116, 0
  %and120 = and i32 %asmresult6.i291, 8388608
  %cmp121 = icmp ne i32 %and120, 0
  %4 = and i32 %asmresult4.i289, 469762048
  %.not = icmp eq i32 %4, 469762048
  %5 = and i32 %asmresult4.i289, 268566528
  %.not314 = icmp eq i32 %5, 268566528
  br i1 %.not314, label %land.end135, label %cleanup

land.end135:                                      ; preds = %if.end
  %6 = insertelement <4 x i32> poison, i32 %asmresult5.i290, i64 0
  %shuffle = shufflevector <4 x i32> %6, <4 x i32> poison, <4 x i32> zeroinitializer
  %shuffle.fr = freeze <4 x i32> %shuffle
  %7 = and <4 x i32> %shuffle.fr, <i32 2048, i32 64, i32 256, i32 512>
  %and80 = and i32 %asmresult5.i290, 1024
  %cmp81.not = icmp eq i32 %and80, 0
  %8 = icmp eq <4 x i32> %7, zeroinitializer
  %9 = icmp ugt i32 %asmresult4.i289, -1073741825
  %not. = xor i1 %9, true
  %10 = bitcast <4 x i1> %8 to i4
  %11 = icmp ne i4 %10, 0
  %12 = select i1 %11, i1 true, i1 %cmp81.not
  %op.rdx321 = select i1 %12, i1 true, i1 %not.
  br i1 %op.rdx321, label %if.end184, label %land.lhs.true156

land.lhs.true156:                                 ; preds = %land.end135
  %13 = and i32 %asmresult5.i290, 20480
  %14 = icmp eq i32 %13, 20480
  %brmerge269 = select i1 %14, i1 %cmp101.not, i1 false
  %brmerge270 = select i1 %brmerge269, i1 %cmp109.not, i1 false
  %brmerge271 = select i1 %brmerge270, i1 %cmp113.not, i1 false
  %brmerge272 = select i1 %brmerge271, i1 %cmp117.not, i1 false
  %15 = select i1 %brmerge272, i1 %cmp97.not, i1 false
  %or.cond279 = select i1 %15, i1 %cmp121, i1 false
  %brmerge280 = select i1 %or.cond279, i1 true, i1 %9
  %.mux = select i1 %or.cond279, i32 7, i32 6
  %brmerge320 = select i1 %brmerge280, i1 true, i1 %.not
  %.mux.mux = select i1 %brmerge280, i32 %.mux, i32 5
  br i1 %brmerge320, label %cleanup244, label %if.end219

if.end184:                                        ; preds = %land.end135
  %brmerge318 = or i1 %9, %.not
  %.mux319 = select i1 %9, i32 6, i32 5
  br i1 %brmerge318, label %cleanup244, label %if.end219

cleanup:                                          ; preds = %if.end
  br i1 %.not, label %cleanup244, label %if.end219

if.end219:                                        ; preds = %land.lhs.true156, %if.end184, %entry, %cleanup, %land.lhs.true37
  %16 = and i32 %asmresult5.i, 402653184
  %.not317 = icmp eq i32 %16, 402653184
  br i1 %.not317, label %land.lhs.true223, label %if.else234

land.lhs.true223:                                 ; preds = %if.end219
  %17 = tail call { i32, i32 } asm sideeffect ".byte 0x0f, 0x01, 0xd0", "={ax},={dx},{cx},~{dirflag},~{fpsr},~{flags}"(i32 0) nounwind
  %asmresult.i300 = extractvalue { i32, i32 } %17, 0
  %and.i301 = and i32 %asmresult.i300, 6
  %cmp.i302.not = icmp eq i32 %and.i301, 6
  br i1 %cmp.i302.not, label %if.then226, label %if.else234

if.then226:                                       ; preds = %land.lhs.true223
  %18 = and i32 %asmresult5.i, 1610612736
  %19 = icmp ne i32 %18, 1610612736
  %brmerge277 = select i1 %19, i1 true, i1 %cmp16.not
  %spec.select281 = select i1 %brmerge277, i32 3, i32 4
  br label %cleanup244

if.else234:                                       ; preds = %if.end219, %land.lhs.true223
  %20 = and i32 %asmresult5.i, 1572864
  %21 = icmp ne i32 %20, 0
  %.mux283 = select i1 %cmp8.not.not, i32 1, i32 2
  %brmerge284 = select i1 %21, i1 true, i1 %cmp.not
  %.mux283.mux = select i1 %21, i32 %.mux283, i32 0
  br i1 %brmerge284, label %cleanup244, label %if.else243

if.else243:                                       ; preds = %if.else234
  tail call void @abort() noreturn nounwind
  unreachable

cleanup244:                                       ; preds = %land.lhs.true156, %if.end184, %if.then226, %if.else234, %cleanup
  %retval.1 = phi i32 [ 5, %cleanup ], [ %.mux283.mux, %if.else234 ], [ %spec.select281, %if.then226 ], [ %.mux.mux, %land.lhs.true156 ], [ %.mux319, %if.end184 ]
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

