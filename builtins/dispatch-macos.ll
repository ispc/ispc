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

;; MACOS VERSION OF DISPATCH: the key difference is absense of OS support check
;; for AVX512 - see issue #1854 for more details.
;; This file is also not updated for ISAs newer than SKX, as no such Macs exist.

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
  %asmresult.i154 = extractvalue { i32, i32, i32, i32 } %1, 0
  %asmresult4.i155 = extractvalue { i32, i32, i32, i32 } %1, 1
  %cmp = icmp sgt i32 %asmresult.i154, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %2 = tail call { i32, i32, i32, i32 } asm sideeffect "cpuid", "={ax},={bx},={cx},={dx},0,2,~{dirflag},~{fpsr},~{flags}"(i32 7, i32 1) nounwind
  %asmresult.i161 = extractvalue { i32, i32, i32, i32 } %2, 0
  %3 = and i32 %asmresult.i161, 16
  %4 = icmp eq i32 %3, 0
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %info3.sroa.0.0 = phi i1 [ %4, %if.then ], [ true, %entry ]
  %and = and i32 %asmresult6.i, 67108864
  %cmp4.not = icmp ne i32 %and, 0
  %and10 = and i32 %asmresult5.i, 1048576
  %cmp11.not.not = icmp eq i32 %and10, 0
  %and18 = and i32 %asmresult4.i155, 32
  %cmp19.not = icmp eq i32 %and18, 0
  %and34 = and i32 %asmresult5.i, 134217728
  %cmp35.not = icmp eq i32 %and34, 0
  %and38 = and i32 %asmresult4.i155, 65536
  %cmp39.not = icmp eq i32 %and38, 0
  %brmerge = select i1 %cmp35.not, i1 true, i1 %cmp19.not
  %brmerge140 = select i1 %brmerge, i1 true, i1 %cmp39.not
  br i1 %brmerge140, label %if.end96, label %if.then44

if.then44:                                        ; preds = %if.end
  %5 = and i32 %asmresult4.i155, 469762048
  %spec.select146 = icmp ne i32 %5, 469762048
  %6 = and i32 %asmresult4.i155, 268566528
  %brmerge141.not167 = icmp ne i32 %6, 268566528
  %spec.select = icmp ult i32 %asmresult4.i155, -1073741824
  %or.cond147.not = or i1 %spec.select, %brmerge141.not167
  %retval.0 = select i1 %or.cond147.not, i32 6, i32 7
  %cond = and i1 %spec.select146, %or.cond147.not
  br i1 %cond, label %if.end96, label %cleanup123

if.end96:                                         ; preds = %if.end, %if.then44
  %7 = and i32 %asmresult5.i, 402653184
  %brmerge143.not = icmp eq i32 %7, 402653184
  br i1 %brmerge143.not, label %land.lhs.true100, label %if.else113

land.lhs.true100:                                 ; preds = %if.end96
  %8 = tail call { i32, i32 } asm sideeffect ".byte 0x0f, 0x01, 0xd0", "={ax},={dx},{cx},~{dirflag},~{fpsr},~{flags}"(i32 0) nounwind
  %asmresult.i165 = extractvalue { i32, i32 } %8, 0
  %and.i = and i32 %asmresult.i165, 6
  %cmp.i.not = icmp eq i32 %and.i, 6
  br i1 %cmp.i.not, label %if.then102, label %if.else113

if.then102:                                       ; preds = %land.lhs.true100
  br i1 %info3.sroa.0.0, label %if.end105, label %cleanup123

if.end105:                                        ; preds = %if.then102
  %9 = and i32 %asmresult5.i, 1610612736
  %brmerge144 = icmp ne i32 %9, 1610612736
  %brmerge145 = select i1 %brmerge144, i1 true, i1 %cmp19.not
  %spec.select148 = select i1 %brmerge145, i32 3, i32 4
  br label %cleanup123

if.else113:                                       ; preds = %if.end96, %land.lhs.true100
  %10 = and i32 %asmresult5.i, 1572864
  %brmerge149 = icmp ne i32 %10, 0
  %.mux = select i1 %cmp11.not.not, i32 1, i32 2
  %brmerge150 = select i1 %brmerge149, i1 true, i1 %cmp4.not
  %.mux.mux = select i1 %brmerge149, i32 %.mux, i32 0
  br i1 %brmerge150, label %cleanup123, label %if.else122

if.else122:                                       ; preds = %if.else113
  tail call void @abort() #3
  unreachable

cleanup123:                                       ; preds = %if.end105, %if.else113, %if.then102, %if.then44
  %retval.1 = phi i32 [ %retval.0, %if.then44 ], [ 5, %if.then102 ], [ %.mux.mux, %if.else113 ], [ %spec.select148, %if.end105 ]
  ret i32 %retval.1
}

declare void @abort() noreturn nounwind

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

