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

;; MACOS VERSION OF DISPATCH: the key difference is absense of OS support check
;; for AVX512 - see issue #1854 for more details.
;; This file is also not updated for ISAs newer than SKX, as no such Macs exist.

@__system_best_isa = internal global i32 -1

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

define(`PTR_OP_ARGS',
  `$1 , $1 *'
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

define i32 @__get_system_isa() local_unnamed_addr  nounwind uwtable {
entry:
  %0 = tail call { i32, i32, i32, i32 } asm sideeffect "cpuid", "={ax},={bx},={cx},={dx},0,~{dirflag},~{fpsr},~{flags}"(i32 1) nounwind
  %asmresult5.i = extractvalue { i32, i32, i32, i32 } %0, 2
  %asmresult6.i = extractvalue { i32, i32, i32, i32 } %0, 3
  %1 = tail call { i32, i32, i32, i32 } asm sideeffect "cpuid", "={ax},={bx},={cx},={dx},0,2,~{dirflag},~{fpsr},~{flags}"(i32 7, i32 0) nounwind
  %asmresult4.i147 = extractvalue { i32, i32, i32, i32 } %1, 1
  %and = and i32 %asmresult6.i, 67108864
  %cmp.not = icmp ne i32 %and, 0
  %and7 = and i32 %asmresult5.i, 1048576
  %cmp8.not.not = icmp eq i32 %and7, 0
  %and15 = and i32 %asmresult4.i147, 32
  %cmp16.not = icmp eq i32 %and15, 0
  %and27 = and i32 %asmresult5.i, 134217728
  %cmp28.not = icmp eq i32 %and27, 0
  %and31 = and i32 %asmresult4.i147, 65536
  %cmp32.not = icmp eq i32 %and31, 0
  %brmerge = select i1 %cmp28.not, i1 true, i1 %cmp16.not
  %brmerge132 = select i1 %brmerge, i1 true, i1 %cmp32.not
  br i1 %brmerge132, label %if.end94, label %if.then

if.then:                                          ; preds = %entry
  %asmresult.i146 = extractvalue { i32, i32, i32, i32 } %1, 0
  %cmp38 = icmp sgt i32 %asmresult.i146, 0
  br i1 %cmp38, label %if.then39, label %if.end

if.then39:                                        ; preds = %if.then
  %2 = tail call { i32, i32, i32, i32 } asm sideeffect "cpuid", "={ax},={bx},={cx},={dx},0,2,~{dirflag},~{fpsr},~{flags}"(i32 7, i32 1) nounwind
  br label %if.end

if.end:                                           ; preds = %if.then39, %if.then
  %3 = and i32 %asmresult4.i147, 469762048
  %4 = icmp ne i32 %3, 469762048
  %5 = and i32 %asmresult4.i147, 268566528
  %.not = icmp eq i32 %5, 268566528
  %6 = icmp ugt i32 %asmresult4.i147, -1073741825
  %or.cond139 = and i1 %6, %.not
  %retval.0 = select i1 %or.cond139, i32 6, i32 5
  %not.or.cond139.demorgan = and i1 %6, %.not
  %not.or.cond139 = xor i1 %not.or.cond139.demorgan, true
  %cond = select i1 %not.or.cond139, i1 %4, i1 false
  br i1 %cond, label %if.end94, label %cleanup118

if.end94:                                         ; preds = %entry, %if.end
  %7 = and i32 %asmresult5.i, 402653184
  %.not160 = icmp eq i32 %7, 402653184
  br i1 %.not160, label %land.lhs.true98, label %if.else108

land.lhs.true98:                                  ; preds = %if.end94
  %8 = tail call { i32, i32 } asm sideeffect ".byte 0x0f, 0x01, 0xd0", "={ax},={dx},{cx},~{dirflag},~{fpsr},~{flags}"(i32 0) nounwind
  %asmresult.i157 = extractvalue { i32, i32 } %8, 0
  %and.i = and i32 %asmresult.i157, 6
  %cmp.i.not = icmp eq i32 %and.i, 6
  br i1 %cmp.i.not, label %if.then100, label %if.else108

if.then100:                                       ; preds = %land.lhs.true98
  %9 = and i32 %asmresult5.i, 1610612736
  %10 = icmp ne i32 %9, 1610612736
  %brmerge137 = select i1 %10, i1 true, i1 %cmp16.not
  %spec.select140 = select i1 %brmerge137, i32 3, i32 4
  br label %cleanup118

if.else108:                                       ; preds = %if.end94, %land.lhs.true98
  %11 = and i32 %asmresult5.i, 1572864
  %12 = icmp ne i32 %11, 0
  %.mux = select i1 %cmp8.not.not, i32 1, i32 2
  %brmerge142 = select i1 %12, i1 true, i1 %cmp.not
  %.mux.mux = select i1 %12, i32 %.mux, i32 0
  br i1 %brmerge142, label %cleanup118, label %if.else117

if.else117:                                       ; preds = %if.else108
  tail call void @abort() noreturn nounwind
  unreachable

cleanup118:                                       ; preds = %if.then100, %if.else108, %if.end
  %retval.1 = phi i32 [ %retval.0, %if.end ], [ %.mux.mux, %if.else108 ], [ %spec.select140, %if.then100 ]
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

