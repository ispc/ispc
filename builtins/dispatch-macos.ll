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

define i32 @__get_system_isa() nounwind uwtable {
entry:
  %0 = tail call { i32, i32, i32, i32 } asm sideeffect "cpuid", "={ax},={bx},={cx},={dx},0,~{dirflag},~{fpsr},~{flags}"(i32 1) nounwind
  %asmresult5.i = extractvalue { i32, i32, i32, i32 } %0, 2
  %asmresult6.i = extractvalue { i32, i32, i32, i32 } %0, 3
  %1 = tail call { i32, i32, i32, i32 } asm sideeffect "cpuid", "={ax},={bx},={cx},={dx},0,2,~{dirflag},~{fpsr},~{flags}"(i32 7, i32 0) nounwind
  %asmresult4.i84 = extractvalue { i32, i32, i32, i32 } %1, 1
  %and = and i32 %asmresult5.i, 134217728
  %cmp = icmp eq i32 %and, 0
  br i1 %cmp, label %if.else61, label %land.lhs.true

land.lhs.true:                                    ; preds = %entry
  %2 = and i32 %asmresult4.i84, 65568
  %3 = icmp eq i32 %2, 65568
  br i1 %3, label %if.then, label %if.end38

if.then:                                          ; preds = %land.lhs.true
  %4 = and i32 %asmresult4.i84, -805175296
  %5 = icmp eq i32 %4, -805175296
  br i1 %5, label %cleanup, label %if.else

if.else:                                          ; preds = %if.then
  %6 = and i32 %asmresult4.i84, 469762048
  %7 = icmp eq i32 %6, 469762048
  br i1 %7, label %cleanup, label %if.end38

if.end38:                                         ; preds = %if.else, %land.lhs.true
  %8 = and i32 %asmresult5.i, 402653184
  %9 = icmp eq i32 %8, 402653184
  br i1 %9, label %land.lhs.true46, label %if.else61

land.lhs.true46:                                  ; preds = %if.end38
  %10 = tail call { i32, i32 } asm sideeffect ".byte 0x0f, 0x01, 0xd0", "={ax},={dx},{cx},~{dirflag},~{fpsr},~{flags}"(i32 0) nounwind
  %asmresult.i87 = extractvalue { i32, i32 } %10, 0
  %and.i = and i32 %asmresult.i87, 6
  %cmp.i = icmp eq i32 %and.i, 6
  br i1 %cmp.i, label %if.then47, label %if.else61

if.then47:                                        ; preds = %land.lhs.true46
  %11 = and i32 %asmresult5.i, 1610612736
  %12 = icmp ne i32 %11, 1610612736
  %and57 = and i32 %asmresult4.i84, 32
  %cmp58 = icmp eq i32 %and57, 0
  %or.cond104 = or i1 %12, %cmp58
  %spec.select = select i1 %or.cond104, i32 2, i32 3
  ret i32 %spec.select

if.else61:                                        ; preds = %land.lhs.true46, %if.end38, %entry
  %and63 = and i32 %asmresult5.i, 524288
  %cmp64 = icmp eq i32 %and63, 0
  br i1 %cmp64, label %if.else66, label %cleanup

if.else66:                                        ; preds = %if.else61
  %and68 = and i32 %asmresult6.i, 67108864
  %cmp69 = icmp eq i32 %and68, 0
  br i1 %cmp69, label %if.else71, label %cleanup

if.else71:                                        ; preds = %if.else66
  tail call void @abort() noreturn nounwind
  unreachable

cleanup:                                          ; preds = %if.else, %if.else66, %if.else61, %if.then
  %retval.0 = phi i32 [ 5, %if.then ], [ 4, %if.else ], [ 1, %if.else61 ], [ 0, %if.else66 ]
  ret i32 %retval.0
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

