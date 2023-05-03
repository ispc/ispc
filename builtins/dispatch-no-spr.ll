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

define i32 @__get_system_isa() nounwind uwtable {
entry:
  %0 = tail call { i32, i32, i32, i32 } asm sideeffect "cpuid", "={ax},={bx},={cx},={dx},0,~{dirflag},~{fpsr},~{flags}"(i32 1) nounwind
  %asmresult5.i = extractvalue { i32, i32, i32, i32 } %0, 2
  %asmresult6.i = extractvalue { i32, i32, i32, i32 } %0, 3
  %1 = tail call { i32, i32, i32, i32 } asm sideeffect "cpuid", "={ax},={bx},={cx},={dx},0,2,~{dirflag},~{fpsr},~{flags}"(i32 7, i32 0) nounwind
  %asmresult4.i87 = extractvalue { i32, i32, i32, i32 } %1, 1
  %and = and i32 %asmresult5.i, 134217728
  %cmp = icmp eq i32 %and, 0
  br i1 %cmp, label %if.else65, label %land.lhs.true

land.lhs.true:                                    ; preds = %entry
  %2 = and i32 %asmresult4.i87, 65568
  %3 = icmp eq i32 %2, 65568
  br i1 %3, label %land.lhs.true9, label %if.end39

land.lhs.true9:                                   ; preds = %land.lhs.true
  %4 = tail call { i32, i32 } asm sideeffect ".byte 0x0f, 0x01, 0xd0", "={ax},={dx},{cx},~{dirflag},~{fpsr},~{flags}"(i32 0) nounwind
  %asmresult.i90 = extractvalue { i32, i32 } %4, 0
  %and.i = and i32 %asmresult.i90, 230
  %cmp.i = icmp eq i32 %and.i, 230
  br i1 %cmp.i, label %if.then, label %if.end39

if.then:                                          ; preds = %land.lhs.true9
  %5 = and i32 %asmresult4.i87, -805175296
  %6 = icmp eq i32 %5, -805175296
  br i1 %6, label %return, label %if.else

if.else:                                          ; preds = %if.then
  %7 = and i32 %asmresult4.i87, 469762048
  %8 = icmp eq i32 %7, 469762048
  br i1 %8, label %return, label %if.end39

if.end39:                                         ; preds = %if.else, %land.lhs.true9, %land.lhs.true
  %9 = and i32 %asmresult5.i, 402653184
  %10 = icmp eq i32 %9, 402653184
  br i1 %10, label %land.lhs.true47, label %if.else65

land.lhs.true47:                                  ; preds = %if.end39
  %11 = tail call { i32, i32 } asm sideeffect ".byte 0x0f, 0x01, 0xd0", "={ax},={dx},{cx},~{dirflag},~{fpsr},~{flags}"(i32 0) nounwind
  %asmresult.i91 = extractvalue { i32, i32 } %11, 0
  %and.i92 = and i32 %asmresult.i91, 6
  %cmp.i93 = icmp eq i32 %and.i92, 6
  br i1 %cmp.i93, label %if.then50, label %if.else65

if.then50:                                        ; preds = %land.lhs.true47
  %12 = and i32 %asmresult5.i, 1610612736
  %13 = icmp ne i32 %12, 1610612736
  %and60 = and i32 %asmresult4.i87, 32
  %cmp61 = icmp eq i32 %and60, 0
  %or.cond112 = or i1 %13, %cmp61
  %spec.select = select i1 %or.cond112, i32 2, i32 3
  ret i32 %spec.select

if.else65:                                        ; preds = %land.lhs.true47, %if.end39, %entry
  %and67 = and i32 %asmresult5.i, 524288
  %cmp68 = icmp eq i32 %and67, 0
  br i1 %cmp68, label %if.else70, label %return

if.else70:                                        ; preds = %if.else65
  %and72 = and i32 %asmresult6.i, 67108864
  %cmp73 = icmp eq i32 %and72, 0
  br i1 %cmp73, label %if.else75, label %return

if.else75:                                        ; preds = %if.else70
  tail call void @abort() noreturn nounwind
  unreachable

return:                                          ; preds = %if.else, %if.else70, %if.else65, %if.then
  %retval.0 = phi i32 [ 5, %if.then ], [ 4, %if.else ], [ 1, %if.else65 ], [ 0, %if.else70 ]
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

