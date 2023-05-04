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
  %asmresult.i149 = extractvalue { i32, i32, i32, i32 } %1, 0
  %asmresult4.i150 = extractvalue { i32, i32, i32, i32 } %1, 1
  %and = and i32 %asmresult6.i, 67108864
  %cmp.not = icmp ne i32 %and, 0
  %and7 = and i32 %asmresult5.i, 1048576
  %cmp8.not.not = icmp eq i32 %and7, 0
  %and15 = and i32 %asmresult4.i150, 32
  %cmp16.not = icmp eq i32 %and15, 0
  %and27 = and i32 %asmresult5.i, 134217728
  %cmp28.not = icmp eq i32 %and27, 0
  %and31 = and i32 %asmresult4.i150, 65536
  %cmp32.not = icmp eq i32 %and31, 0
  %brmerge = select i1 %cmp28.not, i1 true, i1 %cmp16.not
  %brmerge135 = select i1 %brmerge, i1 true, i1 %cmp32.not
  br i1 %brmerge135, label %if.end96, label %land.lhs.true37

land.lhs.true37:                                  ; preds = %entry
  %2 = tail call { i32, i32 } asm sideeffect ".byte 0x0f, 0x01, 0xd0", "={ax},={dx},{cx},~{dirflag},~{fpsr},~{flags}"(i32 0) nounwind
  %asmresult.i153 = extractvalue { i32, i32 } %2, 0
  %and.i = and i32 %asmresult.i153, 230
  %cmp.i.not = icmp eq i32 %and.i, 230
  br i1 %cmp.i.not, label %if.then, label %if.end96

if.then:                                          ; preds = %land.lhs.true37
  %cmp40 = icmp sgt i32 %asmresult.i149, 0
  br i1 %cmp40, label %if.then41, label %if.end

if.then41:                                        ; preds = %if.then
  %3 = tail call { i32, i32, i32, i32 } asm sideeffect "cpuid", "={ax},={bx},={cx},={dx},0,2,~{dirflag},~{fpsr},~{flags}"(i32 7, i32 1) nounwind
  br label %if.end

if.end:                                           ; preds = %if.then41, %if.then
  %4 = and i32 %asmresult4.i150, 469762048
  %5 = icmp ne i32 %4, 469762048
  %6 = and i32 %asmresult4.i150, 268566528
  %.not = icmp eq i32 %6, 268566528
  %7 = icmp ugt i32 %asmresult4.i150, -1073741825
  %or.cond142 = and i1 %7, %.not
  %retval.0 = select i1 %or.cond142, i32 6, i32 5
  %not.or.cond142.demorgan = and i1 %7, %.not
  %not.or.cond142 = xor i1 %not.or.cond142.demorgan, true
  %cond = select i1 %not.or.cond142, i1 %5, i1 false
  br i1 %cond, label %if.end96, label %cleanup121

if.end96:                                         ; preds = %entry, %if.end, %land.lhs.true37
  %8 = and i32 %asmresult5.i, 402653184
  %.not167 = icmp eq i32 %8, 402653184
  br i1 %.not167, label %land.lhs.true100, label %if.else111

land.lhs.true100:                                 ; preds = %if.end96
  %9 = tail call { i32, i32 } asm sideeffect ".byte 0x0f, 0x01, 0xd0", "={ax},={dx},{cx},~{dirflag},~{fpsr},~{flags}"(i32 0) nounwind
  %asmresult.i161 = extractvalue { i32, i32 } %9, 0
  %and.i162 = and i32 %asmresult.i161, 6
  %cmp.i163.not = icmp eq i32 %and.i162, 6
  br i1 %cmp.i163.not, label %if.then103, label %if.else111

if.then103:                                       ; preds = %land.lhs.true100
  %10 = and i32 %asmresult5.i, 1610612736
  %11 = icmp ne i32 %10, 1610612736
  %brmerge140 = select i1 %11, i1 true, i1 %cmp16.not
  %spec.select143 = select i1 %brmerge140, i32 3, i32 4
  br label %cleanup121

if.else111:                                       ; preds = %if.end96, %land.lhs.true100
  %12 = and i32 %asmresult5.i, 1572864
  %13 = icmp ne i32 %12, 0
  %.mux = select i1 %cmp8.not.not, i32 1, i32 2
  %brmerge145 = select i1 %13, i1 true, i1 %cmp.not
  %.mux.mux = select i1 %13, i32 %.mux, i32 0
  br i1 %brmerge145, label %cleanup121, label %if.else120

if.else120:                                       ; preds = %if.else111
  tail call void @abort() noreturn nounwind
  unreachable

cleanup121:                                       ; preds = %if.then103, %if.else111, %if.end
  %retval.1 = phi i32 [ %retval.0, %if.end ], [ %.mux.mux, %if.else111 ], [ %spec.select143, %if.then103 ]
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

