;;  Copyright (c) 2011-2013, Intel Corporation
;;  All rights reserved.
;;
;;  Redistribution and use in source and binary forms, with or without
;;  modification, are permitted provided that the following conditions are
;;  met:
;;
;;    * Redistributions of source code must retain the above copyright
;;      notice, this list of conditions and the following disclaimer.
;;
;;    * Redistributions in binary form must reproduce the above copyright
;;      notice, this list of conditions and the following disclaimer in the
;;      documentation and/or other materials provided with the distribution.
;;
;;    * Neither the name of Intel Corporation nor the names of its
;;      contributors may be used to endorse or promote products derived from
;;      this software without specific prior written permission.
;;
;;
;;   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
;;   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
;;   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
;;   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
;;   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
;;   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
;;   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
;;   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
;;   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
;;   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
;;   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  

;; This file defines various functions that are used when generating the
;; the "dispatch" object/assembly file that has entrypoints for each
;; exported function in a module that dispatch to the best available
;; variant of that function that will run on the system's CPU.

;; Stores the best target ISA that the system on which we're actually
;; running supports.  -1 represents "uninitialized", otherwise this value
;; should correspond to one of the enumerant values of Target::ISA from
;; ispc.h.

@__system_best_isa = internal global i32 -1

;; The below is the result of running "clang -O2 -emit-llvm -c -o -" on the
;; following code...  Specifically, __get_system_isa should return a value
;; corresponding to one of the Target::ISA enumerant values that gives the
;; most capable ISA that the curremt system can run.
;;
;;
;; #include <stdint.h>
;; #include <stdlib.h>
;; 
;; static void __cpuid(int info[4], int infoType) {
;;     __asm__ __volatile__ ("cpuid"
;;                           : "=a" (info[0]), "=b" (info[1]), "=c" (info[2]), "=d" (info[3])
;;                           : "0" (infoType));
;; }
;; 
;; // Save %ebx in case it's the PIC register.
;; static void __cpuid_count(int info[4], int level, int count) {
;;   __asm__ __volatile__ ("xchg{l}\t{%%}ebx, %1\n\t"
;;                         "cpuid\n\t"
;;                         "xchg{l}\t{%%}ebx, %1\n\t"
;;                         : "=a" (info[0]), "=r" (info[1]), "=c" (info[2]), "=d" (info[3])
;;                         : "0" (level), "2" (count));
;; }
;; 
;; static int __os_has_avx_support() {
;;     // Check xgetbv; this uses a .byte sequence instead of the instruction
;;     // directly because older assemblers do not include support for xgetbv and
;;     // there is no easy way to conditionally compile based on the assembler used.
;;     int rEAX, rEDX;
;;     __asm__ __volatile__ (".byte 0x0f, 0x01, 0xd0" : "=a" (rEAX), "=d" (rEDX) : "c" (0));
;;     return (rEAX & 6) == 6;
;; }
;; 
;; int32_t __get_system_isa() {
;;     int info[4];
;;     __cpuid(info, 1);
;; 
;;     // NOTE: the values returned below must be the same as the
;;     // corresponding enumerant values in Target::ISA.
;;     if ((info[2] & (1 << 28)) != 0 &&
;;         __os_has_avx_support()) {
;;        if ((info[2] & (1 << 29)) != 0 &&  // F16C
;;            (info[2] & (1 << 30)) != 0) {  // RDRAND
;;            // So far, so good.  AVX2?
;;            // Call cpuid with eax=7, ecx=0
;;            int info2[4];
;;            __cpuid_count(info2, 7, 0);
;;            if ((info2[1] & (1 << 5)) != 0)
;;                return 4;
;;            else
;;                return 3;
;;        }
;;        // Regular AVX
;;        return 2;
;;     }
;;     else if ((info[2] & (1 << 19)) != 0)
;;         return 1; // SSE4
;;     else if ((info[3] & (1 << 26)) != 0)
;;         return 0; // SSE2
;;     else
;;         abort();
;; }

define i32 @__get_system_isa() nounwind uwtable {
entry:
  %0 = tail call { i32, i32, i32, i32 } asm sideeffect "cpuid", "={ax},={bx},={cx},={dx},0,~{dirflag},~{fpsr},~{flags}"(i32 1) nounwind
  %asmresult5.i = extractvalue { i32, i32, i32, i32 } %0, 2
  %asmresult6.i = extractvalue { i32, i32, i32, i32 } %0, 3
  %and = and i32 %asmresult5.i, 268435456
  %cmp = icmp eq i32 %and, 0
  br i1 %cmp, label %if.else14, label %land.lhs.true

land.lhs.true:                                    ; preds = %entry
  %1 = tail call { i32, i32 } asm sideeffect ".byte 0x0f, 0x01, 0xd0", "={ax},={dx},{cx},~{dirflag},~{fpsr},~{flags}"(i32 0) nounwind
  %asmresult.i25 = extractvalue { i32, i32 } %1, 0
  %and.i = and i32 %asmresult.i25, 6
  %cmp.i = icmp eq i32 %and.i, 6
  br i1 %cmp.i, label %if.then, label %if.else14

if.then:                                          ; preds = %land.lhs.true
  %2 = and i32 %asmresult5.i, 1610612736
  %3 = icmp eq i32 %2, 1610612736
  br i1 %3, label %if.then8, label %return

if.then8:                                         ; preds = %if.then
  %4 = tail call { i32, i32, i32, i32 } asm sideeffect "xchg$(l$)\09$(%$)ebx, $1\0A\09cpuid\0A\09xchg$(l$)\09$(%$)ebx, $1\0A\09", "={ax},=r,={cx},={dx},0,2,~{dirflag},~{fpsr},~{flags}"(i32 7, i32 0) nounwind
  %asmresult4.i30 = extractvalue { i32, i32, i32, i32 } %4, 1
  %and11 = lshr i32 %asmresult4.i30, 5
  %5 = and i32 %and11, 1
  %6 = add i32 %5, 3
  br label %return

if.else14:                                        ; preds = %land.lhs.true, %entry
  %and16 = and i32 %asmresult5.i, 524288
  %cmp17 = icmp eq i32 %and16, 0
  br i1 %cmp17, label %if.else19, label %return

if.else19:                                        ; preds = %if.else14
  %and21 = and i32 %asmresult6.i, 67108864
  %cmp22 = icmp eq i32 %and21, 0
  br i1 %cmp22, label %if.else24, label %return

if.else24:                                        ; preds = %if.else19
  tail call void @abort() noreturn nounwind
  unreachable

return:                                           ; preds = %if.else19, %if.else14, %if.then8, %if.then
  %retval.0 = phi i32 [ %6, %if.then8 ], [ 2, %if.then ], [ 1, %if.else14 ], [ 0, %if.else19 ]
  ret i32 %retval.0
}

declare void @abort() noreturn nounwind

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; This function is called by each of the dispatch functions we generate;
;; it sets @__system_best_isa if it is unset.

define void @__set_system_isa() {
entry:
  %bi = load i32* @__system_best_isa
  %unset = icmp eq i32 %bi, -1
  br i1 %unset, label %set_system_isa, label %done

set_system_isa:
  %bival = call i32 @__get_system_isa()
  store i32 %bival, i32* @__system_best_isa
  ret void

done:
  ret void
}

