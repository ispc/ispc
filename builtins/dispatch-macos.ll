;;  Copyright (c) 2011-2020, Intel Corporation
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

;; MACOS VERSION OF DISPATCH: the key difference is absense of OS support check
;; for AVX512 - see issue #1854 for more details.

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
;;     // Call cpuid with eax=7, ecx=0
;;     int info2[4];
;;     __cpuid_count(info2, 7, 0);
;;
;;     // NOTE: the values returned below must be the same as the
;;     // corresponding enumerant values in Target::ISA.
;;     if ((info[2] & (1 << 27)) != 0 &&  // OSXSAVE
;;         (info2[1] & (1 <<  5)) != 0 && // AVX2
;;         (info2[1] & (1 << 16))) { // AVX512 F
;;         // We need to verify that AVX2 is also available,
;;         // as well as AVX512, because our targets are supposed
;;         // to use both.
;;
;;         if ((info2[1] & (1 << 17)) != 0 && // AVX512 DQ
;;             (info2[1] & (1 << 28)) != 0 && // AVX512 CDI
;;             (info2[1] & (1 << 30)) != 0 && // AVX512 BW
;;             (info2[1] & (1 << 31)) != 0) { // AVX512 VL
;;             return 5; // SKX
;;         }
;;         else if ((info2[1] & (1 << 26)) != 0 && // AVX512 PF
;;                  (info2[1] & (1 << 27)) != 0 && // AVX512 ER
;;                  (info2[1] & (1 << 28)) != 0) { // AVX512 CDI
;;             return 4; // KNL_AVX512
;;         }
;;         // If it's unknown AVX512 target, fall through and use AVX2
;;         // or whatever is available in the machine.
;;     }
;;
;;     if ((info[2] & (1 << 27)) != 0 && // OSXSAVE
;;         (info[2] & (1 << 28)) != 0 &&
;;         __os_has_avx_support()) {
;;        if ((info[2] & (1 << 29)) != 0 &&  // F16C
;;            (info[2] & (1 << 30)) != 0 &&  // RDRAND
;;            (info2[1] & (1 << 5)) != 0) {  // AVX2
;;            return 3;
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
  %1 = tail call { i32, i32, i32, i32 } asm sideeffect "xchg$(l$)\09$(%$)ebx, $1\0A\09cpuid\0A\09xchg$(l$)\09$(%$)ebx, $1\0A\09", "={ax},=r,={cx},={dx},0,2,~{dirflag},~{fpsr},~{flags}"(i32 7, i32 0) nounwind
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

