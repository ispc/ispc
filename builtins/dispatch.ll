;;  Copyright (c) 2011, Intel Corporation
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

declare void @abort() noreturn

;; The below is the result of running "clang -O2 -emit-llvm -c -o -" on the
;; following code...  Specifically, __get_system_isa should return a value
;; corresponding to one of the Target::ISA enumerant values that gives the
;; most capable ISA that the curremt system can run.
;;
;; Note: clang from LLVM 3.0 should be used if this is updated, for maximum
;; backwards compatibility for anyone building ispc with LLVM 3.0
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
;; /* Save %ebx in case it's the PIC register */
;; static void __cpuid_count(int info[4], int level, int count) {
;;   __asm__ __volatile__ ("xchg{l}\t{%%}ebx, %1\n\t"
;;                         "cpuid\n\t"
;;                         "xchg{l}\t{%%}ebx, %1\n\t"
;;                         : "=a" (info[0]), "=r" (info[1]), "=c" (info[2]), "=d" (info[3])
;;                         : "0" (level), "2" (count));
;; }
;; 
;; int32_t __get_system_isa() {
;;     int info[4];
;;     __cpuid(info, 1);
;; 
;;     /* NOTE: the values returned below must be the same as the
;;        corresponding enumerant values in Target::ISA. */
;;     if ((info[2] & (1 << 28)) != 0) {
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

define i32 @__get_system_isa() nounwind uwtable ssp {
entry:
  %0 = tail call { i32, i32, i32, i32 } asm sideeffect "cpuid", "={ax},={bx},={cx},={dx},0,~{dirflag},~{fpsr},~{flags}"(i32 1) nounwind
  %asmresult5.i = extractvalue { i32, i32, i32, i32 } %0, 2
  %asmresult6.i = extractvalue { i32, i32, i32, i32 } %0, 3
  %and = and i32 %asmresult5.i, 268435456
  %cmp = icmp eq i32 %and, 0
  br i1 %cmp, label %if.else13, label %if.then

if.then:                                          ; preds = %entry
  %1 = and i32 %asmresult5.i, 1610612736
  %2 = icmp eq i32 %1, 1610612736
  br i1 %2, label %if.then7, label %return

if.then7:                                         ; preds = %if.then
  %3 = tail call { i32, i32, i32, i32 } asm sideeffect "xchg$(l$)\09$(%$)ebx, $1\0A\09cpuid\0A\09xchg$(l$)\09$(%$)ebx, $1\0A\09", "={ax},=r,={cx},={dx},0,2,~{dirflag},~{fpsr},~{flags}"(i32 7, i32 0) nounwind
  %asmresult4.i28 = extractvalue { i32, i32, i32, i32 } %3, 1
  %and10 = lshr i32 %asmresult4.i28, 5
  %4 = and i32 %and10, 1
  %5 = add i32 %4, 3
  br label %return

if.else13:                                        ; preds = %entry
  %and15 = and i32 %asmresult5.i, 524288
  %cmp16 = icmp eq i32 %and15, 0
  br i1 %cmp16, label %if.else18, label %return

if.else18:                                        ; preds = %if.else13
  %and20 = and i32 %asmresult6.i, 67108864
  %cmp21 = icmp eq i32 %and20, 0
  br i1 %cmp21, label %if.else23, label %return

if.else23:                                        ; preds = %if.else18
  tail call void @abort() noreturn nounwind
  unreachable

return:                                           ; preds = %if.else18, %if.else13, %if.then7, %if.then
  %retval.0 = phi i32 [ %5, %if.then7 ], [ 2, %if.then ], [ 1, %if.else13 ], [ 0, %if.else18 ]
  ret i32 %retval.0
}

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

