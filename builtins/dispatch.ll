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
;; #ifdef _MSC_VER
;; extern void __stdcall __cpuid(int info[4], int infoType);
;; #else
;; static void __cpuid(int info[4], int infoType) {
;;     __asm__ __volatile__ ("cpuid"
;;                           : "=a" (info[0]), "=b" (info[1]), "=c" (info[2]), "=d" (info[3])
;;                           : "0" (infoType));
;; }
;; #endif
;; 
;; int32_t __get_system_isa() {
;;     int info[4];
;;     __cpuid(info, 1);
;;     /* NOTE: the values returned below must be the same as the
;;        corresponding enumerant values in Target::ISA. */
;;     if ((info[2] & (1 << 28)) != 0)
;;         return 2; // AVX
;;     else if ((info[2] & (1 << 19)) != 0)
;;         return 1; // SSE4
;;     else if ((info[3] & (1 << 26)) != 0)
;;         return 0; // SSE2
;;     else
;;         abort();
;; }

%0 = type { i32, i32, i32, i32 }

define i32 @__get_system_isa() nounwind ssp {
  %1 = tail call %0 asm sideeffect "cpuid", "={ax},={bx},={cx},={dx},0,~{dirflag},~{fpsr},~{flags}"(i32 1) nounwind
  %2 = extractvalue %0 %1, 2
  %3 = extractvalue %0 %1, 3
  %4 = and i32 %2, 268435456
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %6, label %13

; <label>:6                                       ; preds = %0
  %7 = and i32 %2, 524288
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %9, label %13

; <label>:9                                       ; preds = %6
  %10 = and i32 %3, 67108864
  %11 = icmp eq i32 %10, 0
  br i1 %11, label %12, label %13

; <label>:12                                      ; preds = %9
  tail call void @abort() noreturn nounwind
  unreachable

; <label>:13                                      ; preds = %9, %6, %0
  %.0 = phi i32 [ 2, %0 ], [ 1, %6 ], [ 0, %9 ]
  ret i32 %.0
}


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

