;;  Copyright (c) 2011-2023, Intel Corporation
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
;; static int __os_has_avx512_support() {
;;     // Check if the OS saves the XMM, YMM and ZMM registers, i.e. it supports AVX2 and AVX512.
;;     // See section 2.1 of software.intel.com/sites/default/files/managed/0d/53/319433-022.pdf
;;     // Check xgetbv; this uses a .byte sequence instead of the instruction
;;     // directly because older assemblers do not include support for xgetbv and
;;     // there is no easy way to conditionally compile based on the assembler used.
;;     int rEAX, rEDX;
;;     __asm__ __volatile__ (".byte 0x0f, 0x01, 0xd0" : "=a" (rEAX), "=d" (rEDX) : "c" (0));
;;     return (rEAX & 0xE6) == 0xE6;
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
;;     // Call cpuid with eax=7, ecx=1
;;     int info3[4];
;;     __cpuid_count(info3, 7, 1);
;; 
;;     _Bool sse2 =                (info[3] & (1 << 26))  != 0;
;;     _Bool sse4 =                (info[2] & (1 << 19))  != 0;
;;     _Bool avx_f16c =            (info[2] & (1 << 29))  != 0;
;;     _Bool avx_rdrand =          (info[2] & (1 << 30))  != 0;
;;     _Bool osxsave =             (info[2] & (1 << 27))  != 0;
;;     _Bool avx =                 (info[2] & (1 << 28))  != 0;
;;     _Bool avx2 =                (info2[1] & (1 << 5))  != 0;
;;     _Bool avx512_f =            (info2[1] & (1 << 16)) != 0;
;;     _Bool avx512_dq =           (info2[1] & (1 << 17)) != 0;
;;     _Bool avx512_pf =           (info2[1] & (1 << 26)) != 0;
;;     _Bool avx512_er =           (info2[1] & (1 << 27)) != 0;
;;     _Bool avx512_cd =           (info2[1] & (1 << 28)) != 0;
;;     _Bool avx512_bw =           (info2[1] & (1 << 30)) != 0;
;;     _Bool avx512_vl =           (info2[1] & (1 << 31)) != 0;
;;     _Bool avx512_vbmi2 =        (info2[2] & (1 << 6))  != 0;
;;     _Bool avx512_gfni =         (info2[2] & (1 << 8))  != 0;
;;     _Bool avx512_vaes =         (info2[2] & (1 << 9))  != 0;
;;     _Bool avx512_vpclmulqdq =   (info2[2] & (1 << 10)) != 0;
;;     _Bool avx512_vnni =         (info2[2] & (1 << 11)) != 0;
;;     _Bool avx512_bitalg =       (info2[2] & (1 << 12)) != 0;
;;     _Bool avx512_vpopcntdq =    (info2[2] & (1 << 14)) != 0;
;;     _Bool avx_vnni =            (info3[0] & (1 << 4))  != 0;
;;     _Bool avx512_bf16 =         (info3[0] & (1 << 5))  != 0;
;;     _Bool avx512_vp2intersect = (info2[3] & (1 << 8))  != 0;
;;     _Bool avx512_amx_bf16 =     (info2[3] & (1 << 22)) != 0;
;;     _Bool avx512_amx_tile =     (info2[3] & (1 << 24)) != 0;
;;     _Bool avx512_amx_int8 =     (info2[3] & (1 << 25)) != 0;
;;     _Bool avx512_fp16 =         (info2[3] & (1 << 23)) != 0;
;; 
;;     // NOTE: the values returned below must be the same as the
;;     // corresponding enumerant values in Target::ISA.
;;     if (osxsave && avx2 && avx512_f && __os_has_avx512_support()) {
;;         // We need to verify that AVX2 is also available,
;;         // as well as AVX512, because our targets are supposed
;;         // to use both.
;; 
;;         // Knights Landing:          KNL = F + PF + ER + CD
;;         // Skylake server:           SKX = F + DQ + CD + BW + VL
;;         // Cascade Lake server:      CLX = SKX + VNNI
;;         // Cooper Lake server:       CPX = CLX + BF16
;;         // Ice Lake client & server: ICL = CLX + VBMI2 + GFNI + VAES + VPCLMULQDQ + BITALG + VPOPCNTDQ
;;         // Tiger Lake:               TGL = ICL + VP2INTERSECT
;;         // Sapphire Rapids:          SPR = ICL + BF16 + AMX_BF16 + AMX_TILE + AMX_INT8 + AVX_VNNI + FP16
;;         _Bool knl = avx512_pf && avx512_er && avx512_cd;
;;         _Bool skx = avx512_dq && avx512_cd && avx512_bw && avx512_vl;
;;         _Bool clx = skx && avx512_vnni;
;;         _Bool cpx = clx && avx512_bf16;
;;         _Bool icl =
;;             clx && avx512_vbmi2 && avx512_gfni && avx512_vaes && avx512_vpclmulqdq && avx512_bitalg && avx512_vpopcntdq;
;;         _Bool tgl = icl && avx512_vp2intersect;
;;         _Bool spr =
;;             icl && avx512_bf16 && avx512_amx_bf16 && avx512_amx_tile && avx512_amx_int8 && avx_vnni && avx512_fp16;
;;         if (spr) {
;;             return 6; // SPR
;;         } else if (skx) {
;;             return 5; // SKX
;;         } else if (knl) {
;;             return 4; // KNL
;;         }
;;         // If it's unknown AVX512 target, fall through and use AVX2
;;         // or whatever is available in the machine.
;;     }
;; 
;;     if (osxsave && avx && __os_has_avx_support()) {
;;         if (avx_f16c && avx_rdrand && avx2) {
;;             return 3;
;;         }
;;         // Regular AVX
;;         return 2;
;;     }
;;     else if (sse4)
;;         return 1; // SSE4
;;     else if (sse2)
;;         return 0; // SSE2
;;     else
;;         abort();
;; }


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
  %1 = tail call { i32, i32, i32, i32 } asm sideeffect "xchg$(l$)\09$(%$)ebx, $1\0A\09cpuid\0A\09xchg$(l$)\09$(%$)ebx, $1\0A\09", "={ax},=r,={cx},={dx},0,2,~{dirflag},~{fpsr},~{flags}"(i32 7, i32 0) nounwind
  %asmresult4.i277 = extractvalue { i32, i32, i32, i32 } %1, 1
  %asmresult5.i278 = extractvalue { i32, i32, i32, i32 } %1, 2
  %asmresult6.i279 = extractvalue { i32, i32, i32, i32 } %1, 3
  %2 = tail call { i32, i32, i32, i32 } asm sideeffect "xchg$(l$)\09$(%$)ebx, $1\0A\09cpuid\0A\09xchg$(l$)\09$(%$)ebx, $1\0A\09", "={ax},=r,={cx},={dx},0,2,~{dirflag},~{fpsr},~{flags}"(i32 7, i32 1) nounwind
  %asmresult.i283 = extractvalue { i32, i32, i32, i32 } %2, 0
  %and = and i32 %asmresult6.i, 67108864
  %cmp.not = icmp ne i32 %and, 0
  %and4 = and i32 %asmresult5.i, 524288
  %cmp5.not = icmp ne i32 %and4, 0
  %and16 = and i32 %asmresult5.i, 134217728
  %cmp17.not = icmp eq i32 %and16, 0
  %and24 = and i32 %asmresult4.i277, 32
  %cmp25.not = icmp eq i32 %and24, 0
  %and28 = and i32 %asmresult4.i277, 65536
  %cmp29.not = icmp eq i32 %and28, 0
  %and56 = and i32 %asmresult5.i278, 64
  %cmp57.not = icmp eq i32 %and56, 0
  %and60 = and i32 %asmresult5.i278, 256
  %cmp61.not = icmp eq i32 %and60, 0
  %and64 = and i32 %asmresult5.i278, 512
  %cmp65.not = icmp eq i32 %and64, 0
  %and68 = and i32 %asmresult5.i278, 1024
  %cmp69.not = icmp eq i32 %and68, 0
  %and72 = and i32 %asmresult5.i278, 2048
  %cmp73 = icmp eq i32 %and72, 0
  %and84 = and i32 %asmresult.i283, 16
  %cmp85.not = icmp ne i32 %and84, 0
  %and88 = and i32 %asmresult.i283, 32
  %cmp89.not = icmp ne i32 %and88, 0
  %and96 = and i32 %asmresult6.i279, 4194304
  %cmp97.not = icmp ne i32 %and96, 0
  %and100 = and i32 %asmresult6.i279, 16777216
  %cmp101.not = icmp ne i32 %and100, 0
  %and104 = and i32 %asmresult6.i279, 33554432
  %cmp105.not = icmp ne i32 %and104, 0
  %and108 = and i32 %asmresult6.i279, 8388608
  %cmp109 = icmp ne i32 %and108, 0
  %brmerge = select i1 %cmp17.not, i1 true, i1 %cmp25.not
  %brmerge250 = select i1 %brmerge, i1 true, i1 %cmp29.not
  br i1 %brmerge250, label %if.end190, label %land.lhs.true114

land.lhs.true114:                                 ; preds = %entry
  %3 = tail call { i32, i32 } asm sideeffect ".byte 0x0f, 0x01, 0xd0", "={ax},={dx},{cx},~{dirflag},~{fpsr},~{flags}"(i32 0) nounwind
  %asmresult.i287 = extractvalue { i32, i32 } %3, 0
  %and.i = and i32 %asmresult.i287, 230
  %cmp.i.not = icmp eq i32 %and.i, 230
  br i1 %cmp.i.not, label %if.then, label %if.end190

if.then:                                          ; preds = %land.lhs.true114
  %4 = and i32 %asmresult4.i277, 469762048
  %.not = icmp eq i32 %4, 469762048
  %5 = and i32 %asmresult4.i277, 268566528
  %.not295 = icmp eq i32 %5, 268566528
  %6 = icmp ugt i32 %asmresult4.i277, -1073741825
  %spec.select272 = and i1 %6, %.not295
  %not.spec.select272.demorgan = and i1 %6, %.not295
  %not.spec.select272 = xor i1 %not.spec.select272.demorgan, true
  %7 = select i1 %not.spec.select272, i1 true, i1 %cmp73
  %brmerge253 = select i1 %7, i1 true, i1 %cmp57.not
  %brmerge254 = select i1 %brmerge253, i1 true, i1 %cmp61.not
  %brmerge255 = select i1 %brmerge254, i1 true, i1 %cmp65.not
  %brmerge256 = select i1 %brmerge255, i1 true, i1 %cmp69.not
  br i1 %brmerge256, label %if.else, label %land.lhs.true149

land.lhs.true149:                                 ; preds = %if.then
  %8 = and i32 %asmresult5.i278, 20480
  %9 = icmp eq i32 %8, 20480
  %brmerge258 = select i1 %9, i1 %cmp89.not, i1 false
  %brmerge259 = select i1 %brmerge258, i1 %cmp97.not, i1 false
  %brmerge260 = select i1 %brmerge259, i1 %cmp101.not, i1 false
  %brmerge261 = select i1 %brmerge260, i1 %cmp105.not, i1 false
  %10 = select i1 %brmerge261, i1 %cmp85.not, i1 false
  %or.cond = select i1 %10, i1 %cmp109, i1 false
  %brmerge267 = select i1 %or.cond, i1 true, i1 %spec.select272
  %.mux = select i1 %or.cond, i32 6, i32 5
  %brmerge299 = select i1 %brmerge267, i1 true, i1 %.not
  %.mux.mux = select i1 %brmerge267, i32 %.mux, i32 4
  br i1 %brmerge299, label %cleanup212, label %if.end190

if.else:                                          ; preds = %if.then
  %brmerge297 = or i1 %.not, %spec.select272
  %.mux298 = select i1 %spec.select272, i32 5, i32 4
  br i1 %brmerge297, label %cleanup212, label %if.end190

if.end190:                                        ; preds = %land.lhs.true149, %if.else, %entry, %land.lhs.true114
  %11 = and i32 %asmresult5.i, 402653184
  %.not296 = icmp eq i32 %11, 402653184
  br i1 %.not296, label %land.lhs.true194, label %if.else205

land.lhs.true194:                                 ; preds = %if.end190
  %12 = tail call { i32, i32 } asm sideeffect ".byte 0x0f, 0x01, 0xd0", "={ax},={dx},{cx},~{dirflag},~{fpsr},~{flags}"(i32 0) nounwind
  %asmresult.i288 = extractvalue { i32, i32 } %12, 0
  %and.i289 = and i32 %asmresult.i288, 6
  %cmp.i290.not = icmp eq i32 %and.i289, 6
  br i1 %cmp.i290.not, label %if.then197, label %if.else205

if.then197:                                       ; preds = %land.lhs.true194
  %13 = and i32 %asmresult5.i, 1610612736
  %14 = icmp ne i32 %13, 1610612736
  %brmerge266 = select i1 %14, i1 true, i1 %cmp25.not
  %spec.select268 = select i1 %brmerge266, i32 2, i32 3
  br label %cleanup212

if.else205:                                       ; preds = %if.end190, %land.lhs.true194
  %brmerge269 = select i1 %cmp5.not, i1 true, i1 %cmp.not
  %and4.lobit = lshr exact i32 %and4, 19
  br i1 %brmerge269, label %cleanup212, label %if.else211

if.else211:                                       ; preds = %if.else205
  tail call void @abort() noreturn nounwind
  unreachable

cleanup212:                                       ; preds = %land.lhs.true149, %if.else, %if.then197, %if.else205
  %retval.1 = phi i32 [ %and4.lobit, %if.else205 ], [ %spec.select268, %if.then197 ], [ %.mux.mux, %land.lhs.true149 ], [ %.mux298, %if.else ]
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

