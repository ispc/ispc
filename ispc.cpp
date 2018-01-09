/*
  Copyright (c) 2010-2016, Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.


   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/** @file ispc.cpp
    @brief ispc global definitions
*/

#include "ispc.h"
#include "module.h"
#include "util.h"
#include "llvmutil.h"
#include <stdio.h>
#include <sstream>
#include <stdarg.h>     /* va_list, va_start, va_arg, va_end */
#ifdef ISPC_IS_WINDOWS
  #include <windows.h>
  #include <direct.h>
  #define strcasecmp stricmp
#else
  #include <sys/types.h>
  #include <unistd.h>
#endif
#if ISPC_LLVM_VERSION == ISPC_LLVM_3_2
  #include <llvm/LLVMContext.h>
  #include <llvm/Module.h>
  #include <llvm/Instructions.h>
#else /* 3.3+ */
  #include <llvm/IR/LLVMContext.h>
  #include <llvm/IR/Module.h>
  #include <llvm/IR/Instructions.h>
#endif
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_6 // LLVM 3.6+
  #if ISPC_LLVM_VERSION >= ISPC_LLVM_6_0
    #include <llvm/CodeGen/TargetSubtargetInfo.h>
    #include <llvm/CodeGen/TargetLowering.h>
  #else
    #include <llvm/Target/TargetSubtargetInfo.h>
    #if ISPC_LLVM_VERSION >= ISPC_LLVM_3_7 // LLVM 3.7+
      #include <llvm/Target/TargetLowering.h>
    #endif
  #endif
#endif
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_5 // LLVM 3.5+
  #include <llvm/IR/DebugInfo.h>
  #include <llvm/IR/DIBuilder.h>
#else // LLVM 3.2, 3.3, 3.4
  #include <llvm/DebugInfo.h>
  #include <llvm/DIBuilder.h>
#endif
#if ISPC_LLVM_VERSION >= ISPC_LLVM_5_0 // LLVM 5.0+
  #include <llvm/BinaryFormat/Dwarf.h>
#else // LLVM up to 4.x
  #include <llvm/Support/Dwarf.h>
#endif
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#if ISPC_LLVM_VERSION == ISPC_LLVM_3_2
  #include <llvm/DataLayout.h>
#else // LLVM 3.3+
  #include <llvm/IR/DataLayout.h>
  #include <llvm/IR/Attributes.h>
#endif
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/Host.h>

Globals *g;
Module *m;

///////////////////////////////////////////////////////////////////////////
// Target

#if !defined(ISPC_IS_WINDOWS) && !defined(__arm__)
static void __cpuid(int info[4], int infoType) {
    __asm__ __volatile__ ("cpuid"
                          : "=a" (info[0]), "=b" (info[1]), "=c" (info[2]), "=d" (info[3])
                          : "0" (infoType));
}

/* Save %ebx in case it's the PIC register */
static void __cpuidex(int info[4], int level, int count) {
  __asm__ __volatile__ ("xchg{l}\t{%%}ebx, %1\n\t"
                        "cpuid\n\t"
                        "xchg{l}\t{%%}ebx, %1\n\t"
                        : "=a" (info[0]), "=r" (info[1]), "=c" (info[2]), "=d" (info[3])
                        : "0" (level), "2" (count));
}
#endif // !ISPC_IS_WINDOWS && !__ARM__

#if !defined(__arm__)
static bool __os_has_avx_support() {
#if defined(ISPC_IS_WINDOWS)
    // Check if the OS will save the YMM registers
    unsigned long long xcrFeatureMask = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
    return (xcrFeatureMask & 6) == 6;
#else // !defined(ISPC_IS_WINDOWS)
    // Check xgetbv; this uses a .byte sequence instead of the instruction
    // directly because older assemblers do not include support for xgetbv and
    // there is no easy way to conditionally compile based on the assembler used.
    int rEAX, rEDX;
    __asm__ __volatile__ (".byte 0x0f, 0x01, 0xd0" : "=a" (rEAX), "=d" (rEDX) : "c" (0));
    return (rEAX & 6) == 6;
#endif // !defined(ISPC_IS_WINDOWS)
}

static bool __os_has_avx512_support() {
#if defined(ISPC_IS_WINDOWS)
    // Check if the OS saves the XMM, YMM and ZMM registers, i.e. it supports AVX2 and AVX512.
    // See section 2.1 of software.intel.com/sites/default/files/managed/0d/53/319433-022.pdf
    unsigned long long xcrFeatureMask = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
    return (xcrFeatureMask & 0xE6) == 0xE6;
#else // !defined(ISPC_IS_WINDOWS)
    // Check xgetbv; this uses a .byte sequence instead of the instruction
    // directly because older assemblers do not include support for xgetbv and
    // there is no easy way to conditionally compile based on the assembler used.
    int rEAX, rEDX;
    __asm__ __volatile__ (".byte 0x0f, 0x01, 0xd0" : "=a" (rEAX), "=d" (rEDX) : "c" (0));
    return (rEAX & 0xE6) == 0xE6;
#endif // !defined(ISPC_IS_WINDOWS)
}
#endif // !__arm__

static const char *
lGetSystemISA() {
#ifdef __arm__
    return "neon-i32x4";
#else
    int info[4];
    __cpuid(info, 1);

    int info2[4];
    // Call cpuid with eax=7, ecx=0
    __cpuidex(info2, 7, 0);

    if ((info[2] & (1 << 27)) != 0 &&  // OSXSAVE
        (info2[1] & (1 <<  5)) != 0 && // AVX2
        (info2[1] & (1 << 16)) != 0 && // AVX512 F
        __os_has_avx512_support()) {
        // We need to verify that AVX2 is also available,
        // as well as AVX512, because our targets are supposed
        // to use both.

        if ((info2[1] & (1 << 17)) != 0 && // AVX512 DQ
            (info2[1] & (1 << 28)) != 0 && // AVX512 CDI
            (info2[1] & (1 << 30)) != 0 && // AVX512 BW
            (info2[1] & (1 << 31)) != 0) { // AVX512 VL
            return "avx512skx-i32x16";
        }
        else if ((info2[1] & (1 << 26)) != 0 && // AVX512 PF
                 (info2[1] & (1 << 27)) != 0 && // AVX512 ER
                 (info2[1] & (1 << 28)) != 0) { // AVX512 CDI
            return "avx512knl-i32x16";
        }
        // If it's unknown AVX512 target, fall through and use AVX2
        // or whatever is available in the machine.
    }

    if ((info[2] & (1 << 27)) != 0 && // OSXSAVE
        (info[2] & (1 << 28)) != 0 &&
         __os_has_avx_support()) {  // AVX
        // AVX1 for sure....
        // Ivy Bridge?
        if ((info[2] & (1 << 29)) != 0 &&  // F16C
            (info[2] & (1 << 30)) != 0) {  // RDRAND
            // So far, so good.  AVX2?
            if ((info2[1] & (1 << 5)) != 0)
                return "avx2-i32x8";
            else
                return "avx1.1-i32x8";
        }
        // Regular AVX
        return "avx1-i32x8";
    }
    else if ((info[2] & (1 << 19)) != 0)
        return "sse4-i32x4";
    else if ((info[3] & (1 << 26)) != 0)
        return "sse2-i32x4";
    else {
        Error(SourcePos(), "Unable to detect supported SSE/AVX ISA.  Exiting.");
        exit(1);
    }
#endif
}


typedef enum {
    // Special value, indicates that no CPU is present.
    CPU_None = 0,

    // 'Generic' CPU without any hardware SIMD capabilities.
    CPU_Generic = 1,

    // Early Atom CPU. Supports SSSE3.
    CPU_Bonnell,

    // Generic Core2-like. Supports SSSE3. Isn`t quite compatible with Bonnell,
    // but for ISPC the difference is negligible; ISPC doesn`t make use of it.
    CPU_Core2,

    // Core2 Solo/Duo/Quad/Extreme. Supports SSE 4.1 (but not 4.2).
    CPU_Penryn,

    // Late Core2-like. Supports SSE 4.2 + POPCNT/LZCNT.
    CPU_Nehalem,

    // Sandy Bridge. Supports AVX 1.
    CPU_SandyBridge,

    // Ivy Bridge. Supports AVX 1 + RDRAND.
    CPU_IvyBridge,

    // Haswell. Supports AVX 2.
    CPU_Haswell,

#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_6 // LLVM 3.6+
    // Broadwell. Supports AVX 2 + ADX/RDSEED/SMAP.
    CPU_Broadwell,
#endif

#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_7 // LLVM 3.7+
    // Knights Landing - Xeon Phi.
    // Supports AVX-512F: All the key AVX-512 features: masking, broadcast... ;
    //          AVX-512CDI: Conflict Detection;
    //          AVX-512ERI & PRI: 28-bit precision RCP, RSQRT and EXP transcendentals,
    //                            new prefetch instructions.
    CPU_KNL,
#endif

#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_8 // LLVM 3.8+
    // Skylake Xeon.
    // Supports AVX-512F: All the key AVX-512 features: masking, broadcast... ;
    //          AVX-512CDI: Conflict Detection;
    //          AVX-512VL: Vector Length Orthogonality;
    //          AVX-512DQ: New HPC ISA (vs AVX512F);
    //          AVX-512BW: Byte and Word Support.
    CPU_SKX,
#endif

#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_4 // LLVM 3.4+
    // Late Atom-like design. Supports SSE 4.2 + POPCNT/LZCNT.
    CPU_Silvermont,
#endif

    // FIXME: LLVM supports a ton of different ARM CPU variants--not just
    // cortex-a9 and a15.  We should be able to handle any of them that also
    // have NEON support.
#ifdef ISPC_ARM_ENABLED
    // ARM Cortex A15. Supports NEON VFPv4.
    CPU_CortexA15,

    // ARM Cortex A9. Supports NEON VFPv3.
    CPU_CortexA9,
#endif

#ifdef ISPC_NVPTX_ENABLED
    // NVidia CUDA-compatible SM-35 architecture.
    CPU_SM35,
#endif

    sizeofCPUtype
} CPUtype;


class AllCPUs {
private:
    std::vector<std::vector<std::string> > names;
    std::vector<std::set<CPUtype> > compat;

    std::set<CPUtype> Set(int type, ...) {
        std::set<CPUtype> retn;
        va_list args;

        retn.insert((CPUtype)type);
        va_start(args, type);
        while ((type = va_arg(args, int)) != CPU_None)
            retn.insert((CPUtype)type);
        va_end(args);

        return retn;
    }

public:
    AllCPUs() {
        names = std::vector<std::vector<std::string> >(sizeofCPUtype);
        compat = std::vector<std::set<CPUtype> >(sizeofCPUtype);

        names[CPU_None].push_back("");

        names[CPU_Generic].push_back("generic");

        names[CPU_Bonnell].push_back("atom");
        names[CPU_Bonnell].push_back("bonnell");

        names[CPU_Core2].push_back("core2");

        names[CPU_Penryn].push_back("penryn");

#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_4 // LLVM 3.4+
        names[CPU_Silvermont].push_back("slm");
        names[CPU_Silvermont].push_back("silvermont");
#endif

        names[CPU_Nehalem].push_back("corei7");
        names[CPU_Nehalem].push_back("nehalem");

        names[CPU_SandyBridge].push_back("corei7-avx");
        names[CPU_SandyBridge].push_back("sandybridge");

        names[CPU_IvyBridge].push_back("core-avx-i");
        names[CPU_IvyBridge].push_back("ivybridge");

        names[CPU_Haswell].push_back("core-avx2");
        names[CPU_Haswell].push_back("haswell");

#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_6 // LLVM 3.6+
        names[CPU_Broadwell].push_back("broadwell");
#endif

#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_7 // LLVM 3.7+
         names[CPU_KNL].push_back("knl");
#endif

#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_8 // LLVM 3.8+
         names[CPU_SKX].push_back("skx");
#endif

#ifdef ISPC_ARM_ENABLED
        names[CPU_CortexA15].push_back("cortex-a15");

        names[CPU_CortexA9].push_back("cortex-a9");
#endif

#ifdef ISPC_NVPTX_ENABLED
        names[CPU_SM35].push_back("sm_35");
#endif


#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_3 // LLVM 3.2 or 3.3
        #define CPU_Silvermont CPU_Nehalem
#else /* LLVM 3.4+ */
        compat[CPU_Silvermont]  = Set(CPU_Generic, CPU_Bonnell, CPU_Penryn,
                                      CPU_Core2, CPU_Nehalem, CPU_Silvermont,
                                      CPU_None);
#endif

#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_7 // LLVM 3.7+
        compat[CPU_KNL]         = Set(CPU_KNL, CPU_Generic, CPU_Bonnell, CPU_Penryn,
                                      CPU_Core2, CPU_Nehalem, CPU_Silvermont,
                                      CPU_SandyBridge, CPU_IvyBridge,
                                      CPU_Haswell, CPU_Broadwell, CPU_None);
#endif

#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_8 // LLVM 3.8+
        compat[CPU_SKX]         = Set(CPU_SKX, CPU_Bonnell, CPU_Penryn,
                                      CPU_Core2, CPU_Nehalem, CPU_Silvermont,
                                      CPU_SandyBridge, CPU_IvyBridge,
                                      CPU_Haswell, CPU_Broadwell, CPU_None);
#endif

#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_5 // LLVM 3.2, 3.3, 3.4 or 3.5
        #define CPU_Broadwell CPU_Haswell
#else /* LLVM 3.6+ */
        compat[CPU_Broadwell]   = Set(CPU_Generic, CPU_Bonnell, CPU_Penryn,
                                      CPU_Core2, CPU_Nehalem, CPU_Silvermont,
                                      CPU_SandyBridge, CPU_IvyBridge,
                                      CPU_Haswell, CPU_Broadwell, CPU_None);
#endif
        compat[CPU_Haswell]     = Set(CPU_Generic, CPU_Bonnell, CPU_Penryn,
                                      CPU_Core2, CPU_Nehalem, CPU_Silvermont,
                                      CPU_SandyBridge, CPU_IvyBridge,
                                      CPU_Haswell, CPU_Broadwell, CPU_None);
        compat[CPU_IvyBridge]   = Set(CPU_Generic, CPU_Bonnell, CPU_Penryn,
                                      CPU_Core2, CPU_Nehalem, CPU_Silvermont,
                                      CPU_SandyBridge, CPU_IvyBridge,
                                      CPU_None);
        compat[CPU_SandyBridge] = Set(CPU_Generic, CPU_Bonnell, CPU_Penryn,
                                      CPU_Core2, CPU_Nehalem, CPU_Silvermont,
                                      CPU_SandyBridge, CPU_None);
        compat[CPU_Nehalem]     = Set(CPU_Generic, CPU_Bonnell, CPU_Penryn,
                                      CPU_Core2, CPU_Nehalem, CPU_Silvermont,
                                      CPU_None);
        compat[CPU_Penryn]      = Set(CPU_Generic, CPU_Bonnell, CPU_Penryn,
                                      CPU_Core2, CPU_Nehalem, CPU_Silvermont,
                                      CPU_None);
        compat[CPU_Core2]       = Set(CPU_Generic, CPU_Bonnell, CPU_Core2,
                                      CPU_None);
        compat[CPU_Bonnell]     = Set(CPU_Generic, CPU_Bonnell, CPU_Core2,
                                      CPU_None);
        compat[CPU_Generic]     = Set(CPU_Generic, CPU_None);

#ifdef ISPC_ARM_ENABLED
        compat[CPU_CortexA15]   = Set(CPU_Generic, CPU_CortexA9, CPU_CortexA15,
                                      CPU_None);
        compat[CPU_CortexA9]    = Set(CPU_Generic, CPU_CortexA9, CPU_None);
#endif

#ifdef ISPC_NVPTX_ENABLED
        compat[CPU_SM35]        = Set(CPU_Generic, CPU_SM35, CPU_None);
#endif
    }

    std::string HumanReadableListOfNames() {
        std::stringstream CPUs;
        for (int i = CPU_Generic; i < sizeofCPUtype; i++) {
            CPUs << names[i][0];
            if (names[i].size() > 1) {
                CPUs << " (synonyms: " << names[i][1];
                for (int j = 2, je = names[i].size(); j < je; j++)
                    CPUs << ", " << names[i][j];
                CPUs << ")";
            }
            if (i < sizeofCPUtype - 1)
                CPUs << ", ";
        }
        return CPUs.str();
    }

    std::string &GetDefaultNameFromType(CPUtype type) {
        Assert((type >= CPU_None) && (type < sizeofCPUtype));
        return names[type][0];
    }

    CPUtype GetTypeFromName(std::string name) {
        CPUtype retn = CPU_None;

        for (int i = 1; (retn == CPU_None) && (i < sizeofCPUtype); i++)
            for (int j = 0, je = names[i].size();
                (retn == CPU_None) && (j < je); j++)
                if (!name.compare(names[i][j]))
                    retn = (CPUtype)i;
        return retn;
    }

    bool BackwardCompatible(CPUtype what, CPUtype with) {
        Assert((what > CPU_None) && (what < sizeofCPUtype));
        Assert((with > CPU_None) && (with < sizeofCPUtype));
        return compat[what].find(with) != compat[what].end();
    }
};


Target::Target(const char *arch, const char *cpu, const char *isa, bool pic, bool printTarget, std::string genericAsSmth) :
    m_target(NULL),
    m_targetMachine(NULL),
    m_dataLayout(NULL),
    m_valid(false),
    m_isa(SSE2),
    m_treatGenericAsSmth(genericAsSmth),
    m_arch(""),
    m_is32Bit(true),
    m_cpu(""),
    m_attributes(""),
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_3 
    m_tf_attributes(NULL),
#endif
    m_nativeVectorWidth(-1),
    m_nativeVectorAlignment(-1),
    m_dataTypeWidth(-1),
    m_vectorWidth(-1),
    m_generatePIC(pic),
    m_maskingIsFree(false),
    m_maskBitCount(-1),
    m_hasHalf(false),
    m_hasRand(false),
    m_hasGather(false),
    m_hasScatter(false),
    m_hasTranscendentals(false),
    m_hasTrigonometry(false),
    m_hasRsqrtd(false),
    m_hasRcpd(false),
    m_hasVecPrefetch(false)
{
    CPUtype CPUID = CPU_None, CPUfromISA = CPU_None;
    AllCPUs a;

    if (cpu) {
        CPUID = a.GetTypeFromName(cpu);
        if (CPUID == CPU_None) {
            Error(SourcePos(), "Error: CPU type \"%s\" unknown. Supported"
                  " CPUs: %s.", cpu, a.HumanReadableListOfNames().c_str());
            return;
        }
    }

    if (isa == NULL) {
        // If a CPU was specified explicitly, try to pick the best
        // possible ISA based on that.
        switch (CPUID) {
            case CPU_None:
                // No CPU and no ISA, so use system info to figure out
                // what this CPU supports.
                isa = lGetSystemISA();
                Warning(SourcePos(), "No --target specified on command-line."
                        " Using default system target \"%s\".", isa);
                break;

            case CPU_Generic:
                isa = "generic-1";
                break;

#ifdef ISPC_NVPTX_ENABLED
            case CPU_SM35:
                isa = "nvptx";
                break;
#endif

#ifdef ISPC_ARM_ENABLED
            case CPU_CortexA9:
            case CPU_CortexA15:
                isa = "neon-i32x4";
                break;
#endif

#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_7 // LLVM 3.7+
            case CPU_KNL:
                isa = "avx512knl-i32x16";
                break;
#endif

#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_8 // LLVM 3.8+
            case CPU_SKX:
                isa = "avx512skx-i32x16";
                break;
#endif

#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_6
            case CPU_Broadwell:
#endif
            case CPU_Haswell:
                isa = "avx2-i32x8";
                break;

            case CPU_IvyBridge:
                isa = "avx1.1-i32x8";
                break;

            case CPU_SandyBridge:
                isa = "avx1-i32x8";
                break;

            // Penryn is here because ISPC does not use SSE 4.2
            case CPU_Penryn:
            case CPU_Nehalem:
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_4
            case CPU_Silvermont:
#endif
                isa = "sse4-i32x4";
                break;

            default:
                isa = "sse2-i32x4";
                break;
        }
        if (CPUID != CPU_None)
            Warning(SourcePos(), "No --target specified on command-line."
                    " Using ISA \"%s\" based on specified CPU \"%s\".",
                    isa, cpu);
    }

    if (!strcasecmp(isa, "host")) {
        isa = lGetSystemISA();
    }

    if (arch == NULL) {
#ifdef ISPC_ARM_ENABLED
        if (!strncmp(isa, "neon", 4))
            arch = "arm";
        else
#endif
#ifdef ISPC_NVPTX_ENABLED
         if(!strncmp(isa, "nvptx", 5))
           arch = "nvptx64";
         else
#endif /* ISPC_NVPTX_ENABLED */
            arch = "x86-64";
    }

    // Define arch alias
    if (std::string(arch) == "x86_64")
        arch = "x86-64";

    bool error = false;

    // Make sure the target architecture is a known one; print an error
    // with the valid ones otherwise.
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_7 // LLVM 3.7+
    for (llvm::TargetRegistry::iterator iter = llvm::TargetRegistry::targets().begin();
         iter != llvm::TargetRegistry::targets().end(); ++iter) {
#else
    for (llvm::TargetRegistry::iterator iter = llvm::TargetRegistry::begin();
         iter != llvm::TargetRegistry::end(); ++iter) {
#endif
        if (std::string(arch) == iter->getName()) {
            this->m_target = &*iter;
            break;
        }
    }
    if (this->m_target == NULL) {
        fprintf(stderr, "Invalid architecture \"%s\"\nOptions: ", arch);
        llvm::TargetRegistry::iterator iter;
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_7 // LLVM 3.7+
        for (iter = llvm::TargetRegistry::targets().begin();
             iter != llvm::TargetRegistry::targets().end(); ++iter)
#else
        for (iter = llvm::TargetRegistry::begin();
             iter != llvm::TargetRegistry::end(); ++iter)
#endif
            fprintf(stderr, "%s ", iter->getName());
        fprintf(stderr, "\n");
        error = true;
    }
    else {
        this->m_arch = arch;
    }

    // Check default LLVM generated targets
    if (!strcasecmp(isa, "sse2") ||
        !strcasecmp(isa, "sse2-i32x4")) {
        this->m_isa = Target::SSE2;
        this->m_nativeVectorWidth = 4;
        this->m_nativeVectorAlignment = 16;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 4;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
        CPUfromISA = CPU_Core2;
    }
    else if (!strcasecmp(isa, "sse2-x2") ||
             !strcasecmp(isa, "sse2-i32x8")) {
        this->m_isa = Target::SSE2;
        this->m_nativeVectorWidth = 4;
        this->m_nativeVectorAlignment = 16;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 8;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
        CPUfromISA = CPU_Core2;
    }
    else if (!strcasecmp(isa, "sse4") ||
             !strcasecmp(isa, "sse4-i32x4")) {
        this->m_isa = Target::SSE4;
        this->m_nativeVectorWidth = 4;
        this->m_nativeVectorAlignment = 16;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 4;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
        CPUfromISA = CPU_Nehalem;
    }
    else if (!strcasecmp(isa, "sse4x2") ||
             !strcasecmp(isa, "sse4-x2") ||
             !strcasecmp(isa, "sse4-i32x8")) {
        this->m_isa = Target::SSE4;
        this->m_nativeVectorWidth = 4;
        this->m_nativeVectorAlignment = 16;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 8;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
        CPUfromISA = CPU_Nehalem;
    }
    else if (!strcasecmp(isa, "sse4-i8x16")) {
        this->m_isa = Target::SSE4;
        this->m_nativeVectorWidth = 16;
        this->m_nativeVectorAlignment = 16;
        this->m_dataTypeWidth = 8;
        this->m_vectorWidth = 16;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 8;
        CPUfromISA = CPU_Nehalem;
    }
    else if (!strcasecmp(isa, "sse4-i16x8")) {
        this->m_isa = Target::SSE4;
        this->m_nativeVectorWidth = 8;
        this->m_nativeVectorAlignment = 16;
        this->m_dataTypeWidth = 16;
        this->m_vectorWidth = 8;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 16;
        CPUfromISA = CPU_Nehalem;
    }
    else if (!strcasecmp(isa, "generic-4") ||
             !strcasecmp(isa, "generic-x4")) {
        this->m_isa = Target::GENERIC;
        this->m_nativeVectorWidth = 4;
        this->m_nativeVectorAlignment = 16;
        this->m_vectorWidth = 4;
        this->m_maskingIsFree = true;
        this->m_maskBitCount = 1;
        this->m_hasHalf = true;
        this->m_hasTranscendentals = true;
        this->m_hasTrigonometry = true;
        this->m_hasGather = this->m_hasScatter = true;
        this->m_hasRsqrtd = this->m_hasRcpd = true;
        CPUfromISA = CPU_Generic;
    }
    else if (!strcasecmp(isa, "generic-8") ||
             !strcasecmp(isa, "generic-x8")) {
        this->m_isa = Target::GENERIC;
        this->m_nativeVectorWidth = 8;
        this->m_nativeVectorAlignment = 32;
        this->m_vectorWidth = 8;
        this->m_maskingIsFree = true;
        this->m_maskBitCount = 1;
        this->m_hasHalf = true;
        this->m_hasTranscendentals = true;
        this->m_hasTrigonometry = true;
        this->m_hasGather = this->m_hasScatter = true;
        this->m_hasRsqrtd = this->m_hasRcpd = true;
        CPUfromISA = CPU_Generic;
    }
    else if (!strcasecmp(isa, "generic-16") ||
             !strcasecmp(isa, "generic-x16") ||
             // We treat *-generic-16 as generic-16, but with special name mangling
             strstr(isa, "-generic-16") || 
             strstr(isa, "-generic-x16")) {
        this->m_isa = Target::GENERIC;
        if (strstr(isa, "-generic-16") ||
            strstr(isa, "-generic-x16")) {
            // It is used for appropriate name mangling and dispatch function during multitarget compilation
            this->m_treatGenericAsSmth = isa;
            // We need to create appropriate name for mangling.
            // Remove "-x16" or "-16" and replace "-" with "_".
            this->m_treatGenericAsSmth = this->m_treatGenericAsSmth.substr(0, this->m_treatGenericAsSmth.find_last_of("-"));
            std::replace(this->m_treatGenericAsSmth.begin(), this->m_treatGenericAsSmth.end(), '-', '_');
        }
        this->m_nativeVectorWidth = 16;
        this->m_nativeVectorAlignment = 64;
        this->m_vectorWidth = 16;
        this->m_maskingIsFree = true;
        this->m_maskBitCount = 1;
        this->m_hasHalf = true;
        this->m_hasTranscendentals = true;
        // It's set to false, because stdlib implementation of math functions
        // is faster on MIC, than "native" implementation provided by the
        // icc compiler.
        this->m_hasTrigonometry = false;
        this->m_hasGather = this->m_hasScatter = true;
        this->m_hasRsqrtd = this->m_hasRcpd = true;
        // It's set to true, because MIC has hardware vector prefetch instruction
        this->m_hasVecPrefetch = true;
        CPUfromISA = CPU_Generic;
    }
    else if (!strcasecmp(isa, "generic-32") ||
             !strcasecmp(isa, "generic-x32")) {
        this->m_isa = Target::GENERIC;
        this->m_nativeVectorWidth = 32;
        this->m_nativeVectorAlignment = 64;
        this->m_vectorWidth = 32;
        this->m_maskingIsFree = true;
        this->m_maskBitCount = 1;
        this->m_hasHalf = true;
        this->m_hasTranscendentals = true;
        this->m_hasTrigonometry = true;
        this->m_hasGather = this->m_hasScatter = true;
        this->m_hasRsqrtd = this->m_hasRcpd = true;
        CPUfromISA = CPU_Generic;
    }
    else if (!strcasecmp(isa, "generic-64") ||
             !strcasecmp(isa, "generic-x64")) {
        this->m_isa = Target::GENERIC;
        this->m_nativeVectorWidth = 64;
        this->m_nativeVectorAlignment = 64;
        this->m_vectorWidth = 64;
        this->m_maskingIsFree = true;
        this->m_maskBitCount = 1;
        this->m_hasHalf = true;
        this->m_hasTranscendentals = true;
        this->m_hasTrigonometry = true;
        this->m_hasGather = this->m_hasScatter = true;
        this->m_hasRsqrtd = this->m_hasRcpd = true;
        CPUfromISA = CPU_Generic;
    }
    else if (!strcasecmp(isa, "generic-1") ||
             !strcasecmp(isa, "generic-x1")) {
        this->m_isa = Target::GENERIC;
        this->m_nativeVectorWidth = 1;
        this->m_nativeVectorAlignment = 16;
        this->m_vectorWidth = 1;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
        CPUfromISA = CPU_Generic;
    }
    else if (!strcasecmp(isa, "avx1-i32x4")) {
        this->m_isa = Target::AVX;
        this->m_nativeVectorWidth = 8;
        this->m_nativeVectorAlignment = 32;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 4;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
        CPUfromISA = CPU_SandyBridge;
    }
    else if (!strcasecmp(isa, "avx") ||
             !strcasecmp(isa, "avx1") ||
             !strcasecmp(isa, "avx1-i32x8")) {
        this->m_isa = Target::AVX;
        this->m_nativeVectorWidth = 8;
        this->m_nativeVectorAlignment = 32;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 8;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
        CPUfromISA = CPU_SandyBridge;
    }
    else if (!strcasecmp(isa, "avx-i64x4") ||
             !strcasecmp(isa, "avx1-i64x4")) {
        this->m_isa = Target::AVX;
        this->m_nativeVectorWidth = 8;  /* native vector width in terms of floats */
        this->m_nativeVectorAlignment = 32;
        this->m_dataTypeWidth = 64;
        this->m_vectorWidth = 4;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 64;
        CPUfromISA = CPU_SandyBridge;
    }
    else if (!strcasecmp(isa, "avx-x2") ||
             !strcasecmp(isa, "avx1-x2") ||
             !strcasecmp(isa, "avx1-i32x16")) {
        this->m_isa = Target::AVX;
        this->m_nativeVectorWidth = 8;
        this->m_nativeVectorAlignment = 32;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 16;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
        CPUfromISA = CPU_SandyBridge;
    }
    else if (!strcasecmp(isa, "avx1.1") ||
             !strcasecmp(isa, "avx1.1-i32x8")) {
        this->m_isa = Target::AVX11;
        this->m_nativeVectorWidth = 8;
        this->m_nativeVectorAlignment = 32;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 8;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
        this->m_hasHalf = true;
        this->m_hasRand = true;
        CPUfromISA = CPU_IvyBridge;
    }
    else if (!strcasecmp(isa, "avx1.1-x2") ||
             !strcasecmp(isa, "avx1.1-i32x16")) {
        this->m_isa = Target::AVX11;
        this->m_nativeVectorWidth = 8;
        this->m_nativeVectorAlignment = 32;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 16;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
        this->m_hasHalf = true;
        this->m_hasRand = true;
        CPUfromISA = CPU_IvyBridge;
    }
    else if (!strcasecmp(isa, "avx1.1-i64x4")) {
        this->m_isa = Target::AVX11;
        this->m_nativeVectorWidth = 8;  /* native vector width in terms of floats */
        this->m_nativeVectorAlignment = 32;
        this->m_dataTypeWidth = 64;
        this->m_vectorWidth = 4;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 64;
        this->m_hasHalf = true;
        this->m_hasRand = true;
        CPUfromISA = CPU_IvyBridge;
    }
    else if (!strcasecmp(isa, "avx2") ||
             !strcasecmp(isa, "avx2-i32x8")) {
        this->m_isa = Target::AVX2;
        this->m_nativeVectorWidth = 8;
        this->m_nativeVectorAlignment = 32;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 8;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
        this->m_hasHalf = true;
        this->m_hasRand = true;
        this->m_hasGather = true;
        CPUfromISA = CPU_Haswell;
    }
    else if (!strcasecmp(isa, "avx2-x2") ||
             !strcasecmp(isa, "avx2-i32x16")) {
        this->m_isa = Target::AVX2;
        this->m_nativeVectorWidth = 16;
        this->m_nativeVectorAlignment = 32;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 16;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
        this->m_hasHalf = true;
        this->m_hasRand = true;
        this->m_hasGather = true;
        CPUfromISA = CPU_Haswell;
    }
    else if (!strcasecmp(isa, "avx2-i64x4")) {
        this->m_isa = Target::AVX2;
        this->m_nativeVectorWidth = 8;  /* native vector width in terms of floats */
        this->m_nativeVectorAlignment = 32;
        this->m_dataTypeWidth = 64;
        this->m_vectorWidth = 4;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 64;
        this->m_hasHalf = true;
        this->m_hasRand = true;
        this->m_hasGather = true;
        CPUfromISA = CPU_Haswell;
    }
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_7 // LLVM 3.7+
    else if (!strcasecmp(isa, "avx512knl-i32x16")) {
        this->m_isa = Target::KNL_AVX512;
        this->m_nativeVectorWidth = 16;
        this->m_nativeVectorAlignment = 64;
        // ?? this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 16;
        this->m_maskingIsFree = true;
        this->m_maskBitCount = 8;
        this->m_hasHalf = true;
        this->m_hasRand = true;
        this->m_hasGather = this->m_hasScatter = true;
        this->m_hasTranscendentals = false;
        // For MIC it is set to true due to performance reasons. The option should be tested.
        this->m_hasTrigonometry = false;
        this->m_hasRsqrtd = this->m_hasRcpd = false;
        this->m_hasVecPrefetch = false;
        CPUfromISA = CPU_KNL;
    }
#endif
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_8 // LLVM 3.8+
    else if (!strcasecmp(isa, "avx512skx-i32x16")) {
        this->m_isa = Target::SKX_AVX512;
        this->m_nativeVectorWidth = 16;
        this->m_nativeVectorAlignment = 64;
        // ?? this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 16;
        this->m_maskingIsFree = true;
        this->m_maskBitCount = 8;
        this->m_hasHalf = true;
        this->m_hasRand = true;
        this->m_hasGather = this->m_hasScatter = true;
        this->m_hasTranscendentals = false;
        // For MIC it is set to true due to performance reasons. The option should be tested.
        this->m_hasTrigonometry = false;
        this->m_hasRsqrtd = this->m_hasRcpd = false;
        this->m_hasVecPrefetch = false;
        CPUfromISA = CPU_SKX;
    }
#endif
#ifdef ISPC_ARM_ENABLED
    else if (!strcasecmp(isa, "neon-i8x16")) {
        this->m_isa = Target::NEON8;
        this->m_nativeVectorWidth = 16;
        this->m_nativeVectorAlignment = 16;
        this->m_dataTypeWidth = 8;
        this->m_vectorWidth = 16;
        this->m_attributes = "+neon,+fp16";
        this->m_hasHalf = true; // ??
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 8;
    }
    else if (!strcasecmp(isa, "neon-i16x8")) {
        this->m_isa = Target::NEON16;
        this->m_nativeVectorWidth = 8;
        this->m_nativeVectorAlignment = 16;
        this->m_dataTypeWidth = 16;
        this->m_vectorWidth = 8;
        this->m_attributes = "+neon,+fp16";
        this->m_hasHalf = true; // ??
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 16;
    }
    else if (!strcasecmp(isa, "neon") ||
             !strcasecmp(isa, "neon-i32x4")) {
        this->m_isa = Target::NEON32;
        this->m_nativeVectorWidth = 4;
        this->m_nativeVectorAlignment = 16;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 4;
        this->m_attributes = "+neon,+fp16";
        this->m_hasHalf = true; // ??
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
    }
#endif
#ifdef ISPC_NVPTX_ENABLED
    else if (!strcasecmp(isa, "nvptx")) {
        this->m_isa = Target::NVPTX;
        this->m_cpu = "sm_35";
        this->m_nativeVectorWidth = 32;
        this->m_nativeVectorAlignment = 32;
        this->m_vectorWidth = 1;
        this->m_hasHalf = true;
        this->m_maskingIsFree = true;
        this->m_maskBitCount = 1;
        this->m_hasTranscendentals = true;
        this->m_hasTrigonometry = true;
        this->m_hasGather = this->m_hasScatter = false;
        CPUfromISA = CPU_SM35;
    }
#endif /* ISPC_NVPTX_ENABLED */
    else {
        Error(SourcePos(), "Target \"%s\" is unknown.  Choices are: %s.",
              isa, SupportedTargets());
        error = true;
    }

#if defined(ISPC_ARM_ENABLED) && !defined(__arm__)
    if ((CPUID == CPU_None) && !strncmp(isa, "neon", 4))
        CPUID = CPU_CortexA9;
#endif

    if (CPUID == CPU_None) {
#ifndef ISPC_ARM_ENABLED
        if (isa == NULL) {
#endif
            std::string hostCPU = llvm::sys::getHostCPUName();
            if (hostCPU.size() > 0)
                cpu = strdup(hostCPU.c_str());
            else {
                Warning(SourcePos(), "Unable to determine host CPU!\n");
                cpu = a.GetDefaultNameFromType(CPU_Generic).c_str();
            }
#ifndef ISPC_ARM_ENABLED
        }
        else {
            cpu = a.GetDefaultNameFromType(CPUfromISA).c_str();
        }
#endif
    }
    else {
        if ((CPUfromISA != CPU_None) &&
            !a.BackwardCompatible(CPUID, CPUfromISA)) {
            Error(SourcePos(), "The requested CPU is incompatible"
                  " with the CPU %s needs: %s vs. %s!\n",
                  isa, cpu, a.GetDefaultNameFromType(CPUfromISA).c_str());
            return;
        }
        cpu = a.GetDefaultNameFromType(CPUID).c_str();
    }
    this->m_cpu = cpu;

    if (!error) {
        // Create TargetMachine
        std::string triple = GetTripleString();

#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_8
        llvm::Reloc::Model relocModel = m_generatePIC ? llvm::Reloc::PIC_ :
            llvm::Reloc::Default;
#else
        llvm::Optional<llvm::Reloc::Model> relocModel;
        if (m_generatePIC) {
          relocModel = llvm::Reloc::PIC_;
        }
#endif
        std::string featuresString = m_attributes;
        llvm::TargetOptions options;
#ifdef ISPC_ARM_ENABLED
        if (m_isa == Target::NEON8 || m_isa == Target::NEON16 ||
            m_isa == Target::NEON32)
            options.FloatABIType = llvm::FloatABI::Hard;
#endif
        if (g->opt.disableFMA == false)
            options.AllowFPOpFusion = llvm::FPOpFusion::Fast;
#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_6
        if (g->NoOmitFramePointer)
            options.NoFramePointerElim = true;
#ifdef ISPC_IS_WINDOWS
        if (strcmp("x86", arch) == 0) {
            // Workaround for issue #503 (LLVM issue 14646).
            // It's Win32 specific.
            options.NoFramePointerElim = true;
        }
#endif
#endif
        m_targetMachine =
            m_target->createTargetMachine(triple, m_cpu, featuresString, options,
                    relocModel);
        Assert(m_targetMachine != NULL);

#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_6
        m_targetMachine->setAsmVerbosityDefault(true);
#else /* LLVM 3.7+ */
        m_targetMachine->Options.MCOptions.AsmVerbose = true;
#endif

#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_5
        // Change default version of generated DWARF.
        if (g->generateDWARFVersion != 0) {
            m_targetMachine->Options.MCOptions.DwarfVersion = g->generateDWARFVersion;
        }
#endif

        // Initialize TargetData/DataLayout in 3 steps.
        // 1. Get default data layout first
        std::string dl_string;
#if ISPC_LLVM_VERSION == ISPC_LLVM_3_6
        dl_string = m_targetMachine->getSubtargetImpl()->getDataLayout()->getStringRepresentation();
#elif ISPC_LLVM_VERSION >= ISPC_LLVM_3_8 // LLVM 3.8+
        dl_string = m_targetMachine->createDataLayout().getStringRepresentation();
#else // LLVM 3.5- or LLVM 3.7
        dl_string = m_targetMachine->getDataLayout()->getStringRepresentation();
#endif
        // 2. Adjust for generic
        if (m_isa == Target::GENERIC) {
            // <16 x i1> vectors only need 16 bit / 2 byte alignment, so add
            // that to the regular datalayout string for IA..
            // For generic-4 target we need to treat <4 x i1> as 128 bit value
            // in terms of required memory storage and alignment, as this is
            // translated to __m128 type.
            dl_string = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                "i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-"
                "f80:128:128-n8:16:32:64-S128-v16:16:16-v32:32:32-v4:128:128";
        }
#ifdef ISPC_NVPTX_ENABLED
        else if (m_isa == Target::NVPTX)
        {
          dl_string = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64";
        }
#endif

        // 3. Finally set member data
        m_dataLayout = new llvm::DataLayout(dl_string);

        // Set is32Bit
        // This indicates if we are compiling for 32 bit platform
        // and can assume 32 bit runtime.
        // FIXME: all generic targets are handled as 64 bit, which is incorrect.

        this->m_is32Bit = (getDataLayout()->getPointerSize() == 4);

#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_3
        // This is LLVM 3.3+ feature.
        // Initialize target-specific "target-feature" attribute.
        if (!m_attributes.empty()) {
            llvm::AttrBuilder attrBuilder;
#ifdef ISPC_NVPTX_ENABLED
            if (m_isa != Target::NVPTX)
#endif
            attrBuilder.addAttribute("target-cpu", this->m_cpu);
            attrBuilder.addAttribute("target-features", this->m_attributes);
#if ISPC_LLVM_VERSION <= ISPC_LLVM_4_0
            this->m_tf_attributes = new llvm::AttributeSet(
                llvm::AttributeSet::get(
                    *g->ctx,
                    llvm::AttributeSet::FunctionIndex,
                    attrBuilder));
#else // LLVM 5.0+
            this->m_tf_attributes = new llvm::AttrBuilder(attrBuilder);
#endif
        }
#endif

        Assert(this->m_vectorWidth <= ISPC_MAX_NVEC);
    }

    m_valid = !error;

    if (printTarget) {
        printf("Target Triple: %s\n", m_targetMachine->getTargetTriple().str().c_str());
        printf("Target CPU: %s\n", m_targetMachine->getTargetCPU().str().c_str());
        printf("Target Feature String: %s\n", m_targetMachine->getTargetFeatureString().str().c_str());
    }

    return;
}


std::string
Target::SupportedCPUs() {
    AllCPUs a;
    return a.HumanReadableListOfNames();
}


const char *
Target::SupportedArchs() {
    return
#ifdef ISPC_ARM_ENABLED
        "arm, "
#endif
        "x86, x86-64";
}


const char *
Target::SupportedTargets() {
    return
        "host, sse2-i32x4, sse2-i32x8, "
        "sse4-i32x4, sse4-i32x8, sse4-i16x8, sse4-i8x16, "
        "avx1-i32x4, "
        "avx1-i32x8, avx1-i32x16, avx1-i64x4, "
        "avx1.1-i32x8, avx1.1-i32x16, avx1.1-i64x4, "
        "avx2-i32x8, avx2-i32x16, avx2-i64x4, "
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_7 // LLVM 3.7+
        "avx512knl-i32x16, "
#endif
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_8 // LLVM 3.8+
        "avx512skx-i32x16, "
#endif
        "generic-x1, generic-x4, generic-x8, generic-x16, "
        "generic-x32, generic-x64, *-generic-x16"
#ifdef ISPC_ARM_ENABLED
        ", neon-i8x16, neon-i16x8, neon-i32x4"
#endif
#ifdef ISPC_NVPTX_ENABLED
        ", nvptx"
#endif
;

}


std::string
Target::GetTripleString() const {
    llvm::Triple triple;
#ifdef ISPC_ARM_ENABLED
    if (m_arch == "arm") {
        triple.setTriple("armv7-eabi");
    }
    else
#endif
    {
        // Start with the host triple as the default
        triple.setTriple(llvm::sys::getDefaultTargetTriple());

        // And override the arch in the host triple based on what the user
        // specified.  Here we need to deal with the fact that LLVM uses one
        // naming convention for targets TargetRegistry, but wants some
        // slightly different ones for the triple.  TODO: is there a way to
        // have it do this remapping, which would presumably be a bit less
        // error prone?
        if (m_arch == "x86")
            triple.setArchName("i386");
        else if (m_arch == "x86-64")
            triple.setArchName("x86_64");
#ifdef ISPC_NVPTX_ENABLED
        else if (m_arch == "nvptx64")
          triple = llvm::Triple("nvptx64", "nvidia", "cuda");
#endif /* ISPC_NVPTX_ENABLED */
        else
            triple.setArchName(m_arch);
    }
    return triple.str();
}

// This function returns string representation of ISA for the purpose of
// mangling. And may return any unique string, preferably short, like
// sse4, avx and etc.
const char *
Target::ISAToString(ISA isa) {
    switch (isa) {
#ifdef ISPC_ARM_ENABLED
    case Target::NEON8:
        return "neon-8";
    case Target::NEON16:
        return "neon-16";
    case Target::NEON32:
        return "neon-32";
#endif
    case Target::SSE2:
        return "sse2";
    case Target::SSE4:
        return "sse4";
    case Target::AVX:
        return "avx";
    case Target::AVX11:
        return "avx11";
    case Target::AVX2:
        return "avx2";
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_7 // LLVM 3.7+
    case Target::KNL_AVX512:
        return "avx512knl";
#endif
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_8 // LLVM 3.8+
    case Target::SKX_AVX512:
        return "avx512skx";
#endif
    case Target::GENERIC:
        return "generic";
#ifdef ISPC_NVPTX_ENABLED
    case Target::NVPTX:
        return "nvptx";
#endif /* ISPC_NVPTX_ENABLED */
    default:
        FATAL("Unhandled target in ISAToString()");
    }
    return "";
}

const char *
Target::GetISAString() const {
    return ISAToString(m_isa);
}


// This function returns string representation of default target corresponding
// to ISA. I.e. for SSE4 it's sse4-i32x4, for AVX11 it's avx1.1-i32x8. This
// string may be used to initialize Target.
const char *
Target::ISAToTargetString(ISA isa) {
    switch (isa) {
#ifdef ISPC_ARM_ENABLED
    case Target::NEON8:
        return "neon-8";
    case Target::NEON16:
        return "neon-16";
    case Target::NEON32:
        return "neon-32";
#endif
    case Target::SSE2:
        return "sse2-i32x4";
    case Target::SSE4:
        return "sse4-i32x4";
    case Target::AVX:
        return "avx1-i32x8";
    case Target::AVX11:
        return "avx1.1-i32x8";
    case Target::AVX2:
        return "avx2-i32x8";
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_7 // LLVM 3.7+
    case Target::KNL_AVX512:
        return "avx512knl-i32x16";
#endif
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_8 // LLVM 3.8+
    case Target::SKX_AVX512:
        return "avx512skx-i32x16";
#endif
    case Target::GENERIC:
        return "generic-4";
#ifdef ISPC_NVPTX_ENABLED
    case Target::NVPTX:
        return "nvptx";
#endif /* ISPC_NVPTX_ENABLED */
    default:
        FATAL("Unhandled target in ISAToTargetString()");
    }
    return "";
}


const char *
Target::GetISATargetString() const {
    return ISAToString(m_isa);
}


static bool
lGenericTypeLayoutIndeterminate(llvm::Type *type) {
    if (type->isFloatingPointTy() || type->isX86_MMXTy() || type->isVoidTy() ||
        type->isIntegerTy() || type->isLabelTy() || type->isMetadataTy())
        return false;

    if (type == LLVMTypes::BoolVectorType ||
        type == LLVMTypes::MaskType ||
        type == LLVMTypes::Int1VectorType)
        return true;

    llvm::ArrayType *at =
        llvm::dyn_cast<llvm::ArrayType>(type);
    if (at != NULL)
        return lGenericTypeLayoutIndeterminate(at->getElementType());

    llvm::PointerType *pt =
        llvm::dyn_cast<llvm::PointerType>(type);
    if (pt != NULL)
        return false;

    llvm::StructType *st =
        llvm::dyn_cast<llvm::StructType>(type);
    if (st != NULL) {
        for (int i = 0; i < (int)st->getNumElements(); ++i)
            if (lGenericTypeLayoutIndeterminate(st->getElementType(i)))
                return true;
        return false;
    }

    Assert(llvm::isa<llvm::VectorType>(type));
    return true;
}


llvm::Value *
Target::SizeOf(llvm::Type *type,
               llvm::BasicBlock *insertAtEnd) {
    if (m_isa == Target::GENERIC &&
        lGenericTypeLayoutIndeterminate(type)) {
        llvm::Value *index[1] = { LLVMInt32(1) };
        llvm::PointerType *ptrType = llvm::PointerType::get(type, 0);
        llvm::Value *voidPtr = llvm::ConstantPointerNull::get(ptrType);
        llvm::ArrayRef<llvm::Value *> arrayRef(&index[0], &index[1]);
#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_6
        llvm::Instruction *gep =
            llvm::GetElementPtrInst::Create(voidPtr, arrayRef, "sizeof_gep",
                                            insertAtEnd);
#else /* LLVM 3.7+ */
        llvm::Instruction *gep =
            llvm::GetElementPtrInst::Create(PTYPE(voidPtr), voidPtr,
                                            arrayRef, "sizeof_gep",
                                            insertAtEnd);
#endif
        if (m_is32Bit || g->opt.force32BitAddressing)
            return new llvm::PtrToIntInst(gep, LLVMTypes::Int32Type,
                                          "sizeof_int", insertAtEnd);
        else
            return new llvm::PtrToIntInst(gep, LLVMTypes::Int64Type,
                                          "sizeof_int", insertAtEnd);
    }

    uint64_t byteSize = getDataLayout()->getTypeStoreSize(type);
    if (m_is32Bit || g->opt.force32BitAddressing)
        return LLVMInt32((int32_t)byteSize);
    else
        return LLVMInt64(byteSize);
}


llvm::Value *
Target::StructOffset(llvm::Type *type, int element,
                     llvm::BasicBlock *insertAtEnd) {
    if (m_isa == Target::GENERIC &&
        lGenericTypeLayoutIndeterminate(type) == true) {
        llvm::Value *indices[2] = { LLVMInt32(0), LLVMInt32(element) };
        llvm::PointerType *ptrType = llvm::PointerType::get(type, 0);
        llvm::Value *voidPtr = llvm::ConstantPointerNull::get(ptrType);
        llvm::ArrayRef<llvm::Value *> arrayRef(&indices[0], &indices[2]);
#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_6
        llvm::Instruction *gep =
            llvm::GetElementPtrInst::Create(voidPtr, arrayRef, "offset_gep",
                                            insertAtEnd);
#else /* LLVM 3.7+ */
        llvm::Instruction *gep =
            llvm::GetElementPtrInst::Create(PTYPE(voidPtr), voidPtr,
                                            arrayRef, "offset_gep",
                                            insertAtEnd);
#endif
        if (m_is32Bit || g->opt.force32BitAddressing)
            return new llvm::PtrToIntInst(gep, LLVMTypes::Int32Type,
                                          "offset_int", insertAtEnd);
        else
            return new llvm::PtrToIntInst(gep, LLVMTypes::Int64Type,
                                          "offset_int", insertAtEnd);
    }

    llvm::StructType *structType =
        llvm::dyn_cast<llvm::StructType>(type);
    if (structType == NULL || structType->isSized() == false) {
        Assert(m->errorCount > 0);
        return NULL;
    }

    const llvm::StructLayout *sl = getDataLayout()->getStructLayout(structType);
    Assert(sl != NULL);

    uint64_t offset = sl->getElementOffset(element);
    if (m_is32Bit || g->opt.force32BitAddressing)
        return LLVMInt32((int32_t)offset);
    else
        return LLVMInt64(offset);
}

void Target::markFuncWithTargetAttr(llvm::Function* func) {
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_3
    if (m_tf_attributes) {
#if ISPC_LLVM_VERSION <= ISPC_LLVM_4_0
        func->addAttributes(llvm::AttributeSet::FunctionIndex, *m_tf_attributes);
#else // LLVM 5.0+
        func->addAttributes(llvm::AttributeList::FunctionIndex, *m_tf_attributes);
#endif
    }
#endif
}


///////////////////////////////////////////////////////////////////////////
// Opt

Opt::Opt() {
    level = 1;
    fastMath = false;
    fastMaskedVload = false;
    force32BitAddressing = true;
    unrollLoops = true;
    disableAsserts = false;
    disableFMA = false;
    forceAlignedMemory = false;
    disableMaskAllOnOptimizations = false;
    disableHandlePseudoMemoryOps = false;
    disableBlendedMaskedStores = false;
    disableCoherentControlFlow = false;
    disableUniformControlFlow = false;
    disableGatherScatterOptimizations = false;
    disableMaskedStoreToStore = false;
    disableGatherScatterFlattening = false;
    disableUniformMemoryOptimizations = false;
    disableCoalescing = false;
}

///////////////////////////////////////////////////////////////////////////
// Globals

Globals::Globals() {
    mathLib = Globals::Math_ISPC;

    includeStdlib = true;
    runCPP = true;
    debugPrint = false;
    printTarget = false;
    NoOmitFramePointer = false;
    debugIR = -1;
    disableWarnings = false;
    warningsAsErrors = false;
    quiet = false;
    forceColoredOutput = false;
    disableLineWrap = false;
    emitPerfWarnings = true;
    emitInstrumentation = false;
    generateDebuggingSymbols = false;
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_5
    generateDWARFVersion = 3;
#endif
    enableFuzzTest = false;
    fuzzTestSeed = -1;
    mangleFunctionsWithTarget = false;

    ctx = new llvm::LLVMContext;

#ifdef ISPC_IS_WINDOWS
    _getcwd(currentDirectory, sizeof(currentDirectory));
#else
    if (getcwd(currentDirectory, sizeof(currentDirectory)) == NULL)
        FATAL("Current directory path too long!");
#endif
    forceAlignment = -1;
    dllExport = false;
}

///////////////////////////////////////////////////////////////////////////
// SourcePos

SourcePos::SourcePos(const char *n, int fl, int fc, int ll, int lc) {
    name = n;
    if (name == NULL) {
        if (m != NULL)
            name = m->module->getModuleIdentifier().c_str();
        else
            name = "(unknown)";
    }
    first_line = fl;
    first_column = fc;
    last_line = ll != 0 ? ll : fl;
    last_column = lc != 0 ? lc : fc;
}


#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_6
llvm::DIFile
#else /* LLVM 3.7+ */
llvm::DIFile*
//llvm::MDFile*
#endif
SourcePos::GetDIFile() const {
    std::string directory, filename;
    GetDirectoryAndFileName(g->currentDirectory, name, &directory, &filename);
#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_6
    llvm::DIFile ret = m->diBuilder->createFile(filename, directory);
    Assert(ret.Verify());
#else /* LLVM 3.7+ */
    llvm::DIFile *ret = m->diBuilder->createFile(filename, directory);
#endif
    return ret;
}


void
SourcePos::Print() const {
    printf(" @ [%s:%d.%d - %d.%d] ", name, first_line, first_column,
           last_line, last_column);
}


bool
SourcePos::operator==(const SourcePos &p2) const {
    return (!strcmp(name, p2.name) &&
            first_line == p2.first_line &&
            first_column == p2.first_column &&
            last_line == p2.last_line &&
            last_column == p2.last_column);
}


SourcePos
Union(const SourcePos &p1, const SourcePos &p2) {
    if (strcmp(p1.name, p2.name) != 0)
        return p1;

    SourcePos ret;
    ret.name = p1.name;
    ret.first_line = std::min(p1.first_line, p2.first_line);
    ret.first_column = std::min(p1.first_column, p2.first_column);
    ret.last_line = std::max(p1.last_line, p2.last_line);
    ret.last_column = std::max(p1.last_column, p2.last_column);
    return ret;
}
