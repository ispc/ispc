/*
  Copyright (c) 2010-2019, Intel Corporation
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
#include "llvmutil.h"
#include "module.h"
#include "util.h"
#include <sstream>
#include <stdarg.h> /* va_list, va_start, va_arg, va_end */
#include <stdio.h>
#ifdef ISPC_HOST_IS_WINDOWS
#include <direct.h>
#include <windows.h>
#define strcasecmp stricmp
#include <intrin.h>
#else
#include <sys/types.h>
#include <unistd.h>
#endif
#include <llvm/CodeGen/TargetLowering.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

#include <llvm/BinaryFormat/Dwarf.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>

Globals *g;
Module *m;

///////////////////////////////////////////////////////////////////////////
// Target

#if !defined(ISPC_HOST_IS_WINDOWS) && !defined(__arm__) && !defined(__aarch64__)
// __cpuid() and __cpuidex() are defined on Windows in <intrin.h> for x86/x64.
// On *nix they need to be defined manually through inline assembler.
static void __cpuid(int info[4], int infoType) {
    __asm__ __volatile__("cpuid" : "=a"(info[0]), "=b"(info[1]), "=c"(info[2]), "=d"(info[3]) : "0"(infoType));
}

/* Save %ebx in case it's the PIC register */
static void __cpuidex(int info[4], int level, int count) {
    __asm__ __volatile__("xchg{l}\t{%%}ebx, %1\n\t"
                         "cpuid\n\t"
                         "xchg{l}\t{%%}ebx, %1\n\t"
                         : "=a"(info[0]), "=r"(info[1]), "=c"(info[2]), "=d"(info[3])
                         : "0"(level), "2"(count));
}
#endif // !ISPC_HOST_IS_WINDOWS && !__ARM__ && !__AARCH64__

#if !defined(__arm__) && !defined(__aarch64__)
static bool __os_has_avx_support() {
#if defined(ISPC_HOST_IS_WINDOWS)
    // Check if the OS will save the YMM registers
    unsigned long long xcrFeatureMask = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
    return (xcrFeatureMask & 6) == 6;
#else  // !defined(ISPC_HOST_IS_WINDOWS)
    // Check xgetbv; this uses a .byte sequence instead of the instruction
    // directly because older assemblers do not include support for xgetbv and
    // there is no easy way to conditionally compile based on the assembler used.
    int rEAX, rEDX;
    __asm__ __volatile__(".byte 0x0f, 0x01, 0xd0" : "=a"(rEAX), "=d"(rEDX) : "c"(0));
    return (rEAX & 6) == 6;
#endif // !defined(ISPC_HOST_IS_WINDOWS)
}

static bool __os_has_avx512_support() {
#if defined(ISPC_HOST_IS_WINDOWS)
    // Check if the OS saves the XMM, YMM and ZMM registers, i.e. it supports AVX2 and AVX512.
    // See section 2.1 of software.intel.com/sites/default/files/managed/0d/53/319433-022.pdf
    unsigned long long xcrFeatureMask = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
    return (xcrFeatureMask & 0xE6) == 0xE6;
#else  // !defined(ISPC_HOST_IS_WINDOWS)
    // Check xgetbv; this uses a .byte sequence instead of the instruction
    // directly because older assemblers do not include support for xgetbv and
    // there is no easy way to conditionally compile based on the assembler used.
    int rEAX, rEDX;
    __asm__ __volatile__(".byte 0x0f, 0x01, 0xd0" : "=a"(rEAX), "=d"(rEDX) : "c"(0));
    return (rEAX & 0xE6) == 0xE6;
#endif // !defined(ISPC_HOST_IS_WINDOWS)
}
#endif // !__arm__ && !__aarch64__

static ISPCTarget lGetSystemISA() {
#if defined(__arm__) || defined(__aarch64__)
    return ISPCTarget::neon_i32x4;
#else
    int info[4];
    __cpuid(info, 1);

    int info2[4];
    // Call cpuid with eax=7, ecx=0
    __cpuidex(info2, 7, 0);

    if ((info[2] & (1 << 27)) != 0 &&  // OSXSAVE
        (info2[1] & (1 << 5)) != 0 &&  // AVX2
        (info2[1] & (1 << 16)) != 0 && // AVX512 F
        __os_has_avx512_support()) {
        // We need to verify that AVX2 is also available,
        // as well as AVX512, because our targets are supposed
        // to use both.

        if ((info2[1] & (1 << 17)) != 0 && // AVX512 DQ
            (info2[1] & (1 << 28)) != 0 && // AVX512 CDI
            (info2[1] & (1 << 30)) != 0 && // AVX512 BW
            (info2[1] & (1 << 31)) != 0) { // AVX512 VL
            return ISPCTarget::avx512skx_i32x16;
        } else if ((info2[1] & (1 << 26)) != 0 && // AVX512 PF
                   (info2[1] & (1 << 27)) != 0 && // AVX512 ER
                   (info2[1] & (1 << 28)) != 0) { // AVX512 CDI
            return ISPCTarget::avx512knl_i32x16;
        }
        // If it's unknown AVX512 target, fall through and use AVX2
        // or whatever is available in the machine.
    }

    if ((info[2] & (1 << 27)) != 0 &&                           // OSXSAVE
        (info[2] & (1 << 28)) != 0 && __os_has_avx_support()) { // AVX
        // AVX1 for sure....
        // Ivy Bridge?
        if ((info[2] & (1 << 29)) != 0 && // F16C
            (info[2] & (1 << 30)) != 0 && // RDRAND
            (info2[1] & (1 << 5)) != 0) { // AVX2.
            return ISPCTarget::avx2_i32x8;
        }
        // Regular AVX
        return ISPCTarget::avx1_i32x8;
    } else if ((info[2] & (1 << 19)) != 0)
        return ISPCTarget::sse4_i32x4;
    else if ((info[3] & (1 << 26)) != 0)
        return ISPCTarget::sse2_i32x4;
    else {
        Error(SourcePos(), "Unable to detect supported SSE/AVX ISA.  Exiting.");
        exit(1);
    }
#endif
}

static const bool lIsTargetValidforArch(ISPCTarget target, Arch arch) {
    bool ret = true;
    // If target name starts with sse or avx, has to be x86 or x86-64.
    if (ISPCTargetIsX86(target)) {
        if (arch != Arch::x86_64 && arch != Arch::x86)
            ret = false;
    } else if (target == ISPCTarget::neon_i8x16 || target == ISPCTarget::neon_i16x8) {
        if (arch != Arch::arm)
            ret = false;
    } else if (target == ISPCTarget::neon_i32x4 || target == ISPCTarget::neon_i32x8) {
        if (arch != Arch::arm && arch != Arch::aarch64)
            ret = false;
    }

    return ret;
}

typedef enum {
    // Special value, indicates that no CPU is present.
    CPU_None = 0,

    // 'Generic' CPU without any hardware SIMD capabilities.
    CPU_Generic = 1,

    // A generic 64-bit specific x86 processor model which tries to be good
    // for modern chips without enabling instruction set encodings past the
    // basic SSE2 and 64-bit ones
    CPU_x86_64,

    // Early Atom CPU. Supports SSSE3.
    CPU_Bonnell,

    // Generic Core2-like. Supports SSSE3. Isn`t quite compatible with Bonnell,
    // but for ISPC the difference is negligible; ISPC doesn`t make use of it.
    CPU_Core2,

    // Core2 Solo/Duo/Quad/Extreme. Supports SSE 4.1 (but not 4.2).
    CPU_Penryn,

    // Late Core2-like. Supports SSE 4.2 + POPCNT/LZCNT.
    CPU_Nehalem,

    // CPU in PS4/Xbox One.
    CPU_PS4,

    // Sandy Bridge. Supports AVX 1.
    CPU_SandyBridge,

    // Ivy Bridge. Supports AVX 1 + RDRAND.
    CPU_IvyBridge,

    // Haswell. Supports AVX 2.
    CPU_Haswell,

    // Broadwell. Supports AVX 2 + ADX/RDSEED/SMAP.
    CPU_Broadwell,

    // Knights Landing - Xeon Phi.
    // Supports AVX-512F: All the key AVX-512 features: masking, broadcast... ;
    //          AVX-512CDI: Conflict Detection;
    //          AVX-512ERI & PRI: 28-bit precision RCP, RSQRT and EXP transcendentals,
    //                            new prefetch instructions.
    CPU_KNL,
    // Skylake Xeon.
    // Supports AVX-512F: All the key AVX-512 features: masking, broadcast... ;
    //          AVX-512CDI: Conflict Detection;
    //          AVX-512VL: Vector Length Orthogonality;
    //          AVX-512DQ: New HPC ISA (vs AVX512F);
    //          AVX-512BW: Byte and Word Support.
    CPU_SKX,

#if ISPC_LLVM_VERSION >= ISPC_LLVM_8_0
    // Icelake client
    CPU_ICL,
#endif

    // Late Atom-like design. Supports SSE 4.2 + POPCNT/LZCNT.
    CPU_Silvermont,

// FIXME: LLVM supports a ton of different ARM CPU variants--not just
// cortex-a9 and a15.  We should be able to handle any of them that also
// have NEON support.
#ifdef ISPC_ARM_ENABLED
    // ARM Cortex A15. Supports NEON VFPv4.
    CPU_CortexA15,

    // ARM Cortex A9. Supports NEON VFPv3.
    CPU_CortexA9,

    // ARM Cortex A35, A53, A57.
    CPU_CortexA35,
    CPU_CortexA53,
    CPU_CortexA57,
#endif

    sizeofCPUtype
} CPUtype;

class AllCPUs {
  private:
    std::vector<std::vector<std::string>> names;
    std::vector<std::set<CPUtype>> compat;

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
        names = std::vector<std::vector<std::string>>(sizeofCPUtype);
        compat = std::vector<std::set<CPUtype>>(sizeofCPUtype);

        names[CPU_None].push_back("");

        names[CPU_Generic].push_back("generic");

        names[CPU_x86_64].push_back("x86-64");

        names[CPU_Bonnell].push_back("atom");
        names[CPU_Bonnell].push_back("bonnell");

        names[CPU_Core2].push_back("core2");

        names[CPU_Penryn].push_back("penryn");

        names[CPU_Silvermont].push_back("slm");
        names[CPU_Silvermont].push_back("silvermont");

        names[CPU_Nehalem].push_back("corei7");
        names[CPU_Nehalem].push_back("nehalem");

        names[CPU_PS4].push_back("btver2");
        names[CPU_PS4].push_back("ps4");

        names[CPU_SandyBridge].push_back("corei7-avx");
        names[CPU_SandyBridge].push_back("sandybridge");

        names[CPU_IvyBridge].push_back("core-avx-i");
        names[CPU_IvyBridge].push_back("ivybridge");

        names[CPU_Haswell].push_back("core-avx2");
        names[CPU_Haswell].push_back("haswell");

        names[CPU_Broadwell].push_back("broadwell");

        names[CPU_KNL].push_back("knl");

        names[CPU_SKX].push_back("skx");

#if ISPC_LLVM_VERSION >= ISPC_LLVM_8_0 // LLVM 8.0+
        names[CPU_ICL].push_back("icl");
        names[CPU_ICL].push_back("icelake-client");
#endif

#ifdef ISPC_ARM_ENABLED
        names[CPU_CortexA15].push_back("cortex-a15");

        names[CPU_CortexA9].push_back("cortex-a9");

        names[CPU_CortexA35].push_back("cortex-a35");

        names[CPU_CortexA53].push_back("cortex-a53");

        names[CPU_CortexA57].push_back("cortex-a57");
#endif

        Assert(names.size() == sizeofCPUtype);

        compat[CPU_Silvermont] =
            Set(CPU_Generic, CPU_x86_64, CPU_Bonnell, CPU_Penryn, CPU_Core2, CPU_Nehalem, CPU_Silvermont, CPU_None);

        compat[CPU_KNL] = Set(CPU_KNL, CPU_Generic, CPU_x86_64, CPU_Bonnell, CPU_Penryn, CPU_Core2, CPU_Nehalem,
                              CPU_Silvermont, CPU_SandyBridge, CPU_IvyBridge, CPU_Haswell, CPU_Broadwell, CPU_None);

        compat[CPU_SKX] = Set(CPU_SKX, CPU_x86_64, CPU_Bonnell, CPU_Penryn, CPU_Core2, CPU_Nehalem, CPU_Silvermont,
                              CPU_SandyBridge, CPU_IvyBridge, CPU_Haswell, CPU_Broadwell, CPU_None);

#if ISPC_LLVM_VERSION >= ISPC_LLVM_8_0 // LLVM 8.0+
        compat[CPU_ICL] = Set(CPU_ICL, CPU_SKX, CPU_x86_64, CPU_Bonnell, CPU_Penryn, CPU_Core2, CPU_Nehalem,
                              CPU_Silvermont, CPU_SandyBridge, CPU_IvyBridge, CPU_Haswell, CPU_Broadwell, CPU_None);
        ;
#endif

        compat[CPU_Broadwell] =
            Set(CPU_Generic, CPU_x86_64, CPU_Bonnell, CPU_Penryn, CPU_Core2, CPU_Nehalem, CPU_Silvermont,
                CPU_SandyBridge, CPU_IvyBridge, CPU_Haswell, CPU_Broadwell, CPU_None);
        compat[CPU_Haswell] = Set(CPU_Generic, CPU_x86_64, CPU_Bonnell, CPU_Penryn, CPU_Core2, CPU_Nehalem,
                                  CPU_Silvermont, CPU_SandyBridge, CPU_IvyBridge, CPU_Haswell, CPU_Broadwell, CPU_None);
        compat[CPU_IvyBridge] = Set(CPU_Generic, CPU_x86_64, CPU_Bonnell, CPU_Penryn, CPU_Core2, CPU_Nehalem,
                                    CPU_Silvermont, CPU_SandyBridge, CPU_IvyBridge, CPU_None);
        compat[CPU_SandyBridge] = Set(CPU_Generic, CPU_x86_64, CPU_Bonnell, CPU_Penryn, CPU_Core2, CPU_Nehalem,
                                      CPU_Silvermont, CPU_SandyBridge, CPU_None);
        compat[CPU_PS4] = Set(CPU_Generic, CPU_x86_64, CPU_Bonnell, CPU_Penryn, CPU_Core2, CPU_Nehalem, CPU_Silvermont,
                              CPU_SandyBridge, CPU_PS4, CPU_None);
        compat[CPU_Nehalem] =
            Set(CPU_Generic, CPU_x86_64, CPU_Bonnell, CPU_Penryn, CPU_Core2, CPU_Nehalem, CPU_Silvermont, CPU_None);
        compat[CPU_Penryn] =
            Set(CPU_Generic, CPU_x86_64, CPU_Bonnell, CPU_Penryn, CPU_Core2, CPU_Nehalem, CPU_Silvermont, CPU_None);
        compat[CPU_Core2] = Set(CPU_Generic, CPU_x86_64, CPU_Bonnell, CPU_Core2, CPU_None);
        compat[CPU_Bonnell] = Set(CPU_Generic, CPU_x86_64, CPU_Bonnell, CPU_Core2, CPU_None);
        compat[CPU_Generic] = Set(CPU_Generic, CPU_None);

        compat[CPU_x86_64] = Set(CPU_Generic, CPU_x86_64, CPU_None);

#ifdef ISPC_ARM_ENABLED
        compat[CPU_CortexA15] = Set(CPU_Generic, CPU_CortexA9, CPU_CortexA15, CPU_None);
        compat[CPU_CortexA9] = Set(CPU_Generic, CPU_CortexA9, CPU_None);
        compat[CPU_CortexA35] = Set(CPU_Generic, CPU_CortexA35, CPU_None);
        compat[CPU_CortexA53] = Set(CPU_Generic, CPU_CortexA53, CPU_None);
        compat[CPU_CortexA57] = Set(CPU_Generic, CPU_CortexA57, CPU_None);
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
            for (int j = 0, je = names[i].size(); (retn == CPU_None) && (j < je); j++)
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

Target::Target(Arch arch, const char *cpu, ISPCTarget target, bool pic, bool printTarget)
    : m_target(NULL), m_targetMachine(NULL), m_dataLayout(NULL), m_valid(false), m_isa(SSE2), m_arch(Arch::none),
      m_is32Bit(true), m_cpu(""), m_attributes(""), m_tf_attributes(NULL), m_nativeVectorWidth(-1),
      m_nativeVectorAlignment(-1), m_dataTypeWidth(-1), m_vectorWidth(-1), m_generatePIC(pic), m_maskingIsFree(false),
      m_maskBitCount(-1), m_hasHalf(false), m_hasRand(false), m_hasGather(false), m_hasScatter(false),
      m_hasTranscendentals(false), m_hasTrigonometry(false), m_hasRsqrtd(false), m_hasRcpd(false),
      m_hasVecPrefetch(false) {
    CPUtype CPUID = CPU_None, CPUfromISA = CPU_None;
    AllCPUs a;
    std::string featuresString;

    if (cpu) {
        CPUID = a.GetTypeFromName(cpu);
        if (CPUID == CPU_None) {
            Error(SourcePos(),
                  "Error: CPU type \"%s\" unknown. Supported"
                  " CPUs: %s.",
                  cpu, a.HumanReadableListOfNames().c_str());
            return;
        }
    }

    if (target == ISPCTarget::none) {
        // If a CPU was specified explicitly, try to pick the best
        // possible ISA based on that.
        switch (CPUID) {
        case CPU_None: {
            // No CPU and no ISA, so use system info to figure out
            // what this CPU supports.
            target = lGetSystemISA();
            std::string target_string = ISPCTargetToString(target);
            Warning(SourcePos(),
                    "No --target specified on command-line."
                    " Using default system target \"%s\".",
                    target_string.c_str());
            break;
        }

        case CPU_Generic:
            target = ISPCTarget::generic_1;
            break;

#ifdef ISPC_ARM_ENABLED
        case CPU_CortexA9:
        case CPU_CortexA15:
        case CPU_CortexA35:
        case CPU_CortexA53:
        case CPU_CortexA57:
            target = ISPCTarget::neon_i32x4;
            break;
#endif

        case CPU_KNL:
            target = ISPCTarget::avx512knl_i32x16;
            break;

#if ISPC_LLVM_VERSION >= ISPC_LLVM_8_0 // LLVM 8.0
        case CPU_ICL:
#endif
        case CPU_SKX:
            target = ISPCTarget::avx512skx_i32x16;
            break;

        case CPU_Broadwell:
        case CPU_Haswell:
            target = ISPCTarget::avx2_i32x8;
            break;

        case CPU_IvyBridge:
            // No specific target for IvyBridge anymore.
            target = ISPCTarget::avx1_i32x8;
            break;

        case CPU_SandyBridge:
            target = ISPCTarget::avx1_i32x8;
            break;

        // Penryn is here because ISPC does not use SSE 4.2
        case CPU_Penryn:
        case CPU_Nehalem:
        case CPU_Silvermont:
            target = ISPCTarget::sse4_i32x4;
            break;

        case CPU_PS4:
            target = ISPCTarget::avx1_i32x4;
            break;

        default:
            target = ISPCTarget::sse2_i32x4;
            break;
        }
        if (CPUID != CPU_None) {
            std::string target_string = ISPCTargetToString(target);
            Warning(SourcePos(),
                    "No --target specified on command-line."
                    " Using ISA \"%s\" based on specified CPU \"%s\".",
                    target_string.c_str(), cpu);
        }
    }

    if (target == ISPCTarget::host) {
        target = lGetSystemISA();
    }

    if (arch == Arch::none) {
#ifdef ISPC_ARM_ENABLED
        if (ISPCTargetIsNeon(target)) {
#if defined(__arm__)
            arch = Arch::arm;
#else
            arch = Arch::aarch64;
#endif
        } else
#endif
            arch = Arch::x86_64;
    }

    bool error = false;

    // Make sure the target architecture is a known one; print an error
    // with the valid ones otherwise.
    for (llvm::TargetRegistry::iterator iter = llvm::TargetRegistry::targets().begin();
         iter != llvm::TargetRegistry::targets().end(); ++iter) {
        if (ArchToString(arch) == iter->getName()) {
            this->m_target = &*iter;
            break;
        }
    }
    if (this->m_target == NULL) {
        std::string error_message;
        error_message = "Invalid architecture \"";
        error_message += ArchToString(arch);
        error_message += "\"\nOptions: ";
        llvm::TargetRegistry::iterator iter;
        const char *separator = "";
        for (iter = llvm::TargetRegistry::targets().begin(); iter != llvm::TargetRegistry::targets().end(); ++iter) {
            error_message += separator;
            error_message += iter->getName();
            separator = ", ";
        }
        error_message += ".";
        Error(SourcePos(), "%s", error_message.c_str());
        error = true;
    } else {
        this->m_arch = arch;
    }

    // Ensure that we have a valid target/arch combination.
    if (!lIsTargetValidforArch(target, arch)) {
        std::string str_arch = ArchToString(arch);
        std::string target_string = ISPCTargetToString(target);
        Error(SourcePos(), "arch = %s and target = %s is not a valid combination.", str_arch.c_str(),
              target_string.c_str());
        return;
    }

    // Check default LLVM generated targets
    bool unsupported_target = false;
    switch (target) {
    case ISPCTarget::sse2_i32x4:
        this->m_isa = Target::SSE2;
        this->m_nativeVectorWidth = 4;
        this->m_nativeVectorAlignment = 16;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 4;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
        CPUfromISA = CPU_x86_64;
        break;
    case ISPCTarget::sse2_i32x8:
        this->m_isa = Target::SSE2;
        this->m_nativeVectorWidth = 4;
        this->m_nativeVectorAlignment = 16;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 8;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
        CPUfromISA = CPU_Core2;
        break;
    case ISPCTarget::sse4_i8x16:
        this->m_isa = Target::SSE4;
        this->m_nativeVectorWidth = 16;
        this->m_nativeVectorAlignment = 16;
        this->m_dataTypeWidth = 8;
        this->m_vectorWidth = 16;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 8;
        CPUfromISA = CPU_Nehalem;
        break;
    case ISPCTarget::sse4_i16x8:
        this->m_isa = Target::SSE4;
        this->m_nativeVectorWidth = 8;
        this->m_nativeVectorAlignment = 16;
        this->m_dataTypeWidth = 16;
        this->m_vectorWidth = 8;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 16;
        CPUfromISA = CPU_Nehalem;
        break;
    case ISPCTarget::sse4_i32x4:
        this->m_isa = Target::SSE4;
        this->m_nativeVectorWidth = 4;
        this->m_nativeVectorAlignment = 16;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 4;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
        CPUfromISA = CPU_Nehalem;
        break;
    case ISPCTarget::sse4_i32x8:
        this->m_isa = Target::SSE4;
        this->m_nativeVectorWidth = 4;
        this->m_nativeVectorAlignment = 16;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 8;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
        CPUfromISA = CPU_Nehalem;
        break;
    case ISPCTarget::avx1_i32x4:
        this->m_isa = Target::AVX;
        this->m_nativeVectorWidth = 8;
        this->m_nativeVectorAlignment = 32;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 4;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
        CPUfromISA = CPU_SandyBridge;
        break;
    case ISPCTarget::avx1_i32x8:
        this->m_isa = Target::AVX;
        this->m_nativeVectorWidth = 8;
        this->m_nativeVectorAlignment = 32;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 8;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
        CPUfromISA = CPU_SandyBridge;
        break;
    case ISPCTarget::avx1_i32x16:
        this->m_isa = Target::AVX;
        this->m_nativeVectorWidth = 8;
        this->m_nativeVectorAlignment = 32;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 16;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
        CPUfromISA = CPU_SandyBridge;
        break;
    case ISPCTarget::avx1_i64x4:
        this->m_isa = Target::AVX;
        this->m_nativeVectorWidth = 8; /* native vector width in terms of floats */
        this->m_nativeVectorAlignment = 32;
        this->m_dataTypeWidth = 64;
        this->m_vectorWidth = 4;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 64;
        CPUfromISA = CPU_SandyBridge;
        break;
    case ISPCTarget::avx2_i32x4:
        this->m_isa = Target::AVX2;
        this->m_nativeVectorWidth = 8;
        this->m_nativeVectorAlignment = 32;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 4;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
        this->m_hasHalf = true;
        this->m_hasRand = true;
        this->m_hasGather = true;
        CPUfromISA = CPU_Haswell;
        break;
    case ISPCTarget::avx2_i32x8:
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
        break;
    case ISPCTarget::avx2_i32x16:
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
        break;
    case ISPCTarget::avx2_i64x4:
        this->m_isa = Target::AVX2;
        this->m_nativeVectorWidth = 8; /* native vector width in terms of floats */
        this->m_nativeVectorAlignment = 32;
        this->m_dataTypeWidth = 64;
        this->m_vectorWidth = 4;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 64;
        this->m_hasHalf = true;
        this->m_hasRand = true;
        this->m_hasGather = true;
        CPUfromISA = CPU_Haswell;
        break;
    case ISPCTarget::avx512knl_i32x16:
        this->m_isa = Target::KNL_AVX512;
        this->m_nativeVectorWidth = 16;
        this->m_nativeVectorAlignment = 64;
        this->m_dataTypeWidth = 32;
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
        break;
    case ISPCTarget::avx512skx_i32x8:
#if ISPC_LLVM_VERSION >= ISPC_LLVM_8_0 // LLVM 8.0+
        this->m_isa = Target::SKX_AVX512;
        this->m_nativeVectorWidth = 16;
        this->m_nativeVectorAlignment = 64;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 8;
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
        this->m_funcAttributes.push_back(std::make_pair("prefer-vector-width", "256"));
        this->m_funcAttributes.push_back(std::make_pair("min-legal-vector-width", "256"));
        break;
#else
        unsupported_target = true;
        break;
#endif
    case ISPCTarget::avx512skx_i32x16:
        this->m_isa = Target::SKX_AVX512;
        this->m_nativeVectorWidth = 16;
        this->m_nativeVectorAlignment = 64;
        this->m_dataTypeWidth = 32;
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
        break;
    case ISPCTarget::generic_1:
        this->m_isa = Target::GENERIC;
        this->m_nativeVectorWidth = 1;
        this->m_nativeVectorAlignment = 16;
        this->m_vectorWidth = 1;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
        CPUfromISA = CPU_Generic;
        break;
    case ISPCTarget::generic_4:
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
        break;
    case ISPCTarget::generic_8:
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
        break;
    case ISPCTarget::generic_16:
        this->m_isa = Target::GENERIC;
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
        break;
    case ISPCTarget::generic_32:
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
        break;
    case ISPCTarget::generic_64:
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
        break;
#ifdef ISPC_ARM_ENABLED
    case ISPCTarget::neon_i8x16:
        this->m_isa = Target::NEON;
        this->m_nativeVectorWidth = 16;
        this->m_nativeVectorAlignment = 16;
        this->m_dataTypeWidth = 8;
        this->m_vectorWidth = 16;
        this->m_hasHalf = true; // ??
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 8;
        break;
    case ISPCTarget::neon_i16x8:
        this->m_isa = Target::NEON;
        this->m_nativeVectorWidth = 8;
        this->m_nativeVectorAlignment = 16;
        this->m_dataTypeWidth = 16;
        this->m_vectorWidth = 8;
        this->m_hasHalf = true; // ??
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 16;
        break;
    case ISPCTarget::neon_i32x4:
        this->m_isa = Target::NEON;
        this->m_nativeVectorWidth = 4;
        this->m_nativeVectorAlignment = 16;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 4;
        this->m_hasHalf = true; // ??
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
        break;
    case ISPCTarget::neon_i32x8:
        this->m_isa = Target::NEON;
        this->m_nativeVectorWidth = 4;
        this->m_nativeVectorAlignment = 16;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 8;
        this->m_hasHalf = true; // ??
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
        break;
#else
    case ISPCTarget::neon_i8x16:
    case ISPCTarget::neon_i16x8:
    case ISPCTarget::neon_i32x4:
    case ISPCTarget::neon_i32x8:
        unsupported_target = true;
        break;
#endif
    case ISPCTarget::none:
    case ISPCTarget::host:
    case ISPCTarget::error:
        unsupported_target = true;
        break;
    }

    if (unsupported_target) {
        // Hitting one of unsupported targets is internal error.
        // Proper reporting about incorrect targets is done during options parsing.
        std::string target_string = "Problem with target (" + ISPCTargetToString(target) + ")";
        FATAL(target_string.c_str());
    }

#if defined(ISPC_ARM_ENABLED) && !defined(__arm__)
    if ((CPUID == CPU_None) && ISPCTargetIsNeon(target) && arch == Arch::arm)
        CPUID = CPU_CortexA9;
#endif
#if defined(ISPC_ARM_ENABLED) && !defined(__aarch64__)
    if ((CPUID == CPU_None) && ISPCTargetIsNeon(target) && arch == Arch::aarch64)
        CPUID = CPU_CortexA35;
#endif
    if (CPUID == CPU_None) {
        cpu = a.GetDefaultNameFromType(CPUfromISA).c_str();
    } else {
        if ((CPUfromISA != CPU_None) && !a.BackwardCompatible(CPUID, CPUfromISA)) {
            std::string target_string = ISPCTargetToString(target);
            Error(SourcePos(),
                  "The requested CPU (%s) is incompatible"
                  " with the CPU required for %s target (%s)",
                  cpu, target_string.c_str(), a.GetDefaultNameFromType(CPUfromISA).c_str());
            return;
        }
        cpu = a.GetDefaultNameFromType(CPUID).c_str();
    }
    this->m_cpu = cpu;

    if (!error) {
        // Create TargetMachine
        std::string triple = GetTripleString();

        llvm::Optional<llvm::Reloc::Model> relocModel;
        if (m_generatePIC) {
            relocModel = llvm::Reloc::PIC_;
        }
        llvm::TargetOptions options;
#ifdef ISPC_ARM_ENABLED
        if (m_isa == Target::NEON)
            options.FloatABIType = llvm::FloatABI::Hard;
        if (arch == Arch::arm) {
            this->m_funcAttributes.push_back(std::make_pair("target-features", "+neon,+fp16"));
            featuresString = "+neon,+fp16";
        } else if (arch == Arch::aarch64) {
            this->m_funcAttributes.push_back(std::make_pair("target-features", "+neon"));
            featuresString = "+neon";
        }
#endif
        if (g->opt.disableFMA == false)
            options.AllowFPOpFusion = llvm::FPOpFusion::Fast;

        m_targetMachine = m_target->createTargetMachine(triple, m_cpu, featuresString, options, relocModel);
        Assert(m_targetMachine != NULL);

        // Set Optimization level for llvm codegen based on Optimization level
        // requested by user via ISPC Optimization Flag. Mapping is :
        // ISPC O0 -> Codegen O0
        // ISPC O1,O2,O3,default -> Codegen O3
        llvm::CodeGenOpt::Level cOptLevel = llvm::CodeGenOpt::Level::Aggressive;
        switch (g->codegenOptLevel) {
        case Globals::CodegenOptLevel::None:
            cOptLevel = llvm::CodeGenOpt::Level::None;
            break;

        case Globals::CodegenOptLevel::Aggressive:
            cOptLevel = llvm::CodeGenOpt::Level::Aggressive;
            break;
        }
        m_targetMachine->setOptLevel(cOptLevel);

        m_targetMachine->Options.MCOptions.AsmVerbose = true;

        // Change default version of generated DWARF.
        if (g->generateDWARFVersion != 0) {
            m_targetMachine->Options.MCOptions.DwarfVersion = g->generateDWARFVersion;
        }

        // Initialize TargetData/DataLayout in 3 steps.
        // 1. Get default data layout first
        std::string dl_string;
        dl_string = m_targetMachine->createDataLayout().getStringRepresentation();
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

        // 3. Finally set member data
        m_dataLayout = new llvm::DataLayout(dl_string);

        // Set is32Bit
        // This indicates if we are compiling for 32 bit platform
        // and can assume 32 bit runtime.
        // FIXME: all generic targets are handled as 64 bit, which is incorrect.

        this->m_is32Bit = (getDataLayout()->getPointerSize() == 4);

        // TO-DO : Revisit addition of "target-features" and "target-cpu" for ARM support.
        llvm::AttrBuilder fattrBuilder;
#ifdef ISPC_ARM_ENABLED
        if (m_isa == Target::NEON)
            fattrBuilder.addAttribute("target-cpu", this->m_cpu);
#endif
        for (auto const &f_attr : m_funcAttributes)
            fattrBuilder.addAttribute(f_attr.first, f_attr.second);
        this->m_tf_attributes = new llvm::AttrBuilder(fattrBuilder);

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

std::string Target::SupportedCPUs() {
    AllCPUs a;
    return a.HumanReadableListOfNames();
}

const char *Target::SupportedArchs() {
    return
#ifdef ISPC_ARM_ENABLED
        "arm, aarch64, "
#endif
        "x86, x86-64";
}

const char *Target::SupportedTargets() {
    return "host, sse2-i32x4, sse2-i32x8, "
           "sse4-i32x4, sse4-i32x8, sse4-i16x8, sse4-i8x16, "
           "avx1-i32x4, "
           "avx1-i32x8, avx1-i32x16, avx1-i64x4, "
           "avx2-i32x4, avx2-i32x8, avx2-i32x16, avx2-i64x4, "
           "avx512knl-i32x16, "
           "avx512skx-i32x16, "
#if ISPC_LLVM_VERSION >= ISPC_LLVM_8_0 // LLVM 8.0+
           "avx512skx-i32x8, "
#endif
           "generic-x1, generic-x4, generic-x8, generic-x16, "
           "generic-x32, generic-x64"
#ifdef ISPC_ARM_ENABLED
           ", neon-i8x16, neon-i16x8, neon-i32x4, neon-i32x8"
#endif
        ;
}

std::string Target::SupportedOSes() {
    std::vector<std::string> supported;
#if defined(ISPC_HOST_IS_WINDOWS)
#if !defined(ISPC_WINDOWS_TARGET_OFF)
    supported.push_back("windows");
#endif
#if !defined(ISPC_PS4_TARGET_OFF)
    supported.push_back("ps4");
#endif
#elif defined(ISPC_HOST_IS_APPLE)
#if !defined(ISPC_IOS_TARGET_OFF)
    supported.push_back("ios");
#endif
#endif
#if !defined(ISPC_LINUX_TARGET_OFF)
    supported.push_back("linux");
#endif
#if !defined(ISPC_MACOS_TARGET_OFF)
    supported.push_back("macos");
#endif
#if !defined(ISPC_ANDROID_TARGET_OFF)
    supported.push_back("android");
#endif
    std::string result;
    for (int i = 0; i < supported.size(); i++) {
        result += supported[i];
        if (i < supported.size() - 1) {
            result += ", ";
        }
    }
    return result;
}

std::string Target::GetTripleString() const {
    llvm::Triple triple;
    switch (g->target_os) {
    case TargetOS::windows:
        if (m_arch == Arch::x86) {
            triple.setArchName("i386");
        } else if (m_arch == Arch::x86_64) {
            triple.setArchName("x86_64");
        } else if (m_arch == Arch::arm) {
            Error(SourcePos(), "Arm is not supported on Windows.");
            exit(1);
        } else if (m_arch == Arch::aarch64) {
            Error(SourcePos(), "Aarch64 is not supported on Windows.");
            exit(1);
        } else {
            Error(SourcePos(), "Unknown arch.");
            exit(1);
        }
        //"x86_64-pc-windows-msvc"
        triple.setVendor(llvm::Triple::VendorType::PC);
        triple.setOS(llvm::Triple::OSType::Win32);
        triple.setEnvironment(llvm::Triple::EnvironmentType::MSVC);
        break;
    case TargetOS::linux:
        if (m_arch == Arch::x86) {
            triple.setArchName("i386");
        } else if (m_arch == Arch::x86_64) {
            triple.setArchName("x86_64");
        } else if (m_arch == Arch::arm) {
            triple.setArchName("armv7");
        } else if (m_arch == Arch::aarch64) {
            triple.setArchName("aarch64");
        } else {
            Error(SourcePos(), "Unknown arch.");
            exit(1);
        }
        triple.setVendor(llvm::Triple::VendorType::UnknownVendor);
        triple.setOS(llvm::Triple::OSType::Linux);
        triple.setEnvironment(llvm::Triple::EnvironmentType::GNU);
        break;
    case TargetOS::macos:
        // asserts
        if (m_arch != Arch::x86_64) {
            Error(SourcePos(), "macOS target supports only x86_64.");
            exit(1);
        }
        triple.setArch(llvm::Triple::ArchType::x86_64);
        triple.setVendor(llvm::Triple::VendorType::Apple);
        triple.setOS(llvm::Triple::OSType::MacOSX);
        break;
    case TargetOS::android:
        if (m_arch == Arch::x86) {
            triple.setArchName("i386");
        } else if (m_arch == Arch::x86_64) {
            triple.setArchName("x86_64");
        } else if (m_arch == Arch::arm) {
            triple.setArchName("armv7");
        } else if (m_arch == Arch::aarch64) {
            triple.setArchName("aarch64");
        } else {
            Error(SourcePos(), "Unknown arch.");
            exit(1);
        }
        triple.setVendor(llvm::Triple::VendorType::UnknownVendor);
        triple.setOS(llvm::Triple::OSType::Linux);
        triple.setEnvironment(llvm::Triple::EnvironmentType::Android);
        break;
    case TargetOS::ios:
        if (m_arch != Arch::aarch64) {
            Error(SourcePos(), "iOS target supports only aarch64.");
            exit(1);
        }
        // Note, for iOS arch need to be set to "arm64", instead of "aarch64".
        // Internet say this is for historical reasons.
        // "arm64-apple-ios"
        triple.setArchName("arm64");
        triple.setVendor(llvm::Triple::VendorType::Apple);
        triple.setOS(llvm::Triple::OSType::IOS);
        break;
    case TargetOS::ps4:
        if (m_arch != Arch::x86_64) {
            Error(SourcePos(), "PS4 target supports only x86_64.");
            exit(1);
        }
        // "x86_64-scei-ps4"
        triple.setArch(llvm::Triple::ArchType::x86_64);
        triple.setVendor(llvm::Triple::VendorType::SCEI);
        triple.setOS(llvm::Triple::OSType::PS4);
        break;
    default:
        Error(SourcePos(), "Invalid target OS.");
        exit(1);
    }

    return triple.str();
}

// This function returns string representation of ISA for the purpose of
// mangling. And may return any unique string, preferably short, like
// sse4, avx and etc.
const char *Target::ISAToString(ISA isa) {
    switch (isa) {
#ifdef ISPC_ARM_ENABLED
    case Target::NEON:
        return "neon";
#endif
    case Target::SSE2:
        return "sse2";
    case Target::SSE4:
        return "sse4";
    case Target::AVX:
        return "avx";
    case Target::AVX2:
        return "avx2";
    case Target::KNL_AVX512:
        return "avx512knl";
    case Target::SKX_AVX512:
        return "avx512skx";
    case Target::GENERIC:
        return "generic";
    default:
        FATAL("Unhandled target in ISAToString()");
    }
    return "";
}

const char *Target::GetISAString() const { return ISAToString(m_isa); }

// This function returns string representation of default target corresponding
// to ISA. I.e. for SSE4 it's sse4-i32x4, for AVX2 it's avx2-i32x8. This
// string may be used to initialize Target.
const char *Target::ISAToTargetString(ISA isa) {
    switch (isa) {
#ifdef ISPC_ARM_ENABLED
    case Target::NEON:
        return "neon-i32x4";
#endif
    case Target::SSE2:
        return "sse2-i32x4";
    case Target::SSE4:
        return "sse4-i32x4";
    case Target::AVX:
        return "avx1-i32x8";
    case Target::AVX2:
        return "avx2-i32x8";
    case Target::KNL_AVX512:
        return "avx512knl-i32x16";
    case Target::SKX_AVX512:
        return "avx512skx-i32x16";
    case Target::GENERIC:
        return "generic-4";
    default:
        FATAL("Unhandled target in ISAToTargetString()");
    }
    return "";
}

const char *Target::GetISATargetString() const { return ISAToString(m_isa); }

static bool lGenericTypeLayoutIndeterminate(llvm::Type *type) {
    if (type->isFloatingPointTy() || type->isX86_MMXTy() || type->isVoidTy() || type->isIntegerTy() ||
        type->isLabelTy() || type->isMetadataTy())
        return false;

    if (type == LLVMTypes::BoolVectorType || type == LLVMTypes::MaskType || type == LLVMTypes::Int1VectorType)
        return true;

    llvm::ArrayType *at = llvm::dyn_cast<llvm::ArrayType>(type);
    if (at != NULL)
        return lGenericTypeLayoutIndeterminate(at->getElementType());

    llvm::PointerType *pt = llvm::dyn_cast<llvm::PointerType>(type);
    if (pt != NULL)
        return false;

    llvm::StructType *st = llvm::dyn_cast<llvm::StructType>(type);
    if (st != NULL) {
        for (int i = 0; i < (int)st->getNumElements(); ++i)
            if (lGenericTypeLayoutIndeterminate(st->getElementType(i)))
                return true;
        return false;
    }

    Assert(llvm::isa<llvm::VectorType>(type));
    return true;
}

llvm::Value *Target::SizeOf(llvm::Type *type, llvm::BasicBlock *insertAtEnd) {
    if (m_isa == Target::GENERIC && lGenericTypeLayoutIndeterminate(type)) {
        llvm::Value *index[1] = {LLVMInt32(1)};
        llvm::PointerType *ptrType = llvm::PointerType::get(type, 0);
        llvm::Value *voidPtr = llvm::ConstantPointerNull::get(ptrType);
        llvm::ArrayRef<llvm::Value *> arrayRef(&index[0], &index[1]);
        llvm::Instruction *gep =
            llvm::GetElementPtrInst::Create(PTYPE(voidPtr), voidPtr, arrayRef, "sizeof_gep", insertAtEnd);
        if (m_is32Bit || g->opt.force32BitAddressing)
            return new llvm::PtrToIntInst(gep, LLVMTypes::Int32Type, "sizeof_int", insertAtEnd);
        else
            return new llvm::PtrToIntInst(gep, LLVMTypes::Int64Type, "sizeof_int", insertAtEnd);
    }

    uint64_t byteSize = getDataLayout()->getTypeStoreSize(type);
    if (m_is32Bit || g->opt.force32BitAddressing)
        return LLVMInt32((int32_t)byteSize);
    else
        return LLVMInt64(byteSize);
}

llvm::Value *Target::StructOffset(llvm::Type *type, int element, llvm::BasicBlock *insertAtEnd) {
    if (m_isa == Target::GENERIC && lGenericTypeLayoutIndeterminate(type) == true) {
        llvm::Value *indices[2] = {LLVMInt32(0), LLVMInt32(element)};
        llvm::PointerType *ptrType = llvm::PointerType::get(type, 0);
        llvm::Value *voidPtr = llvm::ConstantPointerNull::get(ptrType);
        llvm::ArrayRef<llvm::Value *> arrayRef(&indices[0], &indices[2]);
        llvm::Instruction *gep =
            llvm::GetElementPtrInst::Create(PTYPE(voidPtr), voidPtr, arrayRef, "offset_gep", insertAtEnd);
        if (m_is32Bit || g->opt.force32BitAddressing)
            return new llvm::PtrToIntInst(gep, LLVMTypes::Int32Type, "offset_int", insertAtEnd);
        else
            return new llvm::PtrToIntInst(gep, LLVMTypes::Int64Type, "offset_int", insertAtEnd);
    }

    llvm::StructType *structType = llvm::dyn_cast<llvm::StructType>(type);
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

void Target::markFuncWithTargetAttr(llvm::Function *func) {
    if (m_tf_attributes) {
        func->addAttributes(llvm::AttributeList::FunctionIndex, *m_tf_attributes);
    }
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
    codegenOptLevel = Globals::Aggressive;

    includeStdlib = true;
    runCPP = true;
    debugPrint = false;
    dumpFile = false;
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
    noPragmaOnce = false;
    generateDebuggingSymbols = false;
    generateDWARFVersion = 3;
    enableFuzzTest = false;
    fuzzTestSeed = -1;
    mangleFunctionsWithTarget = false;

    ctx = new llvm::LLVMContext;

#ifdef ISPC_HOST_IS_WINDOWS
    _getcwd(currentDirectory, sizeof(currentDirectory));
#else
    if (getcwd(currentDirectory, sizeof(currentDirectory)) == NULL)
        FATAL("Current directory path is too long!");
#endif
    forceAlignment = -1;
    dllExport = false;

    // Target OS defaults to host OS.
    target_os = GetHostOS();
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

llvm::DIFile *
// llvm::MDFile*
SourcePos::GetDIFile() const {
    std::string directory, filename;
    GetDirectoryAndFileName(g->currentDirectory, name, &directory, &filename);
    llvm::DIFile *ret = m->diBuilder->createFile(filename, directory);
    return ret;
}

void SourcePos::Print() const {
    printf(" @ [%s:%d.%d - %d.%d] ", name, first_line, first_column, last_line, last_column);
}

bool SourcePos::operator==(const SourcePos &p2) const {
    return (!strcmp(name, p2.name) && first_line == p2.first_line && first_column == p2.first_column &&
            last_line == p2.last_line && last_column == p2.last_column);
}

SourcePos Union(const SourcePos &p1, const SourcePos &p2) {
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

TargetOS StringToOS(std::string os) {
    std::string supportedOses = Target::SupportedOSes();
    if (supportedOses.find(os) == std::string::npos) {
        return TargetOS::error;
    }
    if (os == "windows") {
        return TargetOS::windows;
    } else if (os == "linux") {
        return TargetOS::linux;
    } else if (os == "macos") {
        return TargetOS::macos;
    } else if (os == "android") {
        return TargetOS::android;
    } else if (os == "ios") {
        return TargetOS::ios;
    } else if (os == "ps4") {
        return TargetOS::ps4;
    }
    return TargetOS::error;
}

constexpr TargetOS GetHostOS() {
#if defined(ISPC_HOST_IS_WINDOWS) && !defined(ISPC_WINDOWS_TARGET_OFF)
    return TargetOS::windows;
#elif defined(ISPC_HOST_IS_LINUX) && !defined(ISPC_LINUX_TARGET_OFF)
    return TargetOS::linux;
#elif defined(ISPC_HOST_IS_APPLE) && !defined(ISPC_MACOS_TARGET_OFF)
    return TargetOS::macos;
#else
    return TargetOS::error;
#endif
}

Arch ParseArch(std::string arch) {
    if (arch == "x86") {
        return Arch::x86;
    } else if (arch == "x86_64" || arch == "x86-64") {
        return Arch::x86_64;
    } else if (arch == "arm") {
        return Arch::arm;
    } else if (arch == "aarch64") {
        return Arch::aarch64;
    }
    return Arch::error;
}

std::string ArchToString(Arch arch) {
    switch (arch) {
    case Arch::x86:
        return "x86";
    case Arch::x86_64:
        return "x86-64";
    case Arch::arm:
        return "arm";
    case Arch::aarch64:
        return "aarch64";
    default:
        // none and error are not supposed to be printed.
        Error(SourcePos(), "Invalid arch is processed");
        exit(1);
    }
    return "error";
}

ISPCTarget ParseISPCTarget(std::string target) {
    // TODO: ensure skx-i32x8 is not enabled and linked for earli LLVM version.

    // The first matching string for each target is the canonical way to name the target,
    // all other strings are aliases.
    if (target == "host") {
        return ISPCTarget::host;
    } else if (target == "sse2-i32x4" || target == "sse2") {
        return ISPCTarget::sse2_i32x4;
    } else if (target == "sse2-i32x8" || target == "sse2-x2") {
        return ISPCTarget::sse2_i32x8;
    } else if (target == "sse4-i8x16") {
        return ISPCTarget::sse4_i8x16;
    } else if (target == "sse4-i16x8") {
        return ISPCTarget::sse4_i16x8;
    } else if (target == "sse4-i32x4" || target == "sse4") {
        return ISPCTarget::sse4_i32x4;
    } else if (target == "sse4-i32x8" || target == "sse4-x2" || target == "sse4x2") {
        return ISPCTarget::sse4_i32x8;
    } else if (target == "avx1-i32x4") {
        return ISPCTarget::avx1_i32x4;
    } else if (target == "avx1-i32x8" || target == "avx" || target == "avx1") {
        return ISPCTarget::avx1_i32x8;
    } else if (target == "avx1-i64x4" || target == "avx-i64x4") {
        return ISPCTarget::avx1_i64x4;
    } else if (target == "avx1-i32x16" || target == "avx-x2" || target == "avx1-x2") {
        return ISPCTarget::avx1_i32x16;
    } else if (target == "avx2-i32x4") {
        return ISPCTarget::avx2_i32x4;
    } else if (target == "avx2-i32x8" || target == "avx2") {
        return ISPCTarget::avx2_i32x8;
    } else if (target == "avx2-i64x4") {
        return ISPCTarget::avx2_i64x4;
    } else if (target == "avx2-i32x16" || target == "avx2-x2") {
        return ISPCTarget::avx2_i32x16;
    } else if (target == "avx512knl-i32x16") {
        return ISPCTarget::avx512knl_i32x16;
    } else if (target == "avx512skx-i32x16") {
        return ISPCTarget::avx512skx_i32x16;
    } else if (target == "avx512skx-i32x8") {
        return ISPCTarget::avx512skx_i32x8;
    } else if (target == "generic-1" || target == "generic-x1") {
        return ISPCTarget::generic_1;
    } else if (target == "generic-4" || target == "generic-x4") {
        return ISPCTarget::generic_4;
    } else if (target == "generic-8" || target == "generic-x8") {
        return ISPCTarget::generic_8;
    } else if (target == "generic-16" || target == "generic-x16") {
        return ISPCTarget::generic_16;
    } else if (target == "generic-32" || target == "generic-x32") {
        return ISPCTarget::generic_32;
    } else if (target == "generic-64" || target == "generic-x64") {
        return ISPCTarget::generic_64;
    } else if (target == "neon-i8x16") {
        return ISPCTarget::neon_i8x16;
    } else if (target == "neon-i16x8") {
        return ISPCTarget::neon_i8x16;
    } else if (target == "neon-i32x4" || target == "neon") {
        return ISPCTarget::neon_i32x4;
    } else if (target == "neon-i32x8") {
        return ISPCTarget::neon_i32x8;
    }

    return ISPCTarget::error;
}

// Given a comma-delimited string with one or more compilation targets of
// the form "sse4-i32x4,avx2-i32x8", return a pair. First element of the pair is a vector
// of correctly parsed targets, second element of the pair is a strings with targets, which
// were not recognized.
std::pair<std::vector<ISPCTarget>, std::string> ParseISPCTargets(const char *target) {
    std::vector<ISPCTarget> targets;
    std::string error_target;
    const char *tstart = target;
    bool done = false;
    while (!done) {
        const char *tend = strchr(tstart, ',');
        if (tend == NULL) {
            done = true;
            tend = strchr(tstart, '\0');
        }
        std::string target_string = std::string(tstart, tend);
        ISPCTarget target_parsed = ParseISPCTarget(target_string);
        if (target_parsed == ISPCTarget::error) {
            if (!error_target.empty()) {
                error_target += ",";
            }
            error_target += target_string;
        } else {
            targets.push_back(target_parsed);
        }
        tstart = tend + 1;
    }
    return std::make_pair(targets, error_target);
}

std::string ISPCTargetToString(ISPCTarget target) {
    switch (target) {
    case ISPCTarget::host:
        return "host";
    case ISPCTarget::sse2_i32x4:
        return "sse2-i32x4";
    case ISPCTarget::sse2_i32x8:
        return "sse2-i32x8";
    case ISPCTarget::sse4_i8x16:
        return "sse4-i8x16";
    case ISPCTarget::sse4_i16x8:
        return "sse4-i16x8";
    case ISPCTarget::sse4_i32x4:
        return "sse4-i32x4";
    case ISPCTarget::sse4_i32x8:
        return "sse4-i32x8";
    case ISPCTarget::avx1_i32x4:
        return "avx1-i32x4";
    case ISPCTarget::avx1_i32x8:
        return "avx1-i32x8";
    case ISPCTarget::avx1_i32x16:
        return "avx1-i32x16";
    case ISPCTarget::avx1_i64x4:
        return "avx1-i64x4";
    case ISPCTarget::avx2_i32x4:
        return "avx2-i32x4";
    case ISPCTarget::avx2_i32x8:
        return "avx2-i32x8";
    case ISPCTarget::avx2_i32x16:
        return "avx2-i32x16";
    case ISPCTarget::avx2_i64x4:
        return "avx2-i64x4";
    case ISPCTarget::avx512knl_i32x16:
        return "avx512knl-i32x16";
    case ISPCTarget::avx512skx_i32x8:
        return "avx512skx-i32x8";
    case ISPCTarget::avx512skx_i32x16:
        return "avx512skx-i32x16";
    case ISPCTarget::generic_1:
        return "generic-1";
    case ISPCTarget::generic_4:
        return "generic-4";
    case ISPCTarget::generic_8:
        return "generic-8";
    case ISPCTarget::generic_16:
        return "generic-16";
    case ISPCTarget::generic_32:
        return "generic-32";
    case ISPCTarget::generic_64:
        return "generic-64";
    case ISPCTarget::neon_i8x16:
        return "neon-i8x16";
    case ISPCTarget::neon_i16x8:
        return "neon-i16x8";
    case ISPCTarget::neon_i32x4:
        return "neon-i32x4";
    case ISPCTarget::neon_i32x8:
        return "neon-i32x8";
    case ISPCTarget::none:
    case ISPCTarget::error:
        // Fall through
        ;
    }
    Error(SourcePos(), "Invalid ISPCTarget is processed");
    exit(1);
}

bool ISPCTargetIsX86(ISPCTarget target) {
    switch (target) {
    case ISPCTarget::sse2_i32x4:
    case ISPCTarget::sse2_i32x8:
    case ISPCTarget::sse4_i8x16:
    case ISPCTarget::sse4_i16x8:
    case ISPCTarget::sse4_i32x4:
    case ISPCTarget::sse4_i32x8:
    case ISPCTarget::avx1_i32x4:
    case ISPCTarget::avx1_i32x8:
    case ISPCTarget::avx1_i32x16:
    case ISPCTarget::avx1_i64x4:
    case ISPCTarget::avx2_i32x4:
    case ISPCTarget::avx2_i32x8:
    case ISPCTarget::avx2_i32x16:
    case ISPCTarget::avx2_i64x4:
    case ISPCTarget::avx512knl_i32x16:
    case ISPCTarget::avx512skx_i32x8:
    case ISPCTarget::avx512skx_i32x16:
        return true;
    default:
        return false;
    }
}

bool ISPCTargetIsGeneric(ISPCTarget target) {
    switch (target) {
    case ISPCTarget::generic_1:
    case ISPCTarget::generic_4:
    case ISPCTarget::generic_8:
    case ISPCTarget::generic_16:
    case ISPCTarget::generic_32:
    case ISPCTarget::generic_64:
        return true;
    default:
        return false;
    }
}

bool ISPCTargetIsNeon(ISPCTarget target) {
    switch (target) {
    case ISPCTarget::neon_i8x16:
    case ISPCTarget::neon_i16x8:
    case ISPCTarget::neon_i32x4:
    case ISPCTarget::neon_i32x8:
        return true;
    default:
        return false;
    }
}
