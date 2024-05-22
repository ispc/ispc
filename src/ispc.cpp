/*
  Copyright (c) 2010-2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
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
#else // !ISPC_HOST_IS_WINDOWS
#include <sys/types.h>
#include <unistd.h>
#endif // ISPC_HOST_IS_WINDOWS

#include <llvm/BinaryFormat/Dwarf.h>
#include <llvm/CodeGen/TargetLowering.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/CodeGen.h>
#if ISPC_LLVM_VERSION >= ISPC_LLVM_17_0
#include <llvm/TargetParser/Host.h>
#else
#include <llvm/Support/Host.h>
#endif
#if ISPC_LLVM_VERSION >= ISPC_LLVM_14_0
#include <llvm/MC/TargetRegistry.h>
#else
#include <llvm/Support/TargetRegistry.h>
#endif
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>

#if ISPC_LLVM_VERSION > ISPC_LLVM_17_0
using CodegenOptLevel = llvm::CodeGenOptLevel;
#else
using CodegenOptLevel = llvm::CodeGenOpt::Level;
#endif

using namespace ispc;

Globals *ispc::g;
Module *ispc::m;

///////////////////////////////////////////////////////////////////////////
// Target

#if defined(__arm__) || defined(__aarch64__) || defined(_M_ARM64)
#define ISPC_HOST_IS_ARM
#elif defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
#define ISPC_HOST_IS_X86
#endif

#if !defined(ISPC_HOST_IS_WINDOWS) && defined(ISPC_HOST_IS_X86)
// __cpuid() and __cpuidex() are defined on Windows in <intrin.h> for x86/x64.
// On *nix they need to be defined manually through inline assembler.
static void __cpuid(int info[4], int infoType) {
    __asm__ __volatile__("cpuid" : "=a"(info[0]), "=b"(info[1]), "=c"(info[2]), "=d"(info[3]) : "0"(infoType));
}

static void __cpuidex(int info[4], int level, int count) {
    __asm__ __volatile__("cpuid" : "=a"(info[0]), "=b"(info[1]), "=c"(info[2]), "=d"(info[3]) : "0"(level), "2"(count));
}
#endif // !ISPC_HOST_IS_WINDOWS && __x86_64__

#ifdef ISPC_HOST_IS_X86
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
#elif defined(ISPC_HOST_IS_APPLE)
    // macOS has different way of dealing with AVX512 than Windows and Linux:
    // - by default AVX512 is off in the newly created thread, which means CPUID flags will
    //   indicate AVX512 availability, but OS support check (XCR0) will not succeed.
    // - AVX512 can be enabled either by calling thread_set_state() or by executing any
    //   AVX512 instruction, which would cause #UD exception handled by the OS.
    // The purpose of this check is to identify if AVX512 is potentially available, so we
    // need to bypass OS check and look at CPUID flags only.
    // See ispc issue #1854 for more details.
    return true;
#else  // !defined(ISPC_HOST_IS_WINDOWS)
    // Check xgetbv; this uses a .byte sequence instead of the instruction
    // directly because older assemblers do not include support for xgetbv and
    // there is no easy way to conditionally compile based on the assembler used.
    int rEAX, rEDX;
    __asm__ __volatile__(".byte 0x0f, 0x01, 0xd0" : "=a"(rEAX), "=d"(rEDX) : "c"(0));
    return (rEAX & 0xE6) == 0xE6;
#endif // !defined(ISPC_HOST_IS_WINDOWS)
}
#endif // ISPC_HOST_IS_X86

static ISPCTarget lGetSystemISA() {
#if defined(ISPC_HOST_IS_ARM)
    return ISPCTarget::neon_i32x4;
#elif defined(ISPC_HOST_IS_X86)
    int info[4];
    __cpuid(info, 1);

    int info2[4];
    // Call cpuid with eax=7, ecx=0
    __cpuidex(info2, 7, 0);

    int info3[4] = {0, 0, 0, 0};
    int max_subleaf = info2[0];
    // Call cpuid with eax=7, ecx=1
    if (max_subleaf >= 1)
        __cpuidex(info3, 7, 1);

    // clang-format off
    bool sse2 =                (info[3] & (1 << 26))  != 0;
    bool sse41 =               (info[2] & (1 << 19))  != 0;
    bool sse42 =               (info[2] & (1 << 20))  != 0;
    bool avx_f16c =            (info[2] & (1 << 29))  != 0;
    bool avx_rdrand =          (info[2] & (1 << 30))  != 0;
    bool osxsave =             (info[2] & (1 << 27))  != 0;
    bool avx =                 (info[2] & (1 << 28))  != 0;
    bool avx2 =                (info2[1] & (1 << 5))  != 0;
    bool avx_vnni =            (info3[0] & (1 << 4))  != 0;

    bool avx512_f =            (info2[1] & (1 << 16)) != 0;
    // clang-format on

    if (osxsave && avx2 && avx512_f && __os_has_avx512_support()) {
        // We need to verify that AVX2 is also available,
        // as well as AVX512, because our targets are supposed
        // to use both.

        // clang-format off
        bool avx512_dq =           (info2[1] & (1 << 17)) != 0;
        bool avx512_pf =           (info2[1] & (1 << 26)) != 0;
        bool avx512_er =           (info2[1] & (1 << 27)) != 0;
        bool avx512_cd =           (info2[1] & (1 << 28)) != 0;
        bool avx512_bw =           (info2[1] & (1 << 30)) != 0;
        bool avx512_vl =           (info2[1] & (1 << 31)) != 0;
        bool avx512_vbmi2 =        (info2[2] & (1 << 6))  != 0;
        bool avx512_gfni =         (info2[2] & (1 << 8))  != 0;
        bool avx512_vaes =         (info2[2] & (1 << 9))  != 0;
        bool avx512_vpclmulqdq =   (info2[2] & (1 << 10)) != 0;
        bool avx512_vnni =         (info2[2] & (1 << 11)) != 0;
        bool avx512_bitalg =       (info2[2] & (1 << 12)) != 0;
        bool avx512_vpopcntdq =    (info2[2] & (1 << 14)) != 0;
        bool avx512_bf16 =         (info3[0] & (1 << 5))  != 0;
        bool avx512_vp2intersect = (info2[3] & (1 << 8))  != 0;
        bool avx512_amx_bf16 =     (info2[3] & (1 << 22)) != 0;
        bool avx512_amx_tile =     (info2[3] & (1 << 24)) != 0;
        bool avx512_amx_int8 =     (info2[3] & (1 << 25)) != 0;
        bool avx512_fp16 =         (info2[3] & (1 << 23)) != 0;
        // clang-format on

        // Knights Landing:          KNL = F + PF + ER + CD
        // Skylake server:           SKX = F + DQ + CD + BW + VL
        // Cascade Lake server:      CLX = SKX + VNNI
        // Cooper Lake server:       CPX = CLX + BF16
        // Ice Lake client & server: ICL = CLX + VBMI2 + GFNI + VAES + VPCLMULQDQ + BITALG + VPOPCNTDQ
        // Tiger Lake:               TGL = ICL + VP2INTERSECT
        // Sapphire Rapids:          SPR = ICL + BF16 + AMX_BF16 + AMX_TILE + AMX_INT8 + AVX_VNNI + FP16
        bool knl = avx512_pf && avx512_er && avx512_cd;
        bool skx = avx512_dq && avx512_cd && avx512_bw && avx512_vl;
        bool clx = skx && avx512_vnni;
        [[maybe_unused]] bool cpx = clx && avx512_bf16;
        bool icl =
            clx && avx512_vbmi2 && avx512_gfni && avx512_vaes && avx512_vpclmulqdq && avx512_bitalg && avx512_vpopcntdq;
        [[maybe_unused]] bool tgl = icl && avx512_vp2intersect;
        bool spr =
            icl && avx512_bf16 && avx512_amx_bf16 && avx512_amx_tile && avx512_amx_int8 && avx_vnni && avx512_fp16;
        if (spr) {
            // We don't care if AMX is enabled or not here, as AMX support is not implemented yet.
            return ISPCTarget::avx512spr_x16;
        } else if (icl) {
            return ISPCTarget::avx512icl_x16;
        } else if (skx) {
            return ISPCTarget::avx512skx_x16;
        } else if (knl) {
            return ISPCTarget::avx512knl_x16;
        }
        // If it's unknown AVX512 target, fall through and use AVX2
        // or whatever is available in the machine.
    }

    if (osxsave && avx && __os_has_avx_support()) {
        if (avx_vnni) {
            return ISPCTarget::avx2vnni_i32x8;
        }
        // AVX1 for sure....
        // Ivy Bridge?
        if (avx_f16c && avx_rdrand && avx2) {
            return ISPCTarget::avx2_i32x8;
        }
        // Regular AVX
        return ISPCTarget::avx1_i32x8;
    } else if (sse42) {
        return ISPCTarget::sse4_i32x4;
    } else if (sse41) {
        return ISPCTarget::sse41_i32x4;
    } else if (sse2) {
        return ISPCTarget::sse2_i32x4;
    } else {
        Error(SourcePos(), "Unable to detect supported SSE/AVX ISA.  Exiting.");
        exit(1);
    }
#else
#error "Unsupported host CPU architecture."
#endif
}

static bool lIsTargetValidforArch(ISPCTarget target, Arch arch) {
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
    } else if (ISPCTargetIsGen(target)) {
        if (arch != Arch::xe64)
            ret = false;
    }

    return ret;
}
typedef enum {
    // Special value, indicates that no CPU is present.
    CPU_None = 0,

    // A generic 64-bit specific x86 processor model which tries to be good
    // for modern chips without enabling instruction set encodings past the
    // basic SSE2 and 64-bit ones
    CPU_x86_64 = 1,

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

    // Skylake. AVX2.
    CPU_Skylake,

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

    // Icelake client
    CPU_ICL,

    // Late Atom-like design. Supports SSE 4.2 + POPCNT/LZCNT.
    CPU_Silvermont,

    CPU_ICX,
    CPU_TGL,
    CPU_ADL,
#if ISPC_LLVM_VERSION >= ISPC_LLVM_16_0
    CPU_MTL,
#endif
#if ISPC_LLVM_VERSION >= ISPC_LLVM_14_0
    CPU_SPR,
#endif

    // Zen 1-2-3
    CPU_ZNVER1,
    CPU_ZNVER2,
    CPU_ZNVER3,

// FIXME: LLVM supports a ton of different ARM CPU variants--not just
// cortex-a9 and a15.  We should be able to handle any of them that also
// have NEON support.
#ifdef ISPC_ARM_ENABLED
    // ARM Cortex A9. Supports NEON VFPv3.
    CPU_CortexA9,

    // ARM Cortex A15. Supports NEON VFPv4.
    CPU_CortexA15,

    // ARM Cortex A35, A53, A57.
    CPU_CortexA35,
    CPU_CortexA53,
    CPU_CortexA57,

    // Apple CPUs.
    CPU_AppleA7,
    CPU_AppleA10,
    CPU_AppleA11,
    CPU_AppleA12,
    CPU_AppleA13,
    CPU_AppleA14,
#endif
#ifdef ISPC_XE_ENABLED
    GPU_SKL,
    GPU_TGLLP,
    GPU_ACM_G10,
    GPU_ACM_G11,
    GPU_ACM_G12,
    GPU_PVC,
    GPU_MTL_M,
    GPU_MTL_P,
#endif
    sizeofDeviceType
} DeviceType;

// This map is used to verify features available for supported CPUs
// and is used to filter target dependent intrisics and report an error.
// This mechanism is not precise and doesn't take into account flavors
// of AVX512, for example.
// The following LLVM files were used as reference:
// CPU Features: <llvm>/lib/Support/X86TargetParser.cpp
// X86 Intrinsics: <llvm>/include/llvm/IR/IntrinsicsX86.td
std::map<DeviceType, std::set<std::string>> CPUFeatures = {
    {CPU_x86_64, {"mmx", "sse", "sse2"}},
    {CPU_Bonnell, {"mmx", "sse", "sse2", "ssse3"}},
    {CPU_Core2, {"mmx", "sse", "sse2", "ssse3"}},
    {CPU_Penryn, {"mmx", "sse", "sse2", "ssse3", "sse41"}},
    {CPU_Nehalem, {"mmx", "sse", "sse2", "ssse3", "sse41", "sse42"}},
    {CPU_PS4, {"mmx", "sse", "sse2", "ssse3", "sse41", "sse42", "avx"}},
    {CPU_SandyBridge, {"mmx", "sse", "sse2", "ssse3", "sse41", "sse42", "avx"}},
    {CPU_IvyBridge, {"mmx", "sse", "sse2", "ssse3", "sse41", "sse42", "avx"}},
    {CPU_Haswell, {"mmx", "sse", "sse2", "ssse3", "sse41", "sse42", "avx", "avx2"}},
    {CPU_Broadwell, {"mmx", "sse", "sse2", "ssse3", "sse41", "sse42", "avx", "avx2"}},
    {CPU_Skylake, {"mmx", "sse", "sse2", "ssse3", "sse41", "sse42", "avx", "avx2"}},
    {CPU_KNL, {"mmx", "sse", "sse2", "ssse3", "sse41", "sse42", "avx", "avx2", "avx512"}},
    {CPU_SKX, {"mmx", "sse", "sse2", "ssse3", "sse41", "sse42", "avx", "avx2", "avx512"}},
    {CPU_ICL, {"mmx", "sse", "sse2", "ssse3", "sse41", "sse42", "avx", "avx2", "avx512", "avx512_vnni"}},
    {CPU_Silvermont, {"mmx", "sse", "sse2", "ssse3", "sse41", "sse42"}},
    {CPU_ICX, {"mmx", "sse", "sse2", "ssse3", "sse41", "sse42", "avx", "avx2", "avx512", "avx512_vnni"}},
    {CPU_TGL, {"mmx", "sse", "sse2", "ssse3", "sse41", "sse42", "avx", "avx2", "avx512", "avx512_vnni"}},
    {CPU_ADL, {"mmx", "sse", "sse2", "ssse3", "sse41", "sse42", "avx", "avx2", "avx_vnni"}},
#if ISPC_LLVM_VERSION >= ISPC_LLVM_16_0
    {CPU_MTL, {"mmx", "sse", "sse2", "ssse3", "sse41", "sse42", "avx", "avx2", "avx_vnni"}},
#endif
#if ISPC_LLVM_VERSION >= ISPC_LLVM_14_0
    {CPU_SPR, {"mmx", "sse", "sse2", "ssse3", "sse41", "sse42", "avx", "avx2", "avx512", "avx_vnni", "avx512_vnni"}},
#endif
    {CPU_ZNVER1, {"mmx", "sse", "sse2", "ssse3", "sse41", "sse42", "avx", "avx2"}},
    {CPU_ZNVER2, {"mmx", "sse", "sse2", "ssse3", "sse41", "sse42", "avx", "avx2"}},
    {CPU_ZNVER3, {"mmx", "sse", "sse2", "ssse3", "sse41", "sse42", "avx", "avx2"}},
// TODO: Add features for remaining CPUs if valid.
#ifdef ISPC_ARM_ENABLED
    {CPU_CortexA9, {}},
    {CPU_CortexA15, {}},
    {CPU_CortexA35, {}},
    {CPU_CortexA53, {}},
    {CPU_CortexA57, {}},
    {CPU_AppleA7, {}},
    {CPU_AppleA10, {}},
    {CPU_AppleA11, {}},
    {CPU_AppleA12, {}},
    {CPU_AppleA13, {}},
    {CPU_AppleA14, {}},
#endif
#ifdef ISPC_XE_ENABLED
    {GPU_SKL, {}},
    {GPU_TGLLP, {}},
    {GPU_ACM_G10, {}},
    {GPU_ACM_G11, {}},
    {GPU_ACM_G12, {}},
    {GPU_PVC, {}},
    {GPU_MTL_M, {}},
    {GPU_MTL_P, {}},
#endif
};

class AllCPUs {
  private:
    std::vector<std::vector<std::string>> names;
    std::vector<std::set<DeviceType>> compat;

    std::set<DeviceType> Set(int type, ...) {
        std::set<DeviceType> retn;
        va_list args;

        retn.insert((DeviceType)type);
        va_start(args, type);
        while ((type = va_arg(args, int)) != CPU_None)
            retn.insert((DeviceType)type);
        va_end(args);

        return retn;
    }

  public:
    AllCPUs() {
        names = std::vector<std::vector<std::string>>(sizeofDeviceType);
        compat = std::vector<std::set<DeviceType>>(sizeofDeviceType);

        names[CPU_None].push_back("");

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

        names[CPU_Skylake].push_back("skylake");

        names[CPU_KNL].push_back("knl");

        names[CPU_SKX].push_back("skx");

        names[CPU_ICL].push_back("icelake-client");
        names[CPU_ICL].push_back("icl");

        names[CPU_ICX].push_back("icelake-server");
        names[CPU_ICX].push_back("icx");
        names[CPU_TGL].push_back("tigerlake");
        names[CPU_TGL].push_back("tgl");
        names[CPU_ADL].push_back("alderlake");
        names[CPU_ADL].push_back("adl");
#if ISPC_LLVM_VERSION >= ISPC_LLVM_16_0
        names[CPU_MTL].push_back("meteorlake");
        names[CPU_MTL].push_back("mtl");
#endif
// Note for SPR support: it's suported starting LLVM 12, but the support is complete and correct
// only starting LLVM 15.0.3. LLVM 14 is ok if it's properly patched.
#if ISPC_LLVM_VERSION >= ISPC_LLVM_14_0
        names[CPU_SPR].push_back("sapphirerapids");
        names[CPU_SPR].push_back("spr");
#endif

        names[CPU_ZNVER1].push_back("znver1");
        names[CPU_ZNVER2].push_back("znver2");
        names[CPU_ZNVER2].push_back("ps5");
        names[CPU_ZNVER3].push_back("znver3");

#ifdef ISPC_ARM_ENABLED
        names[CPU_CortexA9].push_back("cortex-a9");
        names[CPU_CortexA15].push_back("cortex-a15");
        names[CPU_CortexA35].push_back("cortex-a35");
        names[CPU_CortexA53].push_back("cortex-a53");
        names[CPU_CortexA57].push_back("cortex-a57");

        names[CPU_AppleA7].push_back("apple-a7");
        names[CPU_AppleA10].push_back("apple-a10");
        names[CPU_AppleA11].push_back("apple-a11");
        names[CPU_AppleA12].push_back("apple-a12");
        names[CPU_AppleA13].push_back("apple-a13");
        names[CPU_AppleA14].push_back("apple-a14");
#endif

#ifdef ISPC_XE_ENABLED
        names[GPU_SKL].push_back("skl");
        names[GPU_TGLLP].push_back("tgllp");
        names[GPU_TGLLP].push_back("dg1");
        // ACM 512EU version
        names[GPU_ACM_G10].push_back("acm-g10");
        // ACM 128EU version
        names[GPU_ACM_G11].push_back("acm-g11");
        // ACM 256EU version
        names[GPU_ACM_G12].push_back("acm-g12");
        names[GPU_PVC].push_back("pvc");
        names[GPU_MTL_M].push_back("mtl-m");
        names[GPU_MTL_P].push_back("mtl-p");
#endif

        Assert(names.size() == sizeofDeviceType);

        compat[CPU_Silvermont] =
            Set(CPU_x86_64, CPU_Bonnell, CPU_Penryn, CPU_Core2, CPU_Nehalem, CPU_Silvermont, CPU_None);

        compat[CPU_KNL] = Set(CPU_KNL, CPU_x86_64, CPU_Bonnell, CPU_Penryn, CPU_Core2, CPU_Nehalem, CPU_Silvermont,
                              CPU_SandyBridge, CPU_IvyBridge, CPU_Haswell, CPU_Broadwell, CPU_Skylake, CPU_None);

        compat[CPU_SKX] = Set(CPU_SKX, CPU_x86_64, CPU_Bonnell, CPU_Penryn, CPU_Core2, CPU_Nehalem, CPU_Silvermont,
                              CPU_SandyBridge, CPU_IvyBridge, CPU_Haswell, CPU_Broadwell, CPU_Skylake, CPU_None);
#if ISPC_LLVM_VERSION >= ISPC_LLVM_14_0
        compat[CPU_SPR] = Set(CPU_SPR, CPU_x86_64, CPU_Bonnell, CPU_Penryn, CPU_Core2, CPU_Nehalem, CPU_Silvermont,
                              CPU_SandyBridge, CPU_IvyBridge, CPU_Haswell, CPU_Broadwell, CPU_Skylake, CPU_SKX, CPU_ICL,
                              CPU_ICX, CPU_TGL, CPU_ADL, CPU_None);
#endif
#if ISPC_LLVM_VERSION >= ISPC_LLVM_16_0
        compat[CPU_MTL] =
            Set(CPU_MTL, CPU_x86_64, CPU_Bonnell, CPU_Penryn, CPU_Core2, CPU_Nehalem, CPU_Silvermont, CPU_SandyBridge,
                CPU_IvyBridge, CPU_Haswell, CPU_Broadwell, CPU_Skylake, CPU_ADL, CPU_None);
#endif
        compat[CPU_ADL] = Set(CPU_ADL, CPU_x86_64, CPU_Bonnell, CPU_Penryn, CPU_Core2, CPU_Nehalem, CPU_Silvermont,
                              CPU_SandyBridge, CPU_IvyBridge, CPU_Haswell, CPU_Broadwell, CPU_Skylake, CPU_None);
        compat[CPU_TGL] =
            Set(CPU_TGL, CPU_x86_64, CPU_Bonnell, CPU_Penryn, CPU_Core2, CPU_Nehalem, CPU_Silvermont, CPU_SandyBridge,
                CPU_IvyBridge, CPU_Haswell, CPU_Broadwell, CPU_Skylake, CPU_SKX, CPU_ICL, CPU_ICX, CPU_None);
        compat[CPU_ICX] =
            Set(CPU_ICX, CPU_x86_64, CPU_Bonnell, CPU_Penryn, CPU_Core2, CPU_Nehalem, CPU_Silvermont, CPU_SandyBridge,
                CPU_IvyBridge, CPU_Haswell, CPU_Broadwell, CPU_Skylake, CPU_SKX, CPU_ICL, CPU_None);

        compat[CPU_ICL] =
            Set(CPU_ICL, CPU_x86_64, CPU_Bonnell, CPU_Penryn, CPU_Core2, CPU_Nehalem, CPU_Silvermont, CPU_SandyBridge,
                CPU_IvyBridge, CPU_Haswell, CPU_Broadwell, CPU_Skylake, CPU_SKX, CPU_None);

        compat[CPU_ZNVER3] =
            Set(CPU_x86_64, CPU_Bonnell, CPU_Penryn, CPU_Core2, CPU_Nehalem, CPU_Silvermont, CPU_SandyBridge,
                CPU_IvyBridge, CPU_Haswell, CPU_Broadwell, CPU_Skylake, CPU_ZNVER3, CPU_None);
        compat[CPU_ZNVER2] =
            Set(CPU_x86_64, CPU_Bonnell, CPU_Penryn, CPU_Core2, CPU_Nehalem, CPU_Silvermont, CPU_SandyBridge,
                CPU_IvyBridge, CPU_Haswell, CPU_Broadwell, CPU_Skylake, CPU_ZNVER2, CPU_None);
        compat[CPU_ZNVER1] =
            Set(CPU_x86_64, CPU_Bonnell, CPU_Penryn, CPU_Core2, CPU_Nehalem, CPU_Silvermont, CPU_SandyBridge,
                CPU_IvyBridge, CPU_Haswell, CPU_Broadwell, CPU_Skylake, CPU_ZNVER1, CPU_None);
        compat[CPU_Skylake] = Set(CPU_x86_64, CPU_Bonnell, CPU_Penryn, CPU_Core2, CPU_Nehalem, CPU_Silvermont,
                                  CPU_SandyBridge, CPU_IvyBridge, CPU_Haswell, CPU_Broadwell, CPU_Skylake, CPU_None);
        compat[CPU_Broadwell] = Set(CPU_x86_64, CPU_Bonnell, CPU_Penryn, CPU_Core2, CPU_Nehalem, CPU_Silvermont,
                                    CPU_SandyBridge, CPU_IvyBridge, CPU_Haswell, CPU_Broadwell, CPU_Skylake, CPU_None);
        compat[CPU_Haswell] = Set(CPU_x86_64, CPU_Bonnell, CPU_Penryn, CPU_Core2, CPU_Nehalem, CPU_Silvermont,
                                  CPU_SandyBridge, CPU_IvyBridge, CPU_Haswell, CPU_Broadwell, CPU_Skylake, CPU_None);
        compat[CPU_IvyBridge] = Set(CPU_x86_64, CPU_Bonnell, CPU_Penryn, CPU_Core2, CPU_Nehalem, CPU_Silvermont,
                                    CPU_SandyBridge, CPU_IvyBridge, CPU_None);
        compat[CPU_SandyBridge] =
            Set(CPU_x86_64, CPU_Bonnell, CPU_Penryn, CPU_Core2, CPU_Nehalem, CPU_Silvermont, CPU_SandyBridge, CPU_None);
        compat[CPU_PS4] = Set(CPU_x86_64, CPU_Bonnell, CPU_Penryn, CPU_Core2, CPU_Nehalem, CPU_Silvermont,
                              CPU_SandyBridge, CPU_PS4, CPU_None);
        compat[CPU_Nehalem] =
            Set(CPU_x86_64, CPU_Bonnell, CPU_Penryn, CPU_Core2, CPU_Nehalem, CPU_Silvermont, CPU_None);
        compat[CPU_Penryn] = Set(CPU_x86_64, CPU_Bonnell, CPU_Penryn, CPU_Core2, CPU_Nehalem, CPU_Silvermont, CPU_None);
        compat[CPU_Core2] = Set(CPU_x86_64, CPU_Bonnell, CPU_Core2, CPU_None);
        compat[CPU_Bonnell] = Set(CPU_x86_64, CPU_Bonnell, CPU_Core2, CPU_None);

        compat[CPU_x86_64] = Set(CPU_x86_64, CPU_None);

#ifdef ISPC_ARM_ENABLED
        compat[CPU_CortexA15] = Set(CPU_CortexA9, CPU_CortexA15, CPU_None);
        compat[CPU_CortexA9] = Set(CPU_CortexA9, CPU_None);
        compat[CPU_CortexA35] = Set(CPU_CortexA35, CPU_None);
        compat[CPU_CortexA53] = Set(CPU_CortexA53, CPU_None);
        compat[CPU_CortexA57] = Set(CPU_CortexA57, CPU_None);
        compat[CPU_AppleA7] = Set(CPU_AppleA7, CPU_None);
        compat[CPU_AppleA10] = Set(CPU_AppleA10, CPU_None);
        compat[CPU_AppleA11] = Set(CPU_AppleA11, CPU_None);
        compat[CPU_AppleA12] = Set(CPU_AppleA12, CPU_None);
        compat[CPU_AppleA13] = Set(CPU_AppleA13, CPU_None);
        compat[CPU_AppleA14] = Set(CPU_AppleA14, CPU_None);
#endif

#ifdef ISPC_XE_ENABLED
        compat[GPU_SKL] = Set(GPU_SKL, CPU_None);
        compat[GPU_TGLLP] = Set(GPU_TGLLP, GPU_SKL, CPU_None);
        compat[GPU_ACM_G10] = Set(GPU_ACM_G10, GPU_ACM_G11, GPU_ACM_G12, GPU_TGLLP, GPU_SKL, CPU_None);
        compat[GPU_ACM_G11] = Set(GPU_ACM_G10, GPU_ACM_G11, GPU_ACM_G12, GPU_TGLLP, GPU_SKL, CPU_None);
        compat[GPU_ACM_G12] = Set(GPU_ACM_G10, GPU_ACM_G11, GPU_ACM_G12, GPU_TGLLP, GPU_SKL, CPU_None);
        compat[GPU_PVC] = Set(GPU_PVC, GPU_SKL, CPU_None);
        compat[GPU_MTL_M] =
            Set(GPU_MTL_M, GPU_MTL_P, GPU_ACM_G10, GPU_ACM_G11, GPU_ACM_G12, GPU_ACM_G11, GPU_TGLLP, GPU_SKL, CPU_None);
        compat[GPU_MTL_P] =
            Set(GPU_MTL_M, GPU_MTL_P, GPU_ACM_G10, GPU_ACM_G11, GPU_ACM_G12, GPU_ACM_G11, GPU_TGLLP, GPU_SKL, CPU_None);
#endif
    }

    std::string HumanReadableListOfNames() {
        std::stringstream CPUs;
        for (int i = CPU_x86_64; i < sizeofDeviceType; i++) {
            CPUs << names[i][0];
            if (names[i].size() > 1) {
                CPUs << " (synonyms: " << names[i][1];
                for (int j = 2, je = names[i].size(); j < je; j++)
                    CPUs << ", " << names[i][j];
                CPUs << ")";
            }
            if (i < sizeofDeviceType - 1)
                CPUs << ", ";
        }
        return CPUs.str();
    }

    std::string &GetDefaultNameFromType(DeviceType type) {
        Assert((type >= CPU_None) && (type < sizeofDeviceType));
        return names[type][0];
    }

    DeviceType GetTypeFromName(std::string name) {
        DeviceType retn = CPU_None;

        for (int i = 1; (retn == CPU_None) && (i < sizeofDeviceType); i++)
            for (int j = 0, je = names[i].size(); (retn == CPU_None) && (j < je); j++)
                if (!name.compare(names[i][j]))
                    retn = (DeviceType)i;
        return retn;
    }

    bool BackwardCompatible(DeviceType what, DeviceType with) {
        Assert((what > CPU_None) && (what < sizeofDeviceType));
        Assert((with > CPU_None) && (with < sizeofDeviceType));
        return compat[what].find(with) != compat[what].end();
    }
};

Target::Target(Arch arch, const char *cpu, ISPCTarget ispc_target, PICLevel picLevel, MCModel code_model,
               bool printTarget)
    : m_target(nullptr), m_targetMachine(nullptr), m_dataLayout(nullptr), m_valid(false), m_ispc_target(ispc_target),
      m_isa(SSE2), m_arch(Arch::none), m_is32Bit(true), m_cpu(""), m_attributes(""), m_tf_attributes(nullptr),
      m_nativeVectorWidth(-1), m_nativeVectorAlignment(-1), m_dataTypeWidth(-1), m_vectorWidth(-1),
      m_picLevel(picLevel), m_codeModel(code_model), m_maskingIsFree(false), m_maskBitCount(-1),
      m_hasDotProductVNNI(false), m_hasHalfConverts(false), m_hasHalfFullSupport(false), m_hasRand(false),
      m_hasGather(false), m_hasScatter(false), m_hasTranscendentals(false), m_hasTrigonometry(false),
      m_hasRsqrtd(false), m_hasRcpd(false), m_hasVecPrefetch(false), m_hasSaturatingArithmetic(false),
      m_hasFp16Support(false), m_hasFp64Support(true), m_warnings(0) {
    DeviceType CPUID = CPU_None, CPUfromISA = CPU_None;
    AllCPUs a;
    std::string featuresString;

    if (cpu) {
        CPUID = a.GetTypeFromName(cpu);
        if (CPUID == CPU_None) {
            Error(SourcePos(),
                  "Error: Device type \"%s\" unknown. Supported"
                  " devices: %s.",
                  cpu, a.HumanReadableListOfNames().c_str());
            return;
        }
    }

    if (m_ispc_target == ISPCTarget::none) {
        // If a CPU was specified explicitly, try to pick the best
        // possible ISA based on that.
        switch (CPUID) {
        case CPU_None: {
            // No CPU and no ISA, so use system info to figure out
            // what this CPU supports.
            m_ispc_target = lGetSystemISA();
            std::string target_string = ISPCTargetToString(m_ispc_target);
            Warning(SourcePos(),
                    "No --target specified on command-line."
                    " Using default system target \"%s\".",
                    target_string.c_str());
            break;
        }

#ifdef ISPC_ARM_ENABLED
        case CPU_CortexA9:
        case CPU_CortexA15:
        case CPU_CortexA35:
        case CPU_CortexA53:
        case CPU_CortexA57:
        case CPU_AppleA7:
        case CPU_AppleA10:
        case CPU_AppleA11:
        case CPU_AppleA12:
        case CPU_AppleA13:
        case CPU_AppleA14:
            m_ispc_target = ISPCTarget::neon_i32x4;
            break;
#endif

#ifdef ISPC_XE_ENABLED
        case GPU_SKL:
            m_ispc_target = ISPCTarget::gen9_x16;
            break;
        case GPU_TGLLP:
            m_ispc_target = ISPCTarget::xelp_x16;
            break;
        case GPU_ACM_G10:
        case GPU_ACM_G11:
        case GPU_ACM_G12:
            m_ispc_target = ISPCTarget::xehpg_x16;
            break;
        case GPU_PVC:
            m_ispc_target = ISPCTarget::xehpc_x16;
            break;
        case GPU_MTL_M:
        case GPU_MTL_P:
            m_ispc_target = ISPCTarget::xelpg_x16;
            break;
#endif

        case CPU_KNL:
            m_ispc_target = ISPCTarget::avx512knl_x16;
            break;

#if ISPC_LLVM_VERSION >= ISPC_LLVM_14_0
        case CPU_SPR:
#endif
        case CPU_TGL:
        case CPU_ICX:
        case CPU_ICL:
        case CPU_SKX:
            m_ispc_target = ISPCTarget::avx512skx_x16;
            break;

#if ISPC_LLVM_VERSION >= ISPC_LLVM_16_0
        case CPU_MTL:
#endif
        case CPU_ADL:
        case CPU_ZNVER3:
        case CPU_ZNVER1:
        case CPU_ZNVER2:
        case CPU_Skylake:
        case CPU_Broadwell:
        case CPU_Haswell:
            m_ispc_target = ISPCTarget::avx2_i32x8;
            break;

        case CPU_IvyBridge:
        case CPU_SandyBridge:
            m_ispc_target = ISPCTarget::avx1_i32x8;
            break;

        case CPU_Penryn:
            m_ispc_target = ISPCTarget::sse41_i32x4;
            break;
        case CPU_Nehalem:
        case CPU_Silvermont:
            m_ispc_target = ISPCTarget::sse4_i32x4;
            break;

        case CPU_PS4:
            m_ispc_target = ISPCTarget::avx1_i32x4;
            break;

        default:
            m_ispc_target = ISPCTarget::sse2_i32x4;
            break;
        }
        if (CPUID != CPU_None) {
            std::string target_string = ISPCTargetToString(m_ispc_target);
            Warning(SourcePos(),
                    "No --target specified on command-line."
                    " Using ISA \"%s\" based on specified device \"%s\".",
                    target_string.c_str(), cpu);
        }
    }

    if (m_ispc_target == ISPCTarget::host) {
        m_ispc_target = lGetSystemISA();
    }

    if (arch == Arch::none) {
#ifdef ISPC_ARM_ENABLED
        if (ISPCTargetIsNeon(m_ispc_target)) {
#if defined(__arm__)
            arch = Arch::arm;
#else
            arch = Arch::aarch64;
#endif
        } else
#endif
#if ISPC_XE_ENABLED
            if (ISPCTargetIsGen(m_ispc_target)) {
            arch = Arch::xe64;
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
    // For Xe target we do not need to create target/targetMachine
    if (this->m_target == nullptr && !ISPCTargetIsGen(m_ispc_target)) {
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
    if (!lIsTargetValidforArch(m_ispc_target, arch)) {
        std::string str_arch = ArchToString(arch);
        std::string target_string = ISPCTargetToString(m_ispc_target);
        Error(SourcePos(), "arch = %s and target = %s is not a valid combination.", str_arch.c_str(),
              target_string.c_str());
        return;
    }

    // FP16 support for Xe and Arm. For x86 set is individually for appropriate targets.
    if (ISPCTargetIsGen(m_ispc_target) || ISPCTargetIsNeon(m_ispc_target)) {
        m_hasFp16Support = true;
    }

#ifdef ISPC_XE_ENABLED
    if ((ISPCTargetIsGen(m_ispc_target)) &&
        (CPUID == GPU_TGLLP || CPUID == GPU_ACM_G10 || CPUID == GPU_ACM_G11 || CPUID == GPU_ACM_G12)) {
        m_hasFp64Support = false;
    }

    // In case of Xe target addressing should correspond to host addressing. Otherwise pointers will not work.
    if (arch == Arch::xe64) {
        g->opt.force32BitAddressing = false;
    }
#endif

    // Check math library
    if (g->mathLib == Globals::MathLib::Math_SVML && !ISPCTargetIsX86(m_ispc_target)) {
        Error(SourcePos(), "SVML math library is supported for x86 targets only.");
        return;
    }

    // Check default LLVM generated targets
    bool unsupported_target = false;
    switch (m_ispc_target) {
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
        CPUfromISA = CPU_x86_64;
        break;
    case ISPCTarget::sse4_i8x16:
    case ISPCTarget::sse41_i8x16:
        this->m_isa = (m_ispc_target == ISPCTarget::sse4_i8x16) ? Target::SSE42 : Target::SSE41;
        CPUfromISA = (m_ispc_target == ISPCTarget::sse4_i8x16) ? CPU_Nehalem : CPU_Penryn;
        this->m_nativeVectorWidth = 16;
        this->m_nativeVectorAlignment = 16;
        this->m_dataTypeWidth = 8;
        this->m_vectorWidth = 16;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 8;
        break;
    case ISPCTarget::sse4_i16x8:
    case ISPCTarget::sse41_i16x8:
        this->m_isa = (m_ispc_target == ISPCTarget::sse4_i16x8) ? Target::SSE42 : Target::SSE41;
        CPUfromISA = (m_ispc_target == ISPCTarget::sse4_i16x8) ? CPU_Nehalem : CPU_Penryn;
        this->m_nativeVectorWidth = 8;
        this->m_nativeVectorAlignment = 16;
        this->m_dataTypeWidth = 16;
        this->m_vectorWidth = 8;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 16;
        break;
    case ISPCTarget::sse4_i32x4:
    case ISPCTarget::sse41_i32x4:
        this->m_isa = (m_ispc_target == ISPCTarget::sse4_i32x4) ? Target::SSE42 : Target::SSE41;
        CPUfromISA = (m_ispc_target == ISPCTarget::sse4_i32x4) ? CPU_Nehalem : CPU_Penryn;
        this->m_nativeVectorWidth = 4;
        this->m_nativeVectorAlignment = 16;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 4;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
        break;
    case ISPCTarget::sse4_i32x8:
    case ISPCTarget::sse41_i32x8:
        this->m_isa = (m_ispc_target == ISPCTarget::sse4_i32x8) ? Target::SSE42 : Target::SSE41;
        CPUfromISA = (m_ispc_target == ISPCTarget::sse4_i32x8) ? CPU_Nehalem : CPU_Penryn;
        this->m_nativeVectorWidth = 4;
        this->m_nativeVectorAlignment = 16;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 8;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
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
    case ISPCTarget::avx2_i8x32:
        this->m_isa = Target::AVX2;
        this->m_nativeVectorWidth = 32;
        this->m_nativeVectorAlignment = 32;
        this->m_dataTypeWidth = 8;
        this->m_vectorWidth = 32;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 8;
        this->m_hasHalfConverts = true;
        this->m_hasRand = true;
        this->m_hasGather = true;
        CPUfromISA = CPU_Haswell;
        break;
    case ISPCTarget::avx2_i16x16:
        this->m_isa = Target::AVX2;
        this->m_nativeVectorWidth = 16;
        this->m_nativeVectorAlignment = 32;
        this->m_dataTypeWidth = 16;
        this->m_vectorWidth = 16;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 16;
        this->m_hasHalfConverts = true;
        this->m_hasRand = true;
        this->m_hasGather = true;
        CPUfromISA = CPU_Haswell;
        break;
    case ISPCTarget::avx2_i32x4:
    case ISPCTarget::avx2vnni_i32x4:
        this->m_isa = (m_ispc_target == ISPCTarget::avx2vnni_i32x4) ? Target::AVX2VNNI : Target::AVX2;
        this->m_nativeVectorWidth = 8;
        this->m_nativeVectorAlignment = 32;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 4;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
        this->m_hasHalfConverts = true;
        this->m_hasRand = true;
        this->m_hasGather = true;
        this->m_hasDotProductVNNI = (m_ispc_target == ISPCTarget::avx2vnni_i32x4) ? true : false;
        CPUfromISA = (m_ispc_target == ISPCTarget::avx2vnni_i32x4) ? CPU_ADL : CPU_Haswell;
        break;
    case ISPCTarget::avx2_i32x8:
    case ISPCTarget::avx2vnni_i32x8:
        this->m_isa = (m_ispc_target == ISPCTarget::avx2vnni_i32x8) ? Target::AVX2VNNI : Target::AVX2;
        this->m_nativeVectorWidth = 8;
        this->m_nativeVectorAlignment = 32;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 8;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
        this->m_hasHalfConverts = true;
        this->m_hasRand = true;
        this->m_hasGather = true;
        this->m_hasDotProductVNNI = (m_ispc_target == ISPCTarget::avx2vnni_i32x8) ? true : false;
        CPUfromISA = (m_ispc_target == ISPCTarget::avx2vnni_i32x8) ? CPU_ADL : CPU_Haswell;
        break;
    case ISPCTarget::avx2_i32x16:
    case ISPCTarget::avx2vnni_i32x16:
        this->m_isa = (m_ispc_target == ISPCTarget::avx2vnni_i32x16) ? Target::AVX2VNNI : Target::AVX2;
        this->m_nativeVectorWidth = 16;
        this->m_nativeVectorAlignment = 32;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 16;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
        this->m_hasHalfConverts = true;
        this->m_hasRand = true;
        this->m_hasGather = true;
        this->m_hasDotProductVNNI = (m_ispc_target == ISPCTarget::avx2vnni_i32x16) ? true : false;
        CPUfromISA = (m_ispc_target == ISPCTarget::avx2vnni_i32x16) ? CPU_ADL : CPU_Haswell;
        break;
    case ISPCTarget::avx2_i64x4:
        this->m_isa = Target::AVX2;
        this->m_nativeVectorWidth = 8; /* native vector width in terms of floats */
        this->m_nativeVectorAlignment = 32;
        this->m_dataTypeWidth = 64;
        this->m_vectorWidth = 4;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 64;
        this->m_hasHalfConverts = true;
        this->m_hasRand = true;
        this->m_hasGather = true;
        CPUfromISA = CPU_Haswell;
        break;
    case ISPCTarget::avx512knl_x16:
        this->m_isa = Target::KNL_AVX512;
        this->m_nativeVectorWidth = 16;
        this->m_nativeVectorAlignment = 64;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 16;
        this->m_maskingIsFree = true;
        this->m_maskBitCount = 1;
        this->m_hasHalfConverts = true;
        this->m_hasRand = true;
        this->m_hasGather = this->m_hasScatter = true;
        this->m_hasTranscendentals = false;
        // For MIC it is set to true due to performance reasons. The option should be tested.
        this->m_hasTrigonometry = false;
        this->m_hasRsqrtd = this->m_hasRcpd = true;
        this->m_hasVecPrefetch = false;
        CPUfromISA = CPU_KNL;
        break;
    case ISPCTarget::avx512skx_x4:
    case ISPCTarget::avx512icl_x4:
        this->m_isa = (m_ispc_target == ISPCTarget::avx512icl_x4) ? Target::ICL_AVX512 : Target::SKX_AVX512;
        this->m_nativeVectorWidth = 16;
        this->m_nativeVectorAlignment = 64;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 4;
        this->m_maskingIsFree = true;
        this->m_maskBitCount = 1;
        this->m_hasHalfConverts = true;
        this->m_hasRand = true;
        this->m_hasGather = this->m_hasScatter = true;
        this->m_hasTranscendentals = false;
        this->m_hasTrigonometry = false;
        this->m_hasRsqrtd = this->m_hasRcpd = true;
        this->m_hasVecPrefetch = false;
        this->m_hasDotProductVNNI = (m_ispc_target == ISPCTarget::avx512icl_x4) ? true : false;
        CPUfromISA = (m_ispc_target == ISPCTarget::avx512icl_x4) ? CPU_ICL : CPU_SKX;
        this->m_funcAttributes.push_back(std::make_pair("prefer-vector-width", "256"));
        this->m_funcAttributes.push_back(std::make_pair("min-legal-vector-width", "256"));
        break;
    case ISPCTarget::avx512skx_x8:
    case ISPCTarget::avx512icl_x8:
        this->m_isa = (m_ispc_target == ISPCTarget::avx512icl_x8) ? Target::ICL_AVX512 : Target::SKX_AVX512;
        this->m_nativeVectorWidth = 16;
        this->m_nativeVectorAlignment = 64;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 8;
        this->m_maskingIsFree = true;
        this->m_maskBitCount = 1;
        this->m_hasHalfConverts = true;
        this->m_hasRand = true;
        this->m_hasGather = this->m_hasScatter = true;
        this->m_hasTranscendentals = false;
        this->m_hasTrigonometry = false;
        this->m_hasRsqrtd = this->m_hasRcpd = true;
        this->m_hasVecPrefetch = false;
        this->m_hasDotProductVNNI = (m_ispc_target == ISPCTarget::avx512icl_x8) ? true : false;
        CPUfromISA = (m_ispc_target == ISPCTarget::avx512icl_x8) ? CPU_ICL : CPU_SKX;
        this->m_funcAttributes.push_back(std::make_pair("prefer-vector-width", "256"));
        this->m_funcAttributes.push_back(std::make_pair("min-legal-vector-width", "256"));
        break;
    case ISPCTarget::avx512skx_x16:
    case ISPCTarget::avx512icl_x16:
        this->m_isa = (m_ispc_target == ISPCTarget::avx512icl_x16) ? Target::ICL_AVX512 : Target::SKX_AVX512;
        this->m_nativeVectorWidth = 16;
        this->m_nativeVectorAlignment = 64;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 16;
        this->m_maskingIsFree = true;
        this->m_maskBitCount = 1;
        this->m_hasHalfConverts = true;
        this->m_hasRand = true;
        this->m_hasGather = this->m_hasScatter = true;
        this->m_hasTranscendentals = false;
        this->m_hasTrigonometry = false;
        this->m_hasRsqrtd = this->m_hasRcpd = true;
        this->m_hasVecPrefetch = false;
        this->m_hasDotProductVNNI = (m_ispc_target == ISPCTarget::avx512icl_x16) ? true : false;
        CPUfromISA = (m_ispc_target == ISPCTarget::avx512icl_x16) ? CPU_ICL : CPU_SKX;
        if (g->opt.disableZMM) {
            this->m_funcAttributes.push_back(std::make_pair("prefer-vector-width", "256"));
            this->m_funcAttributes.push_back(std::make_pair("min-legal-vector-width", "256"));
        } else {
            this->m_funcAttributes.push_back(std::make_pair("prefer-vector-width", "512"));
            this->m_funcAttributes.push_back(std::make_pair("min-legal-vector-width", "512"));
        }
        break;
    case ISPCTarget::avx512skx_x64:
    case ISPCTarget::avx512icl_x64:
        // This target is enabled only for LLVM 10.0 and later
        // because LLVM requires a number of fixes, which are
        // committed to LLVM 11.0 and can be applied to 10.0, but not
        // earlier versions.
        this->m_isa = (m_ispc_target == ISPCTarget::avx512icl_x64) ? Target::ICL_AVX512 : Target::SKX_AVX512;
        this->m_nativeVectorWidth = 64;
        this->m_nativeVectorAlignment = 64;
        this->m_dataTypeWidth = 8;
        this->m_vectorWidth = 64;
        this->m_maskingIsFree = true;
        this->m_maskBitCount = 1;
        this->m_hasHalfConverts = true;
        this->m_hasRand = true;
        this->m_hasGather = this->m_hasScatter = true;
        this->m_hasTranscendentals = false;
        this->m_hasTrigonometry = false;
        this->m_hasRsqrtd = this->m_hasRcpd = false;
        this->m_hasVecPrefetch = false;
        this->m_hasDotProductVNNI = (m_ispc_target == ISPCTarget::avx512icl_x64) ? true : false;
        CPUfromISA = (m_ispc_target == ISPCTarget::avx512icl_x64) ? CPU_ICL : CPU_SKX;
        break;
    case ISPCTarget::avx512skx_x32:
    case ISPCTarget::avx512icl_x32:
        // This target is enabled only for LLVM 10.0 and later
        // because LLVM requires a number of fixes, which are
        // committed to LLVM 11.0 and can be applied to 10.0, but not
        // earlier versions.
        this->m_isa = (m_ispc_target == ISPCTarget::avx512icl_x32) ? Target::ICL_AVX512 : Target::SKX_AVX512;
        this->m_nativeVectorWidth = 64;
        this->m_nativeVectorAlignment = 64;
        this->m_dataTypeWidth = 16;
        this->m_vectorWidth = 32;
        this->m_maskingIsFree = true;
        this->m_maskBitCount = 1;
        this->m_hasHalfConverts = true;
        this->m_hasRand = true;
        this->m_hasGather = this->m_hasScatter = true;
        this->m_hasTranscendentals = false;
        this->m_hasTrigonometry = false;
        this->m_hasRsqrtd = this->m_hasRcpd = false;
        this->m_hasVecPrefetch = false;
        this->m_hasDotProductVNNI = (m_ispc_target == ISPCTarget::avx512icl_x32) ? true : false;
        CPUfromISA = (m_ispc_target == ISPCTarget::avx512icl_x32) ? CPU_ICL : CPU_SKX;
        break;
#if ISPC_LLVM_VERSION >= ISPC_LLVM_14_0
    case ISPCTarget::avx512spr_x4:
        this->m_isa = Target::SPR_AVX512;
        this->m_nativeVectorWidth = 16;
        this->m_nativeVectorAlignment = 64;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 4;
        this->m_maskingIsFree = true;
        this->m_maskBitCount = 1;
        this->m_hasHalfConverts = true;
        this->m_hasHalfFullSupport = true;
        this->m_hasRand = true;
        this->m_hasGather = this->m_hasScatter = true;
        this->m_hasTranscendentals = false;
        this->m_hasTrigonometry = false;
        this->m_hasRsqrtd = this->m_hasRcpd = true;
        this->m_hasVecPrefetch = false;
        this->m_hasFp16Support = true;
        this->m_hasDotProductVNNI = true;
        CPUfromISA = CPU_SPR;
        this->m_funcAttributes.push_back(std::make_pair("prefer-vector-width", "256"));
        this->m_funcAttributes.push_back(std::make_pair("min-legal-vector-width", "256"));
        break;
    case ISPCTarget::avx512spr_x8:
        this->m_isa = Target::SPR_AVX512;
        this->m_nativeVectorWidth = 16;
        this->m_nativeVectorAlignment = 64;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 8;
        this->m_maskingIsFree = true;
        this->m_maskBitCount = 1;
        this->m_hasHalfConverts = true;
        this->m_hasHalfFullSupport = true;
        this->m_hasRand = true;
        this->m_hasGather = this->m_hasScatter = true;
        this->m_hasTranscendentals = false;
        this->m_hasTrigonometry = false;
        this->m_hasRsqrtd = this->m_hasRcpd = true;
        this->m_hasVecPrefetch = false;
        this->m_hasFp16Support = true;
        this->m_hasDotProductVNNI = true;
        CPUfromISA = CPU_SPR;
        this->m_funcAttributes.push_back(std::make_pair("prefer-vector-width", "256"));
        this->m_funcAttributes.push_back(std::make_pair("min-legal-vector-width", "256"));
        break;
    case ISPCTarget::avx512spr_x16:
        this->m_isa = Target::SPR_AVX512;
        this->m_nativeVectorWidth = 16;
        this->m_nativeVectorAlignment = 64;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 16;
        this->m_maskingIsFree = true;
        this->m_maskBitCount = 1;
        this->m_hasHalfConverts = true;
        this->m_hasHalfFullSupport = true;
        this->m_hasRand = true;
        this->m_hasGather = this->m_hasScatter = true;
        this->m_hasTranscendentals = false;
        this->m_hasTrigonometry = false;
        this->m_hasRsqrtd = this->m_hasRcpd = true;
        this->m_hasVecPrefetch = false;
        this->m_hasFp16Support = true;
        this->m_hasDotProductVNNI = true;
        CPUfromISA = CPU_SPR;
        if (g->opt.disableZMM) {
            this->m_funcAttributes.push_back(std::make_pair("prefer-vector-width", "256"));
            this->m_funcAttributes.push_back(std::make_pair("min-legal-vector-width", "256"));
        } else {
            this->m_funcAttributes.push_back(std::make_pair("prefer-vector-width", "512"));
            this->m_funcAttributes.push_back(std::make_pair("min-legal-vector-width", "512"));
        }
        break;
    case ISPCTarget::avx512spr_x64:
        this->m_isa = Target::SPR_AVX512;
        this->m_nativeVectorWidth = 64;
        this->m_nativeVectorAlignment = 64;
        this->m_dataTypeWidth = 8;
        this->m_vectorWidth = 64;
        this->m_maskingIsFree = true;
        this->m_maskBitCount = 1;
        this->m_hasHalfConverts = true;
        this->m_hasHalfFullSupport = true;
        this->m_hasRand = true;
        this->m_hasGather = this->m_hasScatter = true;
        this->m_hasTranscendentals = false;
        this->m_hasTrigonometry = false;
        this->m_hasRsqrtd = this->m_hasRcpd = false;
        this->m_hasVecPrefetch = false;
        this->m_hasFp16Support = true;
        this->m_hasDotProductVNNI = true;
        CPUfromISA = CPU_SPR;
        break;
    case ISPCTarget::avx512spr_x32:
        this->m_isa = Target::SPR_AVX512;
        this->m_nativeVectorWidth = 64;
        this->m_nativeVectorAlignment = 64;
        this->m_dataTypeWidth = 16;
        this->m_vectorWidth = 32;
        this->m_maskingIsFree = true;
        this->m_maskBitCount = 1;
        this->m_hasHalfConverts = true;
        this->m_hasHalfFullSupport = true;
        this->m_hasRand = true;
        this->m_hasGather = this->m_hasScatter = true;
        this->m_hasTranscendentals = false;
        this->m_hasTrigonometry = false;
        this->m_hasRsqrtd = this->m_hasRcpd = false;
        this->m_hasVecPrefetch = false;
        this->m_hasFp16Support = true;
        this->m_hasDotProductVNNI = true;
        CPUfromISA = CPU_SPR;
        break;
#else
    case ISPCTarget::avx512spr_x4:
    case ISPCTarget::avx512spr_x8:
    case ISPCTarget::avx512spr_x16:
    case ISPCTarget::avx512spr_x32:
    case ISPCTarget::avx512spr_x64:
        unsupported_target = true;
        break;
#endif
#ifdef ISPC_ARM_ENABLED
    case ISPCTarget::neon_i8x16:
        this->m_isa = Target::NEON;
        this->m_nativeVectorWidth = 16;
        this->m_nativeVectorAlignment = 16;
        this->m_dataTypeWidth = 8;
        this->m_vectorWidth = 16;
        this->m_hasHalfConverts = true; // ??
        // https://github.com/ispc/ispc/issues/2052
        // AArch64 disables Coherent Control Flow optimization because of a bug in
        // LLVM aarch64 back-end that reduces the efficiency of simplifyCFG.
        // Branches added by CCF can only be removed after the back-end formed
        // fused-multiply-adds.  This reduces the quality of code as most of scalar
        // optimizations will not apply.
        // FIXME: Consider turning this optimization back on after
        // https://reviews.llvm.org/D100963 gets committed to LLVM-13.
        // This note applies to all NEON targets below.
        this->m_maskingIsFree = (arch == Arch::aarch64);
        this->m_maskBitCount = 8;
        break;
    case ISPCTarget::neon_i16x8:
        this->m_isa = Target::NEON;
        this->m_nativeVectorWidth = 8;
        this->m_nativeVectorAlignment = 16;
        this->m_dataTypeWidth = 16;
        this->m_vectorWidth = 8;
        this->m_hasHalfConverts = true; // ??
        this->m_maskingIsFree = (arch == Arch::aarch64);
        this->m_maskBitCount = 16;
        break;
    case ISPCTarget::neon_i32x4:
        this->m_isa = Target::NEON;
        this->m_nativeVectorWidth = 4;
        this->m_nativeVectorAlignment = 16;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 4;
        this->m_hasHalfConverts = true; // ??
        // TODO: m_hasHalfFullSupport is not enabled here, as it's only supported starting from ARMv8.2 / Cortex A75
        // We nned to defferentiate ARM target with and without float16 support.
        this->m_maskingIsFree = (arch == Arch::aarch64);
        this->m_maskBitCount = 32;
        break;
    case ISPCTarget::neon_i32x8:
        this->m_isa = Target::NEON;
        this->m_nativeVectorWidth = 4;
        this->m_nativeVectorAlignment = 16;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 8;
        this->m_hasHalfConverts = true; // ??
        this->m_maskingIsFree = (arch == Arch::aarch64);
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
#ifdef ISPC_WASM_ENABLED
    case ISPCTarget::wasm_i32x4:
        this->m_isa = Target::WASM;
        this->m_nativeVectorWidth = 4;
        this->m_nativeVectorAlignment = 16;
        this->m_dataTypeWidth = 32;
        this->m_vectorWidth = 4;
        this->m_hasHalfConverts = false;
        this->m_hasHalfFullSupport = false;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
        this->m_hasTranscendentals = false;
        this->m_hasTrigonometry = false;
        this->m_hasRcpd = false;
        this->m_hasRsqrtd = false;
        this->m_hasScatter = false;
        this->m_hasGather = false;
        this->m_hasVecPrefetch = false;
        break;
#else
    case ISPCTarget::wasm_i32x4:
        unsupported_target = true;
        break;
#endif
#ifdef ISPC_XE_ENABLED
    case ISPCTarget::gen9_x8:
        this->m_isa = Target::GEN9;
        this->m_nativeVectorWidth = 8;
        this->m_nativeVectorAlignment = 64;
        this->m_vectorWidth = 8;
        this->m_dataTypeWidth = 32;
        this->m_hasHalfConverts = true;
        this->m_hasHalfFullSupport = true;
        this->m_maskingIsFree = true;
        this->m_maskBitCount = 1;
        this->m_hasSaturatingArithmetic = true;
        this->m_hasTranscendentals = true;
        this->m_hasTrigonometry = true;
        this->m_hasGather = this->m_hasScatter = true;
        CPUfromISA = GPU_SKL;
        break;
    case ISPCTarget::xelp_x8:
        this->m_isa = Target::XELP;
        this->m_nativeVectorWidth = 8;
        this->m_nativeVectorAlignment = 64;
        this->m_vectorWidth = 8;
        this->m_dataTypeWidth = 32;
        this->m_hasHalfConverts = true;
        this->m_hasHalfFullSupport = true;
        this->m_maskingIsFree = true;
        this->m_maskBitCount = 1;
        this->m_hasSaturatingArithmetic = true;
        this->m_hasTranscendentals = true;
        this->m_hasTrigonometry = true;
        this->m_hasGather = this->m_hasScatter = true;
        CPUfromISA = GPU_TGLLP;
        break;
    case ISPCTarget::gen9_x16:
        this->m_isa = Target::GEN9;
        this->m_nativeVectorWidth = 16;
        this->m_nativeVectorAlignment = 64;
        this->m_vectorWidth = 16;
        this->m_dataTypeWidth = 32;
        this->m_hasHalfConverts = true;
        this->m_hasHalfFullSupport = true;
        this->m_maskingIsFree = true;
        this->m_maskBitCount = 1;
        this->m_hasSaturatingArithmetic = true;
        this->m_hasTranscendentals = true;
        this->m_hasTrigonometry = true;
        this->m_hasGather = this->m_hasScatter = true;
        CPUfromISA = GPU_SKL;
        break;
    case ISPCTarget::xelp_x16:
        this->m_isa = Target::XELP;
        this->m_nativeVectorWidth = 16;
        this->m_nativeVectorAlignment = 64;
        this->m_vectorWidth = 16;
        this->m_dataTypeWidth = 32;
        this->m_hasHalfConverts = true;
        this->m_hasHalfFullSupport = true;
        this->m_maskingIsFree = true;
        this->m_maskBitCount = 1;
        this->m_hasSaturatingArithmetic = true;
        this->m_hasTranscendentals = true;
        this->m_hasTrigonometry = true;
        this->m_hasGather = this->m_hasScatter = true;
        CPUfromISA = GPU_TGLLP;
        break;
    case ISPCTarget::xehpg_x8:
        this->m_isa = Target::XEHPG;
        this->m_nativeVectorWidth = 8;
        this->m_nativeVectorAlignment = 64;
        this->m_vectorWidth = 8;
        this->m_dataTypeWidth = 32;
        this->m_hasHalfConverts = true;
        this->m_hasHalfFullSupport = true;
        this->m_maskingIsFree = true;
        this->m_maskBitCount = 1;
        this->m_hasSaturatingArithmetic = true;
        this->m_hasTranscendentals = true;
        this->m_hasTrigonometry = true;
        this->m_hasGather = this->m_hasScatter = true;
        CPUfromISA = GPU_ACM_G11;
        break;
    case ISPCTarget::xehpg_x16:
        this->m_isa = Target::XEHPG;
        this->m_nativeVectorWidth = 16;
        this->m_nativeVectorAlignment = 64;
        this->m_vectorWidth = 16;
        this->m_dataTypeWidth = 32;
        this->m_hasHalfConverts = true;
        this->m_hasHalfFullSupport = true;
        this->m_maskingIsFree = true;
        this->m_maskBitCount = 1;
        this->m_hasSaturatingArithmetic = true;
        this->m_hasTranscendentals = true;
        this->m_hasTrigonometry = true;
        this->m_hasGather = this->m_hasScatter = true;
        CPUfromISA = GPU_ACM_G11;
        break;
    case ISPCTarget::xehpc_x16:
        this->m_isa = Target::XEHPC;
        this->m_nativeVectorWidth = 16;
        this->m_nativeVectorAlignment = 64;
        this->m_vectorWidth = 16;
        this->m_dataTypeWidth = 32;
        this->m_hasHalfConverts = true;
        this->m_hasHalfFullSupport = true;
        this->m_maskingIsFree = true;
        this->m_maskBitCount = 1;
        this->m_hasSaturatingArithmetic = true;
        this->m_hasTranscendentals = true;
        this->m_hasTrigonometry = true;
        this->m_hasGather = this->m_hasScatter = true;
        CPUfromISA = GPU_PVC;
        break;
    case ISPCTarget::xehpc_x32:
        this->m_isa = Target::XEHPC;
        this->m_nativeVectorWidth = 32;
        this->m_nativeVectorAlignment = 64;
        this->m_vectorWidth = 32;
        this->m_dataTypeWidth = 32;
        this->m_hasHalfConverts = true;
        this->m_hasHalfFullSupport = true;
        this->m_maskingIsFree = true;
        this->m_maskBitCount = 1;
        this->m_hasSaturatingArithmetic = true;
        this->m_hasTranscendentals = true;
        this->m_hasTrigonometry = true;
        this->m_hasGather = this->m_hasScatter = true;
        CPUfromISA = GPU_PVC;
        break;
    case ISPCTarget::xelpg_x8:
        this->m_isa = Target::XELPG;
        this->m_nativeVectorWidth = 8;
        this->m_nativeVectorAlignment = 64;
        this->m_vectorWidth = 8;
        this->m_dataTypeWidth = 32;
        this->m_hasHalfConverts = true;
        this->m_hasHalfFullSupport = true;
        this->m_maskingIsFree = true;
        this->m_maskBitCount = 1;
        this->m_hasSaturatingArithmetic = true;
        this->m_hasTranscendentals = true;
        this->m_hasTrigonometry = true;
        this->m_hasGather = this->m_hasScatter = true;
        CPUfromISA = GPU_MTL_P;
        break;
    case ISPCTarget::xelpg_x16:
        this->m_isa = Target::XELPG;
        this->m_nativeVectorWidth = 16;
        this->m_nativeVectorAlignment = 64;
        this->m_vectorWidth = 16;
        this->m_dataTypeWidth = 32;
        this->m_hasHalfConverts = true;
        this->m_hasHalfFullSupport = true;
        this->m_maskingIsFree = true;
        this->m_maskBitCount = 1;
        this->m_hasSaturatingArithmetic = true;
        this->m_hasTranscendentals = true;
        this->m_hasTrigonometry = true;
        this->m_hasGather = this->m_hasScatter = true;
        CPUfromISA = GPU_MTL_P;
        break;
#else
    case ISPCTarget::gen9_x8:
    case ISPCTarget::gen9_x16:
    case ISPCTarget::xelp_x8:
    case ISPCTarget::xelp_x16:
    case ISPCTarget::xehpg_x8:
    case ISPCTarget::xehpg_x16:
    case ISPCTarget::xehpc_x16:
    case ISPCTarget::xehpc_x32:
    case ISPCTarget::xelpg_x8:
    case ISPCTarget::xelpg_x16:
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
        std::string target_string = "Problem with target (" + ISPCTargetToString(m_ispc_target) + ")";
        FATAL(target_string.c_str());
    }

    // Enable ISA-dependnent warnings
    switch (this->m_isa) {
    case Target::SSE2:
    case Target::SSE41:
    case Target::SSE42:
    case Target::AVX:
        this->setWarning(PerfWarningType::CVTUIntFloat);
        this->setWarning(PerfWarningType::DIVModInt);
        this->setWarning(PerfWarningType::VariableShiftRight);
        break;
    case Target::AVX2:
    case Target::AVX2VNNI:
        this->setWarning(PerfWarningType::CVTUIntFloat);
        this->setWarning(PerfWarningType::CVTUIntFloat16);
        this->setWarning(PerfWarningType::DIVModInt);
        break;
    case Target::KNL_AVX512:
    case Target::SKX_AVX512:
    case Target::ICL_AVX512:
    case Target::SPR_AVX512:
        this->setWarning(PerfWarningType::DIVModInt);
        break;
    default:
        // Fall through
        ;
    }

#if defined(ISPC_ARM_ENABLED)
    if ((CPUID == CPU_None) && ISPCTargetIsNeon(m_ispc_target)) {
        if (arch == Arch::arm) {
            CPUID = CPU_CortexA9;
        } else if (arch == Arch::aarch64) {
            if (g->target_os == TargetOS::ios) {
                CPUID = CPU_AppleA7;
            } else if (g->target_os == TargetOS::macos) {
                // Open source LLVM doesn't has definition for M1 CPU, so use the latest iPhone CPU.
                CPUID = CPU_AppleA14;
            } else {
                CPUID = CPU_CortexA35;
            }
        } else {
            UNREACHABLE();
        }
    }
#endif

    if (CPUID == CPU_None) {
        cpu = a.GetDefaultNameFromType(CPUfromISA).c_str();
    } else {
        if ((CPUfromISA != CPU_None) && !a.BackwardCompatible(CPUID, CPUfromISA)) {
            std::string target_string = ISPCTargetToString(m_ispc_target);
            Error(SourcePos(),
                  "The requested device (%s) is incompatible"
                  " with the device required for %s target (%s)",
                  cpu, target_string.c_str(), a.GetDefaultNameFromType(CPUfromISA).c_str());
            return;
        }
        cpu = a.GetDefaultNameFromType(CPUID).c_str();
    }

    this->m_cpu = cpu;

    if (!error) {
        // Create TargetMachine
        std::string triple = GetTripleString();

        // The last validity check to ensure that supported for this target was enabled in the build.
        if (!g->target_registry->isSupported(m_ispc_target, g->target_os, arch)) {
            std::string target_string = ISPCTargetToString(m_ispc_target);
            std::string arch_str = ArchToString(arch);
            std::string os_str = OSToString(g->target_os);
            Error(SourcePos(), "%s target for %s on %s is not supported in current build.", target_string.c_str(),
                  arch_str.c_str(), os_str.c_str());
            return;
        }

#if ISPC_LLVM_VERSION >= ISPC_LLVM_16_0
        std::optional<llvm::Reloc::Model> relocModel;
        std::optional<llvm::CodeModel::Model> mcModel;
#else
        llvm::Optional<llvm::Reloc::Model> relocModel;
        llvm::Optional<llvm::CodeModel::Model> mcModel;
#endif

        if (m_picLevel == PICLevel::SmallPIC || m_picLevel == PICLevel::BigPIC) {
            relocModel = llvm::Reloc::PIC_;
        }
        switch (m_codeModel) {
        case MCModel::Small:
            mcModel = llvm::CodeModel::Small;
            break;
        case MCModel::Large:
            mcModel = llvm::CodeModel::Large;
            break;
        case ispc::MCModel::Default:
            break;
        }
        llvm::TargetOptions options;
#ifdef ISPC_ARM_ENABLED
        options.FloatABIType = llvm::FloatABI::Hard;
        if (arch == Arch::arm) {
            if (g->target_os == TargetOS::custom_linux) {
                this->m_funcAttributes.push_back(std::make_pair("target-features", "+crypto,+fp-armv8,+neon,+sha2"));
            } else {
                this->m_funcAttributes.push_back(std::make_pair("target-features", "+neon,+fp16"));
            }
            featuresString = "+neon,+fp16";
        } else if (arch == Arch::aarch64) {
            if (g->target_os == TargetOS::custom_linux) {
                this->m_funcAttributes.push_back(
                    std::make_pair("target-features", "+aes,+crc,+crypto,+fp-armv8,+neon,+sha2"));
            } else {
                this->m_funcAttributes.push_back(std::make_pair("target-features", "+neon"));
            }
            featuresString = "+neon";
        }
#endif

        // Support 'i64' and 'double' types in cm
        if (isXeTarget())
            featuresString += "+longlong";

        if (g->opt.disableFMA == false)
            options.AllowFPOpFusion = llvm::FPOpFusion::Fast;

        // For Xe target we do not need to create target/targetMachine
        if (!isXeTarget()) {
            m_targetMachine =
                m_target->createTargetMachine(triple, m_cpu, featuresString, options, relocModel, mcModel);
            Assert(m_targetMachine != nullptr);

            // Set Optimization level for llvm codegen based on Optimization level
            // requested by user via ISPC Optimization Flag. Mapping is :
            // ISPC O0 -> Codegen O0
            // ISPC O1,O2,O3,default -> Codegen O3
            CodegenOptLevel cOptLevel = CodegenOptLevel::Aggressive;
            switch (g->codegenOptLevel) {
            case Globals::CodegenOptLevel::None:
                cOptLevel = CodegenOptLevel::None;
                break;

            case Globals::CodegenOptLevel::Aggressive:
                cOptLevel = CodegenOptLevel::Aggressive;
                break;
            }
            m_targetMachine->setOptLevel(cOptLevel);

            m_targetMachine->Options.MCOptions.AsmVerbose = true;

            // Change default version of generated DWARF.
            if (g->generateDWARFVersion) {
                m_targetMachine->Options.MCOptions.DwarfVersion = g->generateDWARFVersion;
            }
        }
        // Initialize TargetData/DataLayout in 3 steps.
        // 1. Get default data layout first
        std::string dl_string;
        if (m_targetMachine != nullptr)
            dl_string = m_targetMachine->createDataLayout().getStringRepresentation();
        if (isXeTarget())
            dl_string = m_arch == Arch::xe64 ? "e-p:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:"
                                               "256-v512:512-v1024:1024-n8:16:32:64"
                                             : "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:"
                                               "256-v512:512-v1024:1024-n8:16:32:64";

        // 2. Finally set member data
        m_dataLayout = new llvm::DataLayout(dl_string);

        // Set is32Bit
        // This indicates if we are compiling for 32 bit platform and can assume 32 bit runtime.

        this->m_is32Bit = (getDataLayout()->getPointerSize() == 4);

        // TO-DO : Revisit addition of "target-features" and "target-cpu" for ARM support.
#if ISPC_LLVM_VERSION >= ISPC_LLVM_14_0
        llvm::AttrBuilder *fattrBuilder = new llvm::AttrBuilder(*g->ctx);
#else
        llvm::AttrBuilder *fattrBuilder = new llvm::AttrBuilder();
#endif
#ifdef ISPC_ARM_ENABLED
        if (m_isa == Target::NEON)
            fattrBuilder->addAttribute("target-cpu", this->m_cpu);
#endif
        for (auto const &f_attr : m_funcAttributes)
            fattrBuilder->addAttribute(f_attr.first, f_attr.second);
        this->m_tf_attributes = fattrBuilder;

        Assert(this->m_vectorWidth <= ISPC_MAX_NVEC);
    }

    m_valid = !error;

    if (printTarget) {
        if (!isXeTarget()) {
            printf("Target Triple: %s\n", m_targetMachine->getTargetTriple().str().c_str());
            printf("Target CPU: %s\n", m_targetMachine->getTargetCPU().str().c_str());
            printf("Target Feature String: %s\n", m_targetMachine->getTargetFeatureString().str().c_str());
        } else {
            printf("Target Triple: %s\n", this->GetTripleString().c_str());
            printf("Target GPU: %s\n", this->getCPU().c_str());
            printf("Target Feature String: %s\n", featuresString.c_str());
        }
    }

    return;
}

Target::~Target() {
    if (m_dataLayout)
        delete m_dataLayout;
    if (m_tf_attributes)
        delete m_tf_attributes;
    if (m_targetMachine)
        delete m_targetMachine;
}

bool Target::checkIntrinsticSupport(llvm::StringRef name, SourcePos pos) {
    if (name.consume_front("llvm.") == false) {
        return false;
    }
    // x86 specific intrinsics are verified using 'CPUFeatures'.
    // TODO: Add relevant information to 'CPUFeatures' for non x86 targets.
    if (name.consume_front("x86.") == true) {
        if (!ISPCTargetIsX86(m_ispc_target)) {
            Error(pos, "LLVM intrinsic \"%s\" supported only on \"x86\" target architecture.", name.data());
            return false;
        }
        AllCPUs a;
        std::string featureName = name.substr(0, name.find('.')).str();
        if (CPUFeatures[a.GetTypeFromName(this->getCPU())].count(featureName) == 0) {
            Error(pos, "Target specific LLVM intrinsic \"%s\" not supported on \"%s\" CPU.", name.data(),
                  this->getCPU().c_str());
            return false;
        }
    } else if (name.consume_front("arm.") == true) {
        if (m_arch != Arch::arm) {
            Error(pos, "LLVM intrinsic \"%s\" supported only on \"arm\" target architecture.", name.data());
            return false;
        }
        // TODO: Check 'CPUFeatures'.
    } else if (name.consume_front("aarch64.") == true) {
        if (m_arch != Arch::aarch64) {
            Error(pos, "LLVM intrinsic \"%s\" supported only on \"aarch64\" target architecture.", name.data());
            return false;
        }
        // TODO: Check 'CPUFeatures'.
    } else if (name.consume_front("wasm.") == true) {
        // TODO: Add Condition in future if relevant.
        // For now, returning 'true'.
        return true;
    }
    return true;
}

std::string Target::SupportedCPUs() {
    AllCPUs a;
    return a.HumanReadableListOfNames();
}

std::string Target::GetTripleString() const {
    llvm::Triple triple;
    switch (g->target_os) {
    case TargetOS::windows:
        if (m_arch == Arch::x86) {
            triple.setArchName("i686");
        } else if (m_arch == Arch::x86_64) {
            triple.setArchName("x86_64");
        } else if (m_arch == Arch::arm) {
            Error(SourcePos(), "arm (32 bit) is not supported on Windows, use aarch64 instead.");
            exit(1);
        } else if (m_arch == Arch::aarch64) {
            triple.setArchName("aarch64");
        } else if (m_arch == Arch::xe64) {
            triple.setArchName("spir64");
        } else {
            Error(SourcePos(), "Unknown arch.");
            exit(1);
        }
#ifdef ISPC_XE_ENABLED
        if (m_arch == Arch::xe64) {
            //"spir64-unknown-unknown"
            triple.setVendor(llvm::Triple::VendorType::UnknownVendor);
            triple.setOS(llvm::Triple::OSType::UnknownOS);
            return triple.str();
        }
#endif
        //"x86_64-pc-windows-msvc"
        triple.setVendor(llvm::Triple::VendorType::PC);
        triple.setOS(llvm::Triple::OSType::Win32);
        triple.setEnvironment(llvm::Triple::EnvironmentType::MSVC);
        break;
    case TargetOS::custom_linux:
    case TargetOS::linux:
        if (m_arch == Arch::x86) {
            triple.setArchName("i686");
        } else if (m_arch == Arch::x86_64) {
            triple.setArchName("x86_64");
        } else if (m_arch == Arch::arm) {
            triple.setArchName("armv7");
        } else if (m_arch == Arch::aarch64) {
            triple.setArchName("aarch64");
        } else if (m_arch == Arch::xe64) {
            triple.setArchName("spir64");
        } else {
            Error(SourcePos(), "Unknown arch.");
            exit(1);
        }
#ifdef ISPC_XE_ENABLED
        if (m_arch == Arch::xe64) {
            //"spir64-unknown-unknown"
            triple.setVendor(llvm::Triple::VendorType::UnknownVendor);
            triple.setOS(llvm::Triple::OSType::UnknownOS);
            return triple.str();
        }
#endif
        triple.setVendor(llvm::Triple::VendorType::UnknownVendor);
        triple.setOS(llvm::Triple::OSType::Linux);
        if (m_arch == Arch::x86 || m_arch == Arch::x86_64 || m_arch == Arch::aarch64 || m_arch == Arch::xe64) {
            triple.setEnvironment(llvm::Triple::EnvironmentType::GNU);
        } else if (m_arch == Arch::arm) {
            triple.setEnvironment(llvm::Triple::EnvironmentType::GNUEABIHF);
        } else {
            Error(SourcePos(), "Unknown arch.");
            exit(1);
        }
        break;
    case TargetOS::freebsd:
        if (m_arch == Arch::x86) {
            triple.setArchName("i686");
        } else if (m_arch == Arch::x86_64) {
            triple.setArchName("amd64");
        } else if (m_arch == Arch::arm) {
            triple.setArchName("armv7");
        } else if (m_arch == Arch::aarch64) {
            triple.setArchName("aarch64");
        } else {
            Error(SourcePos(), "Unknown arch.");
            exit(1);
        }
        triple.setVendor(llvm::Triple::VendorType::UnknownVendor);
        triple.setOS(llvm::Triple::OSType::FreeBSD);
        break;
    case TargetOS::macos:
        // asserts
        if (m_arch == Arch::x86_64) {
            triple.setArchName("x86_64");
        } else if (m_arch == Arch::aarch64) {
            triple.setArchName("arm64");
        } else {
            Error(SourcePos(), "macOS target supports only x86_64 and aarch64.");
            exit(1);
        }
        triple.setVendor(llvm::Triple::VendorType::Apple);
        triple.setOS(llvm::Triple::OSType::MacOSX);
        break;
    case TargetOS::android:
        if (m_arch == Arch::x86) {
            triple.setArchName("i686");
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
    case TargetOS::ps5:
        if (m_arch != Arch::x86_64) {
            Error(SourcePos(), "PS5 target supports only x86_64.");
            exit(1);
        }
        // "x86_64-scei-ps4", as "ps5" was not yet officially upstreamed to LLVM.
        triple.setArch(llvm::Triple::ArchType::x86_64);
        triple.setVendor(llvm::Triple::VendorType::SCEI);
        triple.setOS(llvm::Triple::OSType::PS4);
        break;
    case TargetOS::web:
        if (m_arch != Arch::wasm32 && m_arch != Arch::wasm64) {
            Error(SourcePos(), "Web target supports only wasm32 and wasm64.");
            exit(1);
        }
        if (m_arch == Arch::wasm32) {
            triple.setArch(llvm::Triple::ArchType::wasm32);
        } else if (m_arch == Arch::wasm64) {
            triple.setArch(llvm::Triple::ArchType::wasm64);
        }
        triple.setVendor(llvm::Triple::VendorType::UnknownVendor);
        triple.setOS(llvm::Triple::OSType::UnknownOS);
        break;
    case TargetOS::error:
        Error(SourcePos(), "Invalid target OS.");
        exit(1);
    }

    return triple.str();
}

bool Target::useGather() const { return m_hasGather && !g->opt.disableGathers; }
bool Target::useScatter() const { return m_hasScatter && !g->opt.disableScatters; }

// This function returns string representation of ISA for the purpose of
// mangling. And may return any unique string, preferably short, like
// sse4, avx and etc.
const char *Target::ISAToString(ISA isa) {
    switch (isa) {
#ifdef ISPC_ARM_ENABLED
    case Target::NEON:
        return "neon";
#endif
#ifdef ISPC_WASM_ENABLED
    case Target::WASM:
        return "wasm";
#endif
    case Target::SSE2:
        return "sse2";
    case Target::SSE41:
    case Target::SSE42:
        return "sse4";
    case Target::AVX:
        return "avx";
    case Target::AVX2:
        return "avx2";
    case Target::AVX2VNNI:
        return "avx2vnni";
    case Target::KNL_AVX512:
        return "avx512knl";
    case Target::SKX_AVX512:
        return "avx512skx";
    case Target::ICL_AVX512:
        return "avx512icl";
    case Target::SPR_AVX512:
        return "avx512spr";
#ifdef ISPC_XE_ENABLED
    case Target::GEN9:
        return "gen9";
    case Target::XELP:
        return "xelp";
    case Target::XEHPG:
        return "xehpg";
    case Target::XEHPC:
        return "xehpc";
    case Target::XELPG:
        return "xelpg";
#endif
    default:
        FATAL("Unhandled target in ISAToString()");
    }
    return "";
}

const char *Target::GetISAString() const { return ISAToString(m_isa); }

// This function returns string representation of default target corresponding
// to ISA. I.e. for SSE41 it's sse4.1-i32x4, for AVX2 it's avx2-i32x8. This
// string may be used to initialize Target.
const char *Target::ISAToTargetString(ISA isa) {
    switch (isa) {
#ifdef ISPC_ARM_ENABLED
    case Target::NEON:
        return "neon-i32x4";
#endif
#ifdef ISPC_WASM_ENABLED
    case Target::WASM:
        return "wasm-i32x4";
#endif
#ifdef ISPC_XE_ENABLED
    case Target::GEN9:
        return "gen9-x16";
    case Target::XELP:
        return "xelp-x16";
    case Target::XEHPG:
        return "xehpg-x16";
    case Target::XEHPC:
        return "xehpc-x16";
    case Target::XELPG:
        return "xelpg-x16";
#endif
    case Target::SSE2:
        return "sse2-i32x4";
    case Target::SSE41:
        return "sse4.1-i32x4";
    case Target::SSE42:
        return "sse4.2-i32x4";
    case Target::AVX:
        return "avx1-i32x8";
    case Target::AVX2:
        return "avx2-i32x8";
    case Target::AVX2VNNI:
        return "avx2vnni-i32x8";
    case Target::KNL_AVX512:
        return "avx512knl-x16";
    case Target::SKX_AVX512:
        return "avx512skx-x16";
    case Target::ICL_AVX512:
        return "avx512icl-x16";
    case Target::SPR_AVX512:
        return "avx512spr-x16";
    default:
        FATAL("Unhandled target in ISAToTargetString()");
    }
    return "";
}

const char *Target::GetISATargetString() const { return ISAToTargetString(m_isa); }

llvm::Value *Target::SizeOf(llvm::Type *type, llvm::BasicBlock *insertAtEnd) {
    uint64_t byteSize = getDataLayout()->getTypeStoreSize(type);
    if (m_is32Bit || g->opt.force32BitAddressing)
        return LLVMInt32((int32_t)byteSize);
    else
        return LLVMInt64(byteSize);
}

llvm::Value *Target::StructOffset(llvm::Type *type, int element, llvm::BasicBlock *insertAtEnd) {
    llvm::StructType *structType = llvm::dyn_cast<llvm::StructType>(type);
    if (structType == nullptr || structType->isSized() == false) {
        Assert(m->errorCount > 0);
        return nullptr;
    }

    const llvm::StructLayout *sl = getDataLayout()->getStructLayout(structType);
    Assert(sl != nullptr);

    uint64_t offset = sl->getElementOffset(element);
    if (m_is32Bit || g->opt.force32BitAddressing)
        return LLVMInt32((int32_t)offset);
    else
        return LLVMInt64(offset);
}

void Target::markFuncNameWithRegCallPrefix(std::string &funcName) const { funcName = "__regcall3__" + funcName; }

void Target::markFuncWithTargetAttr(llvm::Function *func) {
    if (m_tf_attributes) {
#if ISPC_LLVM_VERSION >= ISPC_LLVM_14_0
        func->addFnAttrs(*m_tf_attributes);
#else
        func->addAttributes(llvm::AttributeList::FunctionIndex, *m_tf_attributes);
#endif
    }
}

void Target::markFuncWithCallingConv(llvm::Function *func) {
    assert("markFuncWithCallingConv is deprecated, use llvm::Function::setCallingConv(llvm::CallingConv) and "
           "FunctionType::GetCallingConv() instead.");
    assert(g->calling_conv != CallingConv::uninitialized);
    if (g->calling_conv == CallingConv::x86_vectorcall) {
        func->setCallingConv(llvm::CallingConv::X86_VectorCall);
        // Add x86 vectorcall changes as a separate commit.
        /*
        // We have to jump through some hoops for x86.
        // In LLVM IR for x86, arguments which are to be passed in registers
        // have to marked with 'InReg' attribue.
        // Rules(Ref : https://docs.microsoft.com/en-us/cpp/cpp/vectorcall?view=vs-2019 )
        // Definitions:
        // Integer Type : it fits in the native register size of the processor for example,
        // 4 bytes on an x86 machine.Integer types include pointer, reference, and struct or union types of 4 bytes or
        less.
        // Vector Type : either a floating - point type for example, a float or double or an SIMD vector type for
        // example, __m128 or __m256.
        // Rules for x86: Integer Type : The first two integer type arguments found in the
        // parameter list from left to right are placed in ECX and EDX, respectively.
        // Vector Type : The first six vector type arguments in order from left to right are passed by value in SSE
        vector registers 0 to 5.
        //The seventh and subsequent vector type arguments are passed on the stack by reference to memory allocated by
        the caller.
        // Observations from Clang(Is there somewhere these rules are mentioned??)
        // Integer Type : After first Integer Type greater than 32 bit, other integer types NOT passed in reg.
        // Vector Type : After 6 Vector Type args, if 2 Integer Type registers are not yet used, VectorType args
        // passed by reference via register - TO DO

        if (m_arch == Arch::x86) {
            llvm::Function::arg_iterator argIter = func->arg_begin();
            llvm::FunctionType *fType = func->getFunctionType();
            int numArgsVecInReg = 0;
            int numArgsIntInReg = 0;
            for (; argIter != func->arg_end(); ++argIter) {
                llvm::Type *argType = fType->getParamType(argIter->getArgNo());
                if (argType->isIntegerTy() || argType->isStructTy() || argType->isPointerTy()) {
                    if (((argType->isIntegerTy()) || (argType->isStructTy())) &&
                        (g->target->getDataLayout()->getTypeSizeInBits(argType) > 32)) {
                        numArgsIntInReg = 2;
                        continue;
                    }

                    numArgsIntInReg++;
                    argIter->addAttr(llvm::Attribute::InReg);
                    continue;
                }
                if (((llvm::dyn_cast<llvm::VectorType>(argType) != nullptr) || argType->isFloatTy() ||
                     argType->isDoubleTy())) {
                    numArgsVecInReg++;
                    argIter->addAttr(llvm::Attribute::InReg);
                }

                if ((numArgsIntInReg == 2) && (numArgsVecInReg == 6))
                    break;
            }
        }*/
    }
}

#ifdef ISPC_XE_ENABLED
Target::XePlatform Target::getXePlatform() const {
    AllCPUs a;
    switch (a.GetTypeFromName(m_cpu)) {
    case GPU_SKL:
        return XePlatform::gen9;
    case GPU_TGLLP:
        return XePlatform::xe_lp;
    case GPU_ACM_G10:
    case GPU_ACM_G11:
    case GPU_ACM_G12:
        return XePlatform::xe_hpg;
    case GPU_PVC:
        return XePlatform::xe_hpc;
    case GPU_MTL_M:
    case GPU_MTL_P:
        return XePlatform::xe_lpg;
    default:
        return XePlatform::gen9;
    }
    return XePlatform::gen9;
}

uint32_t Target::getXeGrfSize() const {
    switch (getXePlatform()) {
    case XePlatform::gen9:
    case XePlatform::xe_lp:
    case XePlatform::xe_hpg:
        return 32;
    case XePlatform::xe_lpg:
        return 32;
    case XePlatform::xe_hpc:
        return 64;
    default:
        return 32;
    }
    return 32;
}

bool Target::hasXePrefetch() const {
    switch (getXePlatform()) {
    case XePlatform::gen9:
    case XePlatform::xe_lp:
        return false;
    default:
        return true;
    }
    return false;
}
#endif

///////////////////////////////////////////////////////////////////////////
// Opt

Opt::Opt() {
    level = 1;
    fastMath = false;
    fastMaskedVload = false;
    force32BitAddressing = true;
    unrollLoops = true;
    disableAsserts = false;
    disableGathers = false;
    disableScatters = false;
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
    disableZMM = false;
    resetFTZ_DAZ = false;
#ifdef ISPC_XE_ENABLED
    disableXeGatherCoalescing = false;
    thresholdForXeGatherCoalescing = 0;
    enableForeachInsideVarying = false;
    emitXeHardwareMask = false;
    enableXeUnsafeMaskedLoad = false;
#endif
}

///////////////////////////////////////////////////////////////////////////
// Globals

Globals::Globals() {
    target_registry = TargetLibRegistry::getTargetLibRegistry();

    mathLib = Globals::MathLib::Math_ISPC;
    codegenOptLevel = Globals::CodegenOptLevel::Aggressive;

    includeStdlib = true;
    runCPP = true;
    onlyCPP = false;
    ignoreCPPErrors = false;
    debugPrint = false;
    debugPM = false;
    debugPMTimeTrace = false;
    astDump = Globals::ASTDumpKind::None;
    dumpFile = false;
    printTarget = false;
    NoOmitFramePointer = false;
    debugIR = -1;
    disableWarnings = false;
    warningsAsErrors = false;
    wrapSignedInt = false;
    quiet = false;
    forceColoredOutput = false;
    disableLineWrap = false;
    emitPerfWarnings = true;
    emitInstrumentation = false;
    noPragmaOnce = false;
    generateDebuggingSymbols = false;
    debugInfoType = Globals::DebugInfoType::None;
    generateDWARFVersion = 3;
    enableFuzzTest = false;
    enableLLVMIntrinsics = false;
    fuzzTestSeed = -1;
    mangleFunctionsWithTarget = false;
    isMultiTargetCompilation = false;
    errorLimit = -1;

    enableTimeTrace = false;
    // set default granularity to 500.
    timeTraceGranularity = 500;
    target = nullptr;
    ctx = new llvm::LLVMContext;

// Opaque pointers mode is supported starting from LLVM 14,
// became default in LLVM 15
#ifdef ISPC_OPAQUE_PTR_MODE
#if ISPC_LLVM_VERSION >= ISPC_LLVM_15_0
// Do nothing, opaque pointers mode is default
#elif ISPC_LLVM_VERSION == ISPC_LLVM_14_0
    // Explicitly enable opaque pointers mode for LLVM 14.0
    ctx->setOpaquePointers(true);
#else
    FATAL("Opaque pointers mode is not supported with this LLVM version!");
#endif
#else
#if ISPC_LLVM_VERSION >= ISPC_LLVM_15_0
    // Explicitly disable opaque pointers starting LLVM 15.0
    ctx->setOpaquePointers(false);
#endif
#endif

#ifdef ISPC_XE_ENABLED
    stackMemSize = 0;
#endif

#ifdef ISPC_HOST_IS_WINDOWS
    _getcwd(currentDirectory, sizeof(currentDirectory));
#else
    if (getcwd(currentDirectory, sizeof(currentDirectory)) == nullptr)
        FATAL("Current directory path is too long!");
#endif
    forceAlignment = -1;
    dllExport = false;

    // Target OS defaults to host OS.
    target_os = GetHostOS();

    if (target_os == TargetOS::windows) {
        debugInfoType = Globals::DebugInfoType::CodeView;
    } else {
        debugInfoType = Globals::DebugInfoType::DWARF;
    }

    // Set calling convention to 'uninitialized'.
    // This needs to be set once target OS is decided.
    calling_conv = CallingConv::uninitialized;
}

///////////////////////////////////////////////////////////////////////////
// SourcePos

SourcePos::SourcePos(const char *n, int fl, int fc, int ll, int lc) {
    name = n;
    if (name == nullptr) {
        if (m != nullptr)
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
    auto [directory, filename] = GetDirectoryAndFileName(g->currentDirectory, name);
    llvm::DIFile *ret = m->diBuilder->createFile(filename, directory);
    return ret;
}

llvm::DINamespace *SourcePos::GetDINamespace() const {
    llvm::DIScope *discope = GetDIFile();
    llvm::DINamespace *ret = m->diBuilder->createNameSpace(discope, "ispc", true);
    return ret;
}

void SourcePos::Print() const {
    printf(" <%s:%d.%d - %d.%d> ", name, first_line, first_column, last_line, last_column);
}

bool SourcePos::operator==(const SourcePos &p2) const {
    return (!strcmp(name, p2.name) && first_line == p2.first_line && first_column == p2.first_column &&
            last_line == p2.last_line && last_column == p2.last_column);
}

SourcePos ispc::Union(const SourcePos &p1, const SourcePos &p2) {
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

BookKeeper &BookKeeper::in() {
    static BookKeeper instance;
    return instance;
}

// Traverse all bookkeeped objects and call delete for every one.
void BookKeeper::freeAll() { BookKeeper::in().freeOne<Traceable>(); }
