/*
  Copyright (c) 2010-2013, Intel Corporation
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
#ifdef ISPC_IS_WINDOWS
  #include <windows.h>
  #include <direct.h>
  #define strcasecmp stricmp
#else
  #include <sys/types.h>
  #include <unistd.h>
#endif
#if defined(LLVM_3_1) || defined(LLVM_3_2)
  #include <llvm/LLVMContext.h>
  #include <llvm/Module.h>
  #include <llvm/Instructions.h>
#else
  #include <llvm/IR/LLVMContext.h>
  #include <llvm/IR/Module.h>
  #include <llvm/IR/Instructions.h>
#endif
#if defined(LLVM_3_1)
  #include <llvm/Analysis/DebugInfo.h>
  #include <llvm/Analysis/DIBuilder.h>
#else
  #include <llvm/DebugInfo.h>
  #include <llvm/DIBuilder.h>
#endif
#include <llvm/Support/Dwarf.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#if defined(LLVM_3_1)
  #include <llvm/Target/TargetData.h>
#elif defined(LLVM_3_2)
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

#ifndef ISPC_IS_WINDOWS
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
#endif // ISPC_IS_WINDOWS


static const char *
lGetSystemISA() {
    int info[4];
    __cpuid(info, 1);

    if ((info[2] & (1 << 28)) != 0) {  // AVX
        // AVX1 for sure....
        // Ivy Bridge?
        if ((info[2] & (1 << 29)) != 0 &&  // F16C
            (info[2] & (1 << 30)) != 0) {  // RDRAND
            // So far, so good.  AVX2?
            // Call cpuid with eax=7, ecx=0
            int info2[4];
            __cpuidex(info2, 7, 0);
            if ((info2[1] & (1 << 5)) != 0)
                return "avx2";
            else
                return "avx1.1";
        }
        // Regular AVX
        return "avx";
    }
    else if ((info[2] & (1 << 19)) != 0)
        return "sse4";
    else if ((info[3] & (1 << 26)) != 0)
        return "sse2";
    else {
        fprintf(stderr, "Unable to detect supported SSE/AVX ISA.  Exiting.\n");
        exit(1);
    }
}


static const char *supportedCPUs[] = {
    "atom", "penryn", "core2", "corei7", "corei7-avx"
#if !defined(LLVM_3_1)
    , "core-avx-i", "core-avx2"
#endif // LLVM 3.2+
};

Target::Target(const char *arch, const char *cpu, const char *isa, bool pic) :
    m_target(NULL),
    m_targetMachine(NULL),
#if defined(LLVM_3_1)
    m_targetData(NULL),
#else
    m_dataLayout(NULL),
#endif
    m_valid(false),
    m_isa(SSE2),
    m_arch(""),
    m_is32Bit(true),
    m_cpu(""),
    m_attributes(""),
#if !defined(LLVM_3_1) && !defined(LLVM_3_2)
    m_tf_attributes(NULL),
#endif
    m_nativeVectorWidth(-1),
    m_vectorWidth(-1),
    m_generatePIC(pic),
    m_maskingIsFree(false),
    m_maskBitCount(-1),
    m_hasHalf(false),
    m_hasRand(false),
    m_hasGather(false),
    m_hasScatter(false),
    m_hasTranscendentals(false)
{
    if (isa == NULL) {
        if (cpu != NULL) {
            // If a CPU was specified explicitly, try to pick the best
            // possible ISA based on that.
            if (!strcmp(cpu, "core-avx2"))
                isa = "avx2";
            else if (!strcmp(cpu, "core-avx-i"))
                isa = "avx1.1";
            else if (!strcmp(cpu, "sandybridge") ||
                !strcmp(cpu, "corei7-avx"))
                isa = "avx";
            else if (!strcmp(cpu, "corei7") ||
                     !strcmp(cpu, "penryn"))
                isa = "sse4";
            else
                isa = "sse2";
            Warning(SourcePos(), "No --target specified on command-line.  "
                    "Using ISA \"%s\" based on specified CPU \"%s\".", isa,
                    cpu);
        }
        else {
            // No CPU and no ISA, so use CPUID to figure out what this CPU
            // supports.
            isa = lGetSystemISA();
            Warning(SourcePos(), "No --target specified on command-line.  "
                    "Using system ISA \"%s\".", isa);
        }
    }

    if (cpu == NULL) {
        std::string hostCPU = llvm::sys::getHostCPUName();
        if (hostCPU.size() > 0)
            cpu = strdup(hostCPU.c_str());
        else {
            Warning(SourcePos(), "Unable to determine host CPU!\n");
            cpu = "generic";
        }
    }
    else {
        bool foundCPU = false;
        for (int i = 0; i < int(sizeof(supportedCPUs) / sizeof(supportedCPUs[0]));
             ++i) {
            if (!strcmp(cpu, supportedCPUs[i])) {
                foundCPU = true;
                break;
            }
        }
        if (foundCPU == false) {
            fprintf(stderr, "Error: CPU type \"%s\" unknown. Supported CPUs: "
                    "%s.\n", cpu, SupportedTargetCPUs().c_str());
            return;
        }
    }

    this->m_cpu = cpu;

    if (arch == NULL)
        arch = "x86-64";

    bool error = false;

    // Make sure the target architecture is a known one; print an error
    // with the valid ones otherwise.
    for (llvm::TargetRegistry::iterator iter = llvm::TargetRegistry::begin();
         iter != llvm::TargetRegistry::end(); ++iter) {
        if (std::string(arch) == iter->getName()) {
            this->m_target = &*iter;
            break;
        }
    }
    if (this->m_target == NULL) {
        fprintf(stderr, "Invalid architecture \"%s\"\nOptions: ", arch);
        llvm::TargetRegistry::iterator iter;
        for (iter = llvm::TargetRegistry::begin();
             iter != llvm::TargetRegistry::end(); ++iter)
            fprintf(stderr, "%s ", iter->getName());
        fprintf(stderr, "\n");
        error = true;
    }
    else {
        this->m_arch = arch;
    }

    // Check default LLVM generated targets
    if (!strcasecmp(isa, "sse2")) {
        this->m_isa = Target::SSE2;
        this->m_nativeVectorWidth = 4;
        this->m_vectorWidth = 4;
        this->m_attributes = "+sse,+sse2,-sse3,-sse41,-sse42,-sse4a,-ssse3,-popcnt";
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
    }
    else if (!strcasecmp(isa, "sse2-x2")) {
        this->m_isa = Target::SSE2;
        this->m_nativeVectorWidth = 4;
        this->m_vectorWidth = 8;
        this->m_attributes = "+sse,+sse2,-sse3,-sse41,-sse42,-sse4a,-ssse3,-popcnt";
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
    }
    else if (!strcasecmp(isa, "sse4")) {
        this->m_isa = Target::SSE4;
        this->m_nativeVectorWidth = 4;
        this->m_vectorWidth = 4;
        // TODO: why not sse42 and popcnt?
        this->m_attributes = "+sse,+sse2,+sse3,+sse41,-sse42,-sse4a,+ssse3,-popcnt,+cmov";
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
    }
    else if (!strcasecmp(isa, "sse4x2") || !strcasecmp(isa, "sse4-x2")) {
        this->m_isa = Target::SSE4;
        this->m_nativeVectorWidth = 4;
        this->m_vectorWidth = 8;
        this->m_attributes = "+sse,+sse2,+sse3,+sse41,-sse42,-sse4a,+ssse3,-popcnt,+cmov";
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
    }
    else if (!strcasecmp(isa, "generic-4")) {
        this->m_isa = Target::GENERIC;
        this->m_nativeVectorWidth = 4;
        this->m_vectorWidth = 4;
        this->m_maskingIsFree = true;
        this->m_maskBitCount = 1;
        this->m_hasHalf = true;
        this->m_hasTranscendentals = true;
        this->m_hasGather = this->m_hasScatter = true;
    }
    else if (!strcasecmp(isa, "generic-8")) {
        this->m_isa = Target::GENERIC;
        this->m_nativeVectorWidth = 8;
        this->m_vectorWidth = 8;
        this->m_maskingIsFree = true;
        this->m_maskBitCount = 1;
        this->m_hasHalf = true;
        this->m_hasTranscendentals = true;
        this->m_hasGather = this->m_hasScatter = true;
    }
    else if (!strcasecmp(isa, "generic-16")) {
        this->m_isa = Target::GENERIC;
        this->m_nativeVectorWidth = 16;
        this->m_vectorWidth = 16;
        this->m_maskingIsFree = true;
        this->m_maskBitCount = 1;
        this->m_hasHalf = true;
        this->m_hasTranscendentals = true;
        this->m_hasGather = this->m_hasScatter = true;
    }
    else if (!strcasecmp(isa, "generic-32")) {
        this->m_isa = Target::GENERIC;
        this->m_nativeVectorWidth = 32;
        this->m_vectorWidth = 32;
        this->m_maskingIsFree = true;
        this->m_maskBitCount = 1;
        this->m_hasHalf = true;
        this->m_hasTranscendentals = true;
        this->m_hasGather = this->m_hasScatter = true;
    }
    else if (!strcasecmp(isa, "generic-64")) {
        this->m_isa = Target::GENERIC;
        this->m_nativeVectorWidth = 64;
        this->m_vectorWidth = 64;
        this->m_maskingIsFree = true;
        this->m_maskBitCount = 1;
        this->m_hasHalf = true;
        this->m_hasTranscendentals = true;
        this->m_hasGather = this->m_hasScatter = true;
    }
    else if (!strcasecmp(isa, "generic-1")) {
        this->m_isa = Target::GENERIC;
        this->m_nativeVectorWidth = 1;
        this->m_vectorWidth = 1;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
    }
    else if (!strcasecmp(isa, "avx") || !strcasecmp(isa, "avx1")) {
        this->m_isa = Target::AVX;
        this->m_nativeVectorWidth = 8;
        this->m_vectorWidth = 8;
        this->m_attributes = "+avx,+popcnt,+cmov";
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
    }
    else if (!strcasecmp(isa, "avx-x2") || !strcasecmp(isa, "avx1-x2")) {
        this->m_isa = Target::AVX;
        this->m_nativeVectorWidth = 8;
        this->m_vectorWidth = 16;
        this->m_attributes = "+avx,+popcnt,+cmov";
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
    }
    else if (!strcasecmp(isa, "avx1.1")) {
        this->m_isa = Target::AVX11;
        this->m_nativeVectorWidth = 8;
        this->m_vectorWidth = 8;
        this->m_attributes = "+avx,+popcnt,+cmov,+f16c,+rdrand";
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
        this->m_hasHalf = true;
#if !defined(LLVM_3_1)
        // LLVM 3.2+ only
        this->m_hasRand = true;
#endif
    }
    else if (!strcasecmp(isa, "avx1.1-x2")) {
        this->m_isa = Target::AVX11;
        this->m_nativeVectorWidth = 8;
        this->m_vectorWidth = 16;
        this->m_attributes = "+avx,+popcnt,+cmov,+f16c,+rdrand";
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
        this->m_hasHalf = true;
#if !defined(LLVM_3_1)
        // LLVM 3.2+ only
        this->m_hasRand = true;
#endif
    }
    else if (!strcasecmp(isa, "avx2")) {
        this->m_isa = Target::AVX2;
        this->m_nativeVectorWidth = 8;
        this->m_vectorWidth = 8;
        this->m_attributes = "+avx2,+popcnt,+cmov,+f16c,+rdrand"
#ifndef LLVM_3_1
            ",+fma"
#endif // !LLVM_3_1
            ;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
        this->m_hasHalf = true;
#if !defined(LLVM_3_1)
        // LLVM 3.2+ only
        this->m_hasRand = true;
        this->m_hasGather = true;
#endif
    }
    else if (!strcasecmp(isa, "avx2-x2")) {
        this->m_isa = Target::AVX2;
        this->m_nativeVectorWidth = 16;
        this->m_vectorWidth = 16;
        this->m_attributes = "+avx2,+popcnt,+cmov,+f16c,+rdrand"
#ifndef LLVM_3_1
            ",+fma"
#endif // !LLVM_3_1
            ;
        this->m_maskingIsFree = false;
        this->m_maskBitCount = 32;
        this->m_hasHalf = true;
#if !defined(LLVM_3_1)
        // LLVM 3.2+ only
        this->m_hasRand = true;
        this->m_hasGather = true;
#endif
    }
    else {
        fprintf(stderr, "Target ISA \"%s\" is unknown.  Choices are: %s\n",
                isa, SupportedTargetISAs());
        error = true;
    }

    if (!error) {
        // Create TargetMachine
        std::string triple = GetTripleString();

        llvm::Reloc::Model relocModel = m_generatePIC ? llvm::Reloc::PIC_ :
            llvm::Reloc::Default;
        std::string featuresString = m_attributes;
        llvm::TargetOptions options;
#if !defined(LLVM_3_1)
        if (g->opt.disableFMA == false)
            options.AllowFPOpFusion = llvm::FPOpFusion::Fast;
#endif // !LLVM_3_1
        m_targetMachine =
            m_target->createTargetMachine(triple, m_cpu, featuresString, options,
                    relocModel);
        Assert(m_targetMachine != NULL);

        m_targetMachine->setAsmVerbosityDefault(true);

        // Initialize TargetData/DataLayout in 3 steps.
        // 1. Get default data layout first
        std::string dl_string;
#if defined(LLVM_3_1)
        dl_string = m_targetMachine->getTargetData()->getStringRepresentation();
#else
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

        // 3. Finally set member data
#if defined(LLVM_3_1)
        m_targetData = new llvm::TargetData(dl_string);
#else
        m_dataLayout = new llvm::DataLayout(dl_string);
#endif

        // Set is32Bit
        // This indicates if we are compiling for 32 bit platform
        // and can assume 32 bit runtime.
        // FIXME: all generic targets are handled as 64 bit, which is incorrect.
        this->m_is32Bit = (getDataLayout()->getPointerSize() == 4);

#if !defined(LLVM_3_1) && !defined(LLVM_3_2)
        // This is LLVM 3.3+ feature.
        // Initialize target-specific "target-feature" attribute.
        if (!m_attributes.empty()) {
            llvm::AttrBuilder attrBuilder;
            attrBuilder.addAttribute("target-features", this->m_attributes);
            this->m_tf_attributes = new llvm::AttributeSet(
                llvm::AttributeSet::get(
                    *g->ctx,
                    llvm::AttributeSet::FunctionIndex,
                    attrBuilder));
        }
#endif

        Assert(this->m_vectorWidth <= ISPC_MAX_NVEC);
    }

    m_valid = !error;

    return;
}


std::string
Target::SupportedTargetCPUs() {
    std::string ret;
    int count = sizeof(supportedCPUs) / sizeof(supportedCPUs[0]);
    for (int i = 0; i < count; ++i) {
        ret += supportedCPUs[i];
        if (i != count - 1)
            ret += ", ";
    }
    return ret;
}


const char *
Target::SupportedTargetArchs() {
    return "x86, x86-64";
}


const char *
Target::SupportedTargetISAs() {
    return "sse2, sse2-x2, sse4, sse4-x2, avx, avx-x2"
        ", avx1.1, avx1.1-x2, avx2, avx2-x2"
        ", generic-1, generic-4, generic-8, generic-16, generic-32";
}


std::string
Target::GetTripleString() const {
    llvm::Triple triple;
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
    else
        triple.setArchName(m_arch);

    return triple.str();
}

const char *
Target::ISAToString(ISA isa) {
    switch (isa) {
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
    case Target::GENERIC:
        return "generic";
    default:
        FATAL("Unhandled target in ISAToString()");
    }
    return "";
}

const char *
Target::GetISAString() const {
    return ISAToString(m_isa);
}


static bool
lGenericTypeLayoutIndeterminate(llvm::Type *type) {
    if (type->isPrimitiveType() || type->isIntegerTy())
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
        llvm::Instruction *gep =
            llvm::GetElementPtrInst::Create(voidPtr, arrayRef, "sizeof_gep",
                                            insertAtEnd);

        if (m_is32Bit || g->opt.force32BitAddressing)
            return new llvm::PtrToIntInst(gep, LLVMTypes::Int32Type,
                                          "sizeof_int", insertAtEnd);
        else
            return new llvm::PtrToIntInst(gep, LLVMTypes::Int64Type,
                                          "sizeof_int", insertAtEnd);
    }

    uint64_t bitSize = getDataLayout()->getTypeSizeInBits(type);

    Assert((bitSize % 8) == 0);
    uint64_t byteSize = bitSize / 8;
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
        llvm::Instruction *gep =
            llvm::GetElementPtrInst::Create(voidPtr, arrayRef, "offset_gep",
                                            insertAtEnd);

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
#if !defined(LLVM_3_1) && !defined(LLVM_3_2)
    if (m_tf_attributes) {
        func->addAttributes(llvm::AttributeSet::FunctionIndex, *m_tf_attributes);
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
    disableWarnings = false;
    warningsAsErrors = false;
    quiet = false;
    forceColoredOutput = false;
    disableLineWrap = false;
    emitPerfWarnings = true;
    emitInstrumentation = false;
    generateDebuggingSymbols = false;
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


llvm::DIFile
SourcePos::GetDIFile() const {
    std::string directory, filename;
    GetDirectoryAndFileName(g->currentDirectory, name, &directory, &filename);
    llvm::DIFile ret = m->diBuilder->createFile(filename, directory);
    Assert(ret.Verify());
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
