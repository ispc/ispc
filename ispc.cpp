/*
  Copyright (c) 2010-2011, Intel Corporation
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
#endif
#include <llvm/LLVMContext.h>
#include <llvm/Module.h>
#include <llvm/Analysis/DIBuilder.h>
#include <llvm/Analysis/DebugInfo.h>
#include <llvm/Support/Dwarf.h>
#include <llvm/Instructions.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Target/TargetData.h>
#if defined(LLVM_3_0) || defined(LLVM_3_0svn) || defined(LLVM_3_1svn)
  #include <llvm/Support/TargetRegistry.h>
  #include <llvm/Support/TargetSelect.h>
#else
  #include <llvm/Target/TargetRegistry.h>
  #include <llvm/Target/TargetSelect.h>
  #include <llvm/Target/SubtargetFeature.h>
#endif
#include <llvm/Support/Host.h>

Globals *g;
Module *m;

///////////////////////////////////////////////////////////////////////////
// Target

bool
Target::GetTarget(const char *arch, const char *cpu, const char *isa,
                  bool pic, Target *t) {
    if (cpu == NULL) {
        std::string hostCPU = llvm::sys::getHostCPUName();
        if (hostCPU.size() > 0)
            cpu = strdup(hostCPU.c_str());
        else {
            fprintf(stderr, "Warning: unable to determine host CPU!\n");
            cpu = "generic";
        }
    }
    t->cpu = cpu;

    if (isa == NULL) {
        if (!strcasecmp(cpu, "atom"))
            isa = "sse2";
#if defined(LLVM_3_0) || defined(LLVM_3_0svn) || defined(LLVM_3_1svn)
        else if (!strcasecmp(cpu, "sandybridge") ||
                 !strcasecmp(cpu, "corei7-avx"))
            isa = "avx";
#endif // LLVM_3_0
        else
            isa = "sse4";
    }
    if (arch == NULL)
        arch = "x86-64";

    bool error = false;

    t->generatePIC = pic;

    // Make sure the target architecture is a known one; print an error
    // with the valid ones otherwise.
    t->target = NULL;
    for (llvm::TargetRegistry::iterator iter = llvm::TargetRegistry::begin();
         iter != llvm::TargetRegistry::end(); ++iter) {
        if (std::string(arch) == iter->getName()) {
            t->target = &*iter;
            break;
        }
    }
    if (t->target == NULL) {
        fprintf(stderr, "Invalid architecture \"%s\"\nOptions: ", arch);
        llvm::TargetRegistry::iterator iter;
        for (iter = llvm::TargetRegistry::begin();
             iter != llvm::TargetRegistry::end(); ++iter)
            fprintf(stderr, "%s ", iter->getName());
        fprintf(stderr, "\n");
        error = true;
    }
    else {
        t->arch = arch;
    }

    if (!strcasecmp(isa, "sse2")) {
        t->isa = Target::SSE2;
        t->nativeVectorWidth = 4;
        t->vectorWidth = 4;
        t->attributes = "+sse,+sse2,-sse3,-sse41,-sse42,-sse4a,-ssse3,-popcnt";
        t->maskingIsFree = false;
        t->allOffMaskIsSafe = false;
        t->maskBitCount = 32;
    }
    else if (!strcasecmp(isa, "sse2-x2")) {
        t->isa = Target::SSE2;
        t->nativeVectorWidth = 4;
        t->vectorWidth = 8;
        t->attributes = "+sse,+sse2,-sse3,-sse41,-sse42,-sse4a,-ssse3,-popcnt";
        t->maskingIsFree = false;
        t->allOffMaskIsSafe = false;
        t->maskBitCount = 32;
    }
    else if (!strcasecmp(isa, "sse4")) {
        t->isa = Target::SSE4;
        t->nativeVectorWidth = 4;
        t->vectorWidth = 4;
        t->attributes = "+sse,+sse2,+sse3,+sse41,-sse42,-sse4a,+ssse3,-popcnt,+cmov";
        t->maskingIsFree = false;
        t->allOffMaskIsSafe = false;
        t->maskBitCount = 32;
    }
    else if (!strcasecmp(isa, "sse4x2") || !strcasecmp(isa, "sse4-x2")) {
        t->isa = Target::SSE4;
        t->nativeVectorWidth = 4;
        t->vectorWidth = 8;
        t->attributes = "+sse,+sse2,+sse3,+sse41,-sse42,-sse4a,+ssse3,-popcnt,+cmov";
        t->maskingIsFree = false;
        t->allOffMaskIsSafe = false;
        t->maskBitCount = 32;
    }
    else if (!strcasecmp(isa, "generic-4")) {
        t->isa = Target::GENERIC;
        t->nativeVectorWidth = 4;
        t->vectorWidth = 4;
        t->maskingIsFree = true;
        t->allOffMaskIsSafe = true;
        t->maskBitCount = 1;
    }
    else if (!strcasecmp(isa, "generic-8")) {
        t->isa = Target::GENERIC;
        t->nativeVectorWidth = 8;
        t->vectorWidth = 8;
        t->maskingIsFree = true;
        t->allOffMaskIsSafe = true;
        t->maskBitCount = 1;
    }
    else if (!strcasecmp(isa, "generic-16")) {
        t->isa = Target::GENERIC;
        t->nativeVectorWidth = 16;
        t->vectorWidth = 16;
        t->maskingIsFree = true;
        t->allOffMaskIsSafe = true;
        t->maskBitCount = 1;
    }
    else if (!strcasecmp(isa, "generic-1")) {
        t->isa = Target::GENERIC;
        t->nativeVectorWidth = 1;
        t->vectorWidth = 1;
        t->maskingIsFree = false;
        t->allOffMaskIsSafe = false;
        t->maskBitCount = 32;
    }
#if defined(LLVM_3_0) || defined(LLVM_3_0svn) || defined(LLVM_3_1svn)
    else if (!strcasecmp(isa, "avx")) {
        t->isa = Target::AVX;
        t->nativeVectorWidth = 8;
        t->vectorWidth = 8;
        t->attributes = "+avx,+popcnt,+cmov";
        t->maskingIsFree = false;
        t->allOffMaskIsSafe = false;
        t->maskBitCount = 32;
    }
    else if (!strcasecmp(isa, "avx-x2")) {
        t->isa = Target::AVX;
        t->nativeVectorWidth = 8;
        t->vectorWidth = 16;
        t->attributes = "+avx,+popcnt,+cmov";
        t->maskingIsFree = false;
        t->allOffMaskIsSafe = false;
        t->maskBitCount = 32;
    }
#endif // LLVM 3.0+
#if defined(LLVM_3_1svn)
    else if (!strcasecmp(isa, "avx2")) {
        t->isa = Target::AVX2;
        t->nativeVectorWidth = 8;
        t->vectorWidth = 8;
        t->attributes = "+avx2,+popcnt,+cmov,+f16c";
        t->maskingIsFree = false;
        t->allOffMaskIsSafe = false;
        t->maskBitCount = 32;
    }
    else if (!strcasecmp(isa, "avx2-x2")) {
        t->isa = Target::AVX2;
        t->nativeVectorWidth = 16;
        t->vectorWidth = 16;
        t->attributes = "+avx2,+popcnt,+cmov,+f16c";
        t->maskingIsFree = false;
        t->allOffMaskIsSafe = false;
        t->maskBitCount = 32;
    }
#endif // LLVM 3.1
    else {
        fprintf(stderr, "Target ISA \"%s\" is unknown.  Choices are: %s\n", 
                isa, SupportedTargetISAs());
        error = true;
    }

    if (!error) {
        llvm::TargetMachine *targetMachine = t->GetTargetMachine();
        const llvm::TargetData *targetData = targetMachine->getTargetData();
        t->is32Bit = (targetData->getPointerSize() == 4);
    }

    return !error;
}


const char *
Target::SupportedTargetCPUs() {
    return "atom, barcelona, core2, corei7, "
#if defined(LLVM_3_0) || defined(LLVM_3_0svn) || defined(LLVM_3_1svn)
        "corei7-avx, "
#endif
        "istanbul, nocona, penryn, "
#ifdef LLVM_2_9
        "sandybridge, "
#endif
        "westmere";
}


const char *
Target::SupportedTargetArchs() {
    return "x86, x86-64";
}


const char *
Target::SupportedTargetISAs() {
    return "sse2, sse2-x2, sse4, sse4-x2"
#ifndef LLVM_2_9
        ", avx, avx-x2"
#endif // !LLVM_2_9
#ifdef LLVM_3_1svn
        ", avx2, avx2-x2"
#endif // LLVM_3_1svn
        ", generic-4, generic-8, generic-16, generic-1";
}


std::string
Target::GetTripleString() const {
    llvm::Triple triple;
    // Start with the host triple as the default
#if defined(LLVM_3_1) || defined(LLVM_3_1svn)
    triple.setTriple(llvm::sys::getDefaultTargetTriple());
#else
    triple.setTriple(llvm::sys::getHostTriple());
#endif

    // And override the arch in the host triple based on what the user
    // specified.  Here we need to deal with the fact that LLVM uses one
    // naming convention for targets TargetRegistry, but wants some
    // slightly different ones for the triple.  TODO: is there a way to
    // have it do this remapping, which would presumably be a bit less
    // error prone?
    if (arch == "x86")
        triple.setArchName("i386");
    else if (arch == "x86-64")
        triple.setArchName("x86_64");
    else
        triple.setArchName(arch);

    return triple.str();
}


llvm::TargetMachine *
Target::GetTargetMachine() const {
    std::string triple = GetTripleString();

    llvm::Reloc::Model relocModel = generatePIC ? llvm::Reloc::PIC_ : 
                                                  llvm::Reloc::Default;
#if defined(LLVM_3_1svn)
    std::string featuresString = attributes;
    llvm::TargetOptions options;
    if (g->opt.fastMath == true)
        options.UnsafeFPMath = 1;
    llvm::TargetMachine *targetMachine = 
        target->createTargetMachine(triple, cpu, featuresString, options,
                                    relocModel);
#elif defined(LLVM_3_0)
    std::string featuresString = attributes;
    llvm::TargetMachine *targetMachine = 
        target->createTargetMachine(triple, cpu, featuresString, relocModel);
#else // LLVM 2.9
#ifdef ISPC_IS_APPLE
    relocModel = llvm::Reloc::PIC_;
#endif // ISPC_IS_APPLE
    std::string featuresString = cpu + std::string(",") + attributes;
    llvm::TargetMachine *targetMachine = 
        target->createTargetMachine(triple, featuresString);
#ifndef ISPC_IS_WINDOWS
    targetMachine->setRelocationModel(relocModel);
#endif // !ISPC_IS_WINDOWS
#endif // LLVM_2_9

    Assert(targetMachine != NULL);

    targetMachine->setAsmVerbosityDefault(true);
    return targetMachine;
}


const char *
Target::GetISAString() const {
    switch (isa) {
    case Target::SSE2:
        return "sse2";
    case Target::SSE4:
        return "sse4";
    case Target::AVX:
        return "avx";
    case Target::AVX2:
        return "avx2";
    case Target::GENERIC:
        return "generic";
    default:
        FATAL("Unhandled target in GetISAString()");
    }
    return "";
}


static bool
lGenericTypeLayoutIndeterminate(LLVM_TYPE_CONST llvm::Type *type) {
    if (type->isPrimitiveType() || type->isIntegerTy())
        return false;

    if (type == LLVMTypes::BoolVectorType ||
        type == LLVMTypes::MaskType ||
        type == LLVMTypes::Int1VectorType)
        return true;

    LLVM_TYPE_CONST llvm::ArrayType *at = 
        llvm::dyn_cast<LLVM_TYPE_CONST llvm::ArrayType>(type);
    if (at != NULL)
        return lGenericTypeLayoutIndeterminate(at->getElementType());

    LLVM_TYPE_CONST llvm::PointerType *pt = 
        llvm::dyn_cast<LLVM_TYPE_CONST llvm::PointerType>(type);
    if (pt != NULL)
        return false;

    LLVM_TYPE_CONST llvm::StructType *st =
        llvm::dyn_cast<LLVM_TYPE_CONST llvm::StructType>(type);
    if (st != NULL) {
        for (int i = 0; i < (int)st->getNumElements(); ++i)
            if (lGenericTypeLayoutIndeterminate(st->getElementType(i)))
                return true;
        return false;
    }

    Assert(llvm::isa<LLVM_TYPE_CONST llvm::VectorType>(type));
    return true;
}


llvm::Value *
Target::SizeOf(LLVM_TYPE_CONST llvm::Type *type, 
               llvm::BasicBlock *insertAtEnd) {
    if (isa == Target::GENERIC &&
        lGenericTypeLayoutIndeterminate(type)) {
        llvm::Value *index[1] = { LLVMInt32(1) };
        LLVM_TYPE_CONST llvm::PointerType *ptrType = llvm::PointerType::get(type, 0);
        llvm::Value *voidPtr = llvm::ConstantPointerNull::get(ptrType);
#if defined(LLVM_3_0) || defined(LLVM_3_0svn) || defined(LLVM_3_1svn)
        llvm::ArrayRef<llvm::Value *> arrayRef(&index[0], &index[1]);
        llvm::Instruction *gep = 
            llvm::GetElementPtrInst::Create(voidPtr, arrayRef, "sizeof_gep",
                                            insertAtEnd);
#else
        llvm::Instruction *gep =
            llvm::GetElementPtrInst::Create(voidPtr, &index[0], &index[1],
                                            "sizeof_gep", insertAtEnd);
#endif
        if (is32Bit || g->opt.force32BitAddressing)
            return new llvm::PtrToIntInst(gep, LLVMTypes::Int32Type, 
                                          "sizeof_int", insertAtEnd);
        else
            return new llvm::PtrToIntInst(gep, LLVMTypes::Int64Type, 
                                          "sizeof_int", insertAtEnd);
    }

    const llvm::TargetData *td = GetTargetMachine()->getTargetData();
    Assert(td != NULL);
    uint64_t byteSize = td->getTypeSizeInBits(type) / 8;
    if (is32Bit || g->opt.force32BitAddressing)
        return LLVMInt32((int32_t)byteSize);
    else
        return LLVMInt64(byteSize);
}


llvm::Value *
Target::StructOffset(LLVM_TYPE_CONST llvm::Type *type, int element,
                     llvm::BasicBlock *insertAtEnd) {
    if (isa == Target::GENERIC && 
        lGenericTypeLayoutIndeterminate(type) == true) {
        llvm::Value *indices[2] = { LLVMInt32(0), LLVMInt32(element) };
        LLVM_TYPE_CONST llvm::PointerType *ptrType = llvm::PointerType::get(type, 0);
        llvm::Value *voidPtr = llvm::ConstantPointerNull::get(ptrType);
#if defined(LLVM_3_0) || defined(LLVM_3_0svn) || defined(LLVM_3_1svn)
        llvm::ArrayRef<llvm::Value *> arrayRef(&indices[0], &indices[2]);
        llvm::Instruction *gep = 
            llvm::GetElementPtrInst::Create(voidPtr, arrayRef, "offset_gep",
                                            insertAtEnd);
#else
        llvm::Instruction *gep =
            llvm::GetElementPtrInst::Create(voidPtr, &indices[0], &indices[2],
                                            "offset_gep", insertAtEnd);
#endif
        if (is32Bit || g->opt.force32BitAddressing)
            return new llvm::PtrToIntInst(gep, LLVMTypes::Int32Type, 
                                          "offset_int", insertAtEnd);
        else
            return new llvm::PtrToIntInst(gep, LLVMTypes::Int64Type, 
                                          "offset_int", insertAtEnd);
    }

    const llvm::TargetData *td = GetTargetMachine()->getTargetData();
    Assert(td != NULL);
    LLVM_TYPE_CONST llvm::StructType *structType = 
        llvm::dyn_cast<LLVM_TYPE_CONST llvm::StructType>(type);
    Assert(structType != NULL);
    const llvm::StructLayout *sl = td->getStructLayout(structType);
    Assert(sl != NULL);

    uint64_t offset = sl->getElementOffset(element);
    if (is32Bit || g->opt.force32BitAddressing)
        return LLVMInt32((int32_t)offset);
    else
        return LLVMInt64(offset);
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
    disableMaskAllOnOptimizations = false;
    disableHandlePseudoMemoryOps = false;
    disableBlendedMaskedStores = false;
    disableCoherentControlFlow = false;
    disableUniformControlFlow = false;
    disableGatherScatterOptimizations = false;
    disableMaskedStoreToStore = false;
    disableGatherScatterFlattening = false;
    disableUniformMemoryOptimizations = false;
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
    return m->diBuilder->createFile(filename, directory);
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
