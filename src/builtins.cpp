/*
  Copyright (c) 2010-2025, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file builtins.cpp
    @brief Definitions of functions related to setting up the standard library
           and other builtins.
*/

#include "builtins.h"
#include "ctx.h"
#include "expr.h"
#include "llvmutil.h"
#include "module.h"
#include "sym.h"
#include "type.h"
#include "util.h"

#include <math.h>
#include <stdlib.h>

#include <unordered_set>

#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/Triple.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/GlobalDCE.h>

#ifdef ISPC_XE_ENABLED
#include <llvm/GenXIntrinsics/GenXIntrinsics.h>
#endif

using namespace ispc;
using namespace ispc::builtin;

/** In many of the builtins-*.ll files, we have declarations of various LLVM
    intrinsics that are then used in the implementation of various target-
    specific functions.  This function loops over all of the intrinsic
    declarations and makes sure that the signature we have in our .ll file
    matches the signature of the actual intrinsic.
*/
static void lCheckModuleIntrinsics(llvm::Module *module) {
    llvm::Module::iterator iter;
    for (iter = module->begin(); iter != module->end(); ++iter) {
        llvm::Function *func = &*iter;
        if (!func->isIntrinsic()) {
            continue;
        }

        const std::string funcName = func->getName().str();
        const std::string llvm_x86 = "llvm.x86.";
        // Work around http://llvm.org/bugs/show_bug.cgi?id=10438; only
        // check the llvm.x86.* intrinsics for now...
        if (funcName.length() >= llvm_x86.length() && !funcName.compare(0, llvm_x86.length(), llvm_x86)) {
            llvm::Intrinsic::ID id = (llvm::Intrinsic::ID)func->getIntrinsicID();
            if (id == 0) {
                std::string error_message = "Intrinsic is not found: ";
                error_message += funcName;
                FATAL(error_message.c_str());
            }
            llvm::Type *intrinsicType = llvm::Intrinsic::getType(*g->ctx, id);
            intrinsicType = llvm::PointerType::get(intrinsicType, 0);
            Assert(func->getType() == intrinsicType);
        }
    }
}

static void lUpdateIntrinsicsAttributes(llvm::Module *module) {
#ifdef ISPC_XE_ENABLED
    for (auto F = module->begin(), E = module->end(); F != E; ++F) {
        llvm::Function *Fn = &*F;
        // WA for isGenXIntrinsic(Fn) and getGenXIntrinsicID(Fn)
        // There are crashes if intrinsics is not supported on some platforms
        if (Fn && Fn->getName().contains("prefetch")) {
            continue;
        }
        if (Fn && llvm::GenXIntrinsic::isGenXIntrinsic(Fn)) {
            Fn->setAttributes(
                llvm::GenXIntrinsic::getAttributes(Fn->getContext(), llvm::GenXIntrinsic::getGenXIntrinsicID(Fn)));

            // ReadNone, ReadOnly and WriteOnly are not supported for intrinsics anymore:
            FixFunctionAttribute(*Fn, llvm::Attribute::ReadNone, llvm::MemoryEffects::none());
            FixFunctionAttribute(*Fn, llvm::Attribute::ReadOnly, llvm::MemoryEffects::readOnly());
            FixFunctionAttribute(*Fn, llvm::Attribute::WriteOnly, llvm::MemoryEffects::writeOnly());
        }
    }
#endif
}

static void lSetAsInternal(llvm::Module *module, llvm::StringSet<> &functions) {
    for (llvm::Function &F : module->functions()) {
        if (!F.isDeclaration() && functions.find(F.getName()) != functions.end()) {
            F.setLinkage(llvm::GlobalValue::InternalLinkage);
        }
    }
}

void lSetInternalLinkageGlobal(llvm::Module *module, const char *name) {
    llvm::GlobalValue *GV = module->getNamedGlobal(name);
    if (GV) {
        GV->setLinkage(llvm::GlobalValue::InternalLinkage);
    }
}

void lSetInternalLinkageGlobals(llvm::Module *module) {
    lSetInternalLinkageGlobal(module, "__fast_masked_vload");
    lSetInternalLinkageGlobal(module, "__math_lib");
    lSetInternalLinkageGlobal(module, "__memory_alignment");
    lSetInternalLinkageGlobal(module, "__have_dot_product_arm");
    lSetInternalLinkageGlobal(module, "__have_i8_matmul_arm");
}

void lAddBitcodeToModule(llvm::Module *bcModule, llvm::Module *module) {
    if (!bcModule) {
        Error(SourcePos(), "Error library module is nullptr");
    } else {
        if (g->target->isXeTarget()) {
            // Maybe we will use it for other targets in future,
            // but now it is needed only by Xe. We need
            // to update attributes because Xe intrinsics are
            // separated from the others and it is not done by default
            lUpdateIntrinsicsAttributes(bcModule);
        }

        for (llvm::Function &f : *bcModule) {
            if (f.isDeclaration()) {
                // Declarations with uses will be moved by Linker.
                if (f.getNumUses() > 0) {
                    continue;
                }
                // Declarations with 0 uses are moved by hands.
                module->getOrInsertFunction(f.getName(), f.getFunctionType(), f.getAttributes());
            }
        }

        // Remove clang ID metadata from the bitcode module, as we don't need it.
        llvm::NamedMDNode *identMD = bcModule->getNamedMetadata("llvm.ident");
        if (identMD) {
            identMD->eraseFromParent();
        }

        std::unique_ptr<llvm::Module> M(bcModule);
        if (llvm::Linker::linkModules(*module, std::move(M), llvm::Linker::Flags::LinkOnlyNeeded)) {
            Error(SourcePos(), "Error linking stdlib bitcode.");
        }
    }
}

void lAddDeclarationsToModule(llvm::Module *bcModule, llvm::Module *module) {
    if (!bcModule) {
        Error(SourcePos(), "Error library module is nullptr");
    } else {
        // FIXME: this feels like a bad idea, but the issue is that when we
        // set the llvm::Module's target triple in the ispc Module::Module
        // constructor, we start by calling llvm::sys::getHostTriple() (and
        // then change the arch if needed).  Somehow that ends up giving us
        // strings like 'x86_64-apple-darwin11.0.0', while the stuff we
        // compile to bitcode with clang has module triples like
        // 'i386-apple-macosx10.7.0'.  And then LLVM issues a warning about
        // linking together modules with incompatible target triples..
        llvm::Triple mTriple(m->module->getTargetTriple());
        llvm::Triple bcTriple(bcModule->getTargetTriple());
        Debug(SourcePos(), "module triple: %s\nbitcode triple: %s\n", mTriple.str().c_str(), bcTriple.str().c_str());

        bcModule->setTargetTriple(mTriple.str());
        bcModule->setDataLayout(module->getDataLayout());

        if (g->target->isXeTarget()) {
            // Maybe we will use it for other targets in future,
            // but now it is needed only by Xe. We need
            // to update attributes because Xe intrinsics are
            // separated from the others and it is not done by default
            lUpdateIntrinsicsAttributes(bcModule);
        }

        for (llvm::Function &f : *bcModule) {
            // TODO! do we really try to define already included symbol?
            if (!module->getFunction(f.getName())) {
                module->getOrInsertFunction(f.getName(), f.getFunctionType(), f.getAttributes());
            }
        }
    }
}

llvm::Constant *lFuncAsConstInt8Ptr(llvm::Module &M, const char *name) {
    llvm::LLVMContext &Context = M.getContext();
    llvm::Function *F = M.getFunction(name);
    llvm::Type *type = llvm::PointerType::getUnqual(Context);
    if (F) {
        return llvm::ConstantExpr::getBitCast(F, type);
    }
    return nullptr;
}

void lRemoveUnused(llvm::Module *M) {
    llvm::FunctionAnalysisManager FAM;
    llvm::ModuleAnalysisManager MAM;
    llvm::ModulePassManager PM;
    llvm::PassBuilder PB;
    PB.registerModuleAnalyses(MAM);
    PM.addPass(llvm::GlobalDCEPass());
    PM.run(*M, MAM);
}

// Extract functions from llvm.compiler.used that are currently used in the module.
void lExtractUsedFunctions(llvm::GlobalVariable *llvmUsed, std::unordered_set<llvm::Function *> &usedFunctions) {
    llvm::ConstantArray *initList = llvm::cast<llvm::ConstantArray>(llvmUsed->getInitializer());
    for (unsigned i = 0; i < initList->getNumOperands(); i++) {
        auto *C = initList->getOperand(i);
        llvm::ConstantExpr *CE = llvm::dyn_cast<llvm::ConstantExpr>(C);
        // Bitcast as ConstExpr when opaque pointer is not used, otherwise C is just an opaque pointer.
        llvm::Value *val = CE ? CE->getOperand(0) : C;
        Assert(val);
        if (val->getNumUses() > 1) {
            Assert(llvm::isa<llvm::Function>(val));
            usedFunctions.insert(llvm::cast<llvm::Function>(val));
        }
    }
}

// Find persistent groups that are used in the module.
void lFindUsedPersistentGroups(llvm::Module *M, std::unordered_set<llvm::Function *> &usedFunctions,
                               std::unordered_set<const builtin::PersistentGroup *> &usedPersistentGroups) {
    for (auto const &[group, functions] : builtin::persistentGroups) {
        for (auto const &name : functions) {
            llvm::Function *F = M->getFunction(name);
            if (usedFunctions.find(F) != usedFunctions.end()) {
                usedPersistentGroups.insert(&group);
                break;
            }
        }
    }
}

// Collect functions that should be preserved in the module based on the used persistent groups.
void lCollectPreservedFunctions(llvm::Module *M, std::vector<llvm::Constant *> &newElements,
                                std::unordered_set<const builtin::PersistentGroup *> &usedPersistentGroups) {
    for (auto const &[group, functions] : builtin::persistentGroups) {
        if (usedPersistentGroups.find(&group) != usedPersistentGroups.end()) {
            for (auto const &name : functions) {
                if (llvm::Constant *C = lFuncAsConstInt8Ptr(*M, name)) {
                    newElements.push_back(C);
                }
            }
        }
    }
    // Extend the list of preserved functions with the functions from corresponding persistent groups.
    for (auto const &[name, val] : builtin::persistentFuncs) {
        if (llvm::Constant *C = lFuncAsConstInt8Ptr(*M, name.c_str())) {
            newElements.push_back(C);
        }
    }
}

void lCreateLLVMUsed(llvm::Module &M, std::vector<llvm::Constant *> &ConstPtrs) {
    llvm::LLVMContext &Context = M.getContext();

    // Create the array of i8* that llvm.used will hold
    llvm::Type *type = llvm::PointerType::getUnqual(Context);
    llvm::ArrayType *ATy = llvm::ArrayType::get(type, ConstPtrs.size());
    llvm::Constant *ArrayInit = llvm::ConstantArray::get(ATy, ConstPtrs);

    // Create llvm.used and initialize it with the functions
    llvm::GlobalVariable *llvmUsed = new llvm::GlobalVariable(
        M, ArrayInit->getType(), false, llvm::GlobalValue::AppendingLinkage, ArrayInit, "llvm.compiler.used");
    llvmUsed->setSection("llvm.metadata");
}

// Update llvm.compiler.used with the new list of preserved functions.
void lUpdateLLVMUsed(llvm::Module *M, llvm::GlobalVariable *llvmUsed, std::vector<llvm::Constant *> &newElements) {
    llvmUsed->eraseFromParent();
    lCreateLLVMUsed(*M, newElements);
}

void lRemoveUnusedPersistentFunctions(llvm::Module *M) {
    // The idea here is to preserve only the needed subset of persistent functions.
    // Inspect llvm.compiler.used and find all functions that are used in the
    // module and re-create it with only those functions (and corresponding
    // persistent groups).
    llvm::GlobalVariable *llvmUsed = M->getNamedGlobal("llvm.compiler.used");
    if (llvmUsed) {
        std::unordered_set<llvm::Function *> usedFunctions;
        lExtractUsedFunctions(llvmUsed, usedFunctions);

        std::unordered_set<const builtin::PersistentGroup *> usedPersistentGroups;
        lFindUsedPersistentGroups(M, usedFunctions, usedPersistentGroups);

        std::vector<llvm::Constant *> newElements;
        lCollectPreservedFunctions(M, newElements, usedPersistentGroups);

        lUpdateLLVMUsed(M, llvmUsed, newElements);
        lRemoveUnused(M);
    }
}

void ispc::debugDumpModule(llvm::Module *module, std::string name, int stage) {
    if (!(g->off_stages.find(stage) == g->off_stages.end() && g->debug_stages.find(stage) != g->debug_stages.end())) {
        return;
    }

    SourcePos noPos;
    name = std::string("pre_") + std::to_string(stage) + "_" + name + ".ll";
    if (g->dumpFile && !g->dumpFilePath.empty()) {
        std::error_code EC;
        llvm::SmallString<128> path(g->dumpFilePath);

        if (!llvm::sys::fs::exists(path)) {
            EC = llvm::sys::fs::create_directories(g->dumpFilePath);
            if (EC) {
                Error(noPos, "Error creating directory '%s': %s", g->dumpFilePath.c_str(), EC.message().c_str());
                return;
            }
        }

        if (g->isMultiTargetCompilation) {
            name += std::string("_") + g->target->GetISAString();
        }
        llvm::sys::path::append(path, name);
        llvm::raw_fd_ostream OS(path, EC);

        if (EC) {
            Error(noPos, "Error opening file '%s': %s", path.c_str(), EC.message().c_str());
            return;
        }

        module->print(OS, nullptr);
        OS.flush();
    } else {
        // dump to stdout
        module->print(llvm::outs(), nullptr);
    }
}

void ispc::LinkDispatcher(llvm::Module *module) {
    const BitcodeLib *dispatch = g->target_registry->getDispatchLib(g->target_os);
    Assert(dispatch);
    llvm::Module *dispatchBCModule = dispatch->getLLVMModule();
    lAddDeclarationsToModule(dispatchBCModule, module);
    lAddBitcodeToModule(dispatchBCModule, module);
    llvm::StringSet<> dispatchFunctions = {builtin::__set_system_isa};
    lSetAsInternal(module, dispatchFunctions);
}

void lLinkCommonBuiltins(llvm::Module *module) {
    const BitcodeLib *builtins = g->target_registry->getBuiltinsCLib(g->target_os, g->target->getArch());
    Assert(builtins);
    llvm::Module *builtinsBCModule = builtins->getLLVMModule();

    // Suppress LLVM warnings about incompatible target triples and data layout when linking.
    // It is OK until we have different platform agnostic bc libraries.
    builtinsBCModule->setDataLayout(g->target->getDataLayout()->getStringRepresentation());
    builtinsBCModule->setTargetTriple(module->getTargetTriple());

    // Unlike regular builtins and dispatch module, which don't care about mangling of external functions,
    // so they only differentiate Windows/Unix and 32/64 bit, builtins-c need to take care about mangling.
    // Hence, different version for all potentially supported OSes.
    lAddBitcodeToModule(builtinsBCModule, module);

    llvm::StringSet<> commonBuiltins = {builtin::__do_print, builtin::__num_cores};
    lSetAsInternal(module, commonBuiltins);
}

void lAddPersistentToLLVMUsed(llvm::Module &M) {
    // Bitcast all function pointer to i8*
    std::vector<llvm::Constant *> ConstPtrs;
    for (auto const &[group, functions] : persistentGroups) {
        bool isGroupUsed = false;
        for (auto const &name : functions) {
            llvm::Function *F = M.getFunction(name);
            if (F && F->getNumUses() > 0) {
                isGroupUsed = true;
                break;
            }
        }
        // TODO! comment that we don't need to preserve unused symbols chains
        if (isGroupUsed) {
            for (auto const &name : functions) {
                if (llvm::Constant *C = lFuncAsConstInt8Ptr(M, name)) {
                    ConstPtrs.push_back(C);
                }
            }
        }
    }

    for (auto const &[name, val] : persistentFuncs) {
        if (llvm::Constant *C = lFuncAsConstInt8Ptr(M, name.c_str())) {
            ConstPtrs.push_back(C);
        }
    }

    if (ConstPtrs.empty()) {
        return;
    }

    lCreateLLVMUsed(M, ConstPtrs);
}

bool lStartsWithLLVM(llvm::StringRef name) { return name.starts_with("llvm."); }

void lLinkTargetBuiltins(llvm::Module *module) {
    const BitcodeLib *target =
        g->target_registry->getISPCTargetLib(g->target->getISPCTarget(), g->target_os, g->target->getArch());
    Assert(target);
    llvm::Module *targetBCModule = target->getLLVMModule();

    llvm::StringSet<> targetBuiltins;
    for (llvm::Function &F : targetBCModule->functions()) {
        auto name = F.getName();
        if (!lStartsWithLLVM(name)) {
            targetBuiltins.insert(name);
        }
    }

    targetBCModule->setDataLayout(g->target->getDataLayout()->getStringRepresentation());
    targetBCModule->setTargetTriple(module->getTargetTriple());

    // Next, add the target's custom implementations of the various needed
    // builtin functions (e.g. __masked_store_32(), etc).
    lAddBitcodeToModule(targetBCModule, module);

    lSetAsInternal(module, targetBuiltins);
}

void lLinkStdlib(llvm::Module *module) {
    const BitcodeLib *stdlib =
        g->target_registry->getISPCStdLib(g->target->getISPCTarget(), g->target_os, g->target->getArch());
    Assert(stdlib);
    llvm::Module *stdlibBCModule = stdlib->getLLVMModule();

    if (g->isMultiTargetCompilation) {
        for (llvm::Function &F : stdlibBCModule->functions()) {
            if (!F.isDeclaration() && !lStartsWithLLVM(F.getName())) {
                F.setName(F.getName() + g->target->GetTargetSuffix());
            }
        }
    }

    llvm::StringSet<> stdlibFunctions;
    for (llvm::Function &F : stdlibBCModule->functions()) {
        // If compiling with --vectorcall then set calling convention of all
        // functions defined in stdlib and their calls to vectorcall to avoid misoptimization due to
        // calling conventions for the call site and the function
        // declaration/definition.
        if (g->calling_conv == CallingConv::x86_vectorcall) {
            if (!F.isDeclaration()) {
                F.setCallingConv(llvm::CallingConv::X86_VectorCall);
                // Set calling convention for all calls to this function
                for (auto &use : F.uses()) {
                    if (auto *callInst = llvm::dyn_cast<llvm::CallInst>(use.getUser())) {
                        callInst->setCallingConv(llvm::CallingConv::X86_VectorCall);
                    }
                }
            }
        }
        stdlibFunctions.insert(F.getName());
    }

    stdlibBCModule->setDataLayout(g->target->getDataLayout()->getStringRepresentation());
    stdlibBCModule->setTargetTriple(module->getTargetTriple());

    lAddBitcodeToModule(stdlibBCModule, module);
    lSetAsInternal(module, stdlibFunctions);
}

void ispc::LinkStandardLibraries(llvm::Module *module, int &debug_num) {
    if (g->includeStdlib) {
        lLinkStdlib(module);
        // Remove from module here only function definitions that unused (or
        // cannot be used) in module.
        lAddPersistentToLLVMUsed(*module);
        lRemoveUnused(module);
        lRemoveUnusedPersistentFunctions(module);
        debugDumpModule(module, "LinkStdlib", debug_num++);
    } else {
        lAddPersistentToLLVMUsed(*module);
    }

    lLinkCommonBuiltins(module);
    lRemoveUnused(module);
    debugDumpModule(module, "LinkCommonBuiltins", debug_num++);

    lLinkTargetBuiltins(module);
    lRemoveUnused(module);
    debugDumpModule(module, "LinkTargetBuiltins", debug_num++);

    lSetInternalLinkageGlobals(module);
    lCheckModuleIntrinsics(module);
}
