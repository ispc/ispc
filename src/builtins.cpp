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
#include "target_capabilities.h"
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
            Assert(func->getType() == LLVMTypes::VoidPointerType);
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
    // Set internal linkage for configuration globals using centralized list from target_capabilities.h
    // This corresponds to ISPC_CONFIG_CONSTANTS in stdlib/include/core.isph
    for (const char *name : g_configGlobalNames) {
        lSetInternalLinkageGlobal(module, name);
    }

    // Set internal linkage for capability-related globals using centralized metadata
    // This corresponds to ISPC_CAPABILITY_CONSTANTS in stdlib/include/core.isph
    for (const auto &cm : g_capabilityMetadata) {
        if (cm.globalVarName != nullptr) {
            lSetInternalLinkageGlobal(module, cm.globalVarName);
        }
    }
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
        // Remove debug compile unit metadata if debug info is not enabled in ISPC compilation.
        // This prevents llvm.dbg.cu metadata from being added when linking builtins-c bitcode.
        if (!g->generateDebuggingSymbols) {
            llvm::NamedMDNode *cuMD = bcModule->getNamedMetadata("llvm.dbg.cu");
            if (cuMD) {
                cuMD->eraseFromParent();
            }
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
#if ISPC_LLVM_VERSION >= ISPC_LLVM_21_0
        bcModule->setTargetTriple(mTriple);
#else
        bcModule->setTargetTriple(mTriple.str());
#endif
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

void lCreateLLVMUsed(llvm::Module &M, std::vector<llvm::Constant *> &ConstPtrs, const std::string &VarName) {
    llvm::LLVMContext &Context = M.getContext();

    // Create the array of i8* that llvm.compiler.used or llvm.used will hold
    llvm::Type *type = llvm::PointerType::getUnqual(Context);
    llvm::ArrayType *ATy = llvm::ArrayType::get(type, ConstPtrs.size());
    llvm::Constant *ArrayInit = llvm::ConstantArray::get(ATy, ConstPtrs);

    // If we already have llvm.compiler.used or llvm.used then something is wrong.
    llvm::GlobalVariable *GV = M.getGlobalVariable(VarName);
    Assert(GV == nullptr);

    // Create llvm.compiler.used or llvm.used and initialize it with the functions
    llvm::GlobalVariable *llvmUsed = new llvm::GlobalVariable(M, ArrayInit->getType(), false,
                                                              llvm::GlobalValue::AppendingLinkage, ArrayInit, VarName);
    llvmUsed->setSection("llvm.metadata");
}

void lKeepFloat16Conversions(llvm::Module &M) {
    std::vector<llvm::Constant *> ConstPtrs;

    // Add float16 conversion functions if they exist
    for (auto const &[name, val] : builtin::float16ConversionFuncs) {
        if (llvm::Constant *C = lFuncAsConstInt8Ptr(M, name.c_str())) {
            ConstPtrs.push_back(C);
        }
    }

    if (ConstPtrs.empty()) {
        return;
    }

    // Create llvm.used and initialize it with the float16 conversion
    // functions. We use llvm.used instead of llvm.compiler.used to distinguish
    // these functions from persistent functions. Calls to float16 conversion
    // functions can be generated for some targets during ISEL. We never delete
    // llvm.used because it should be done in the backend in machine
    // optimization passes but we don't control them currently.
    lCreateLLVMUsed(M, ConstPtrs, "llvm.used");
}

// Update llvm.compiler.used with the new list of preserved functions.
void lUpdateLLVMUsed(llvm::Module *M, llvm::GlobalVariable *llvmUsed, std::vector<llvm::Constant *> &newElements) {
    llvmUsed->eraseFromParent();
    lCreateLLVMUsed(*M, newElements, "llvm.compiler.used");
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
    llvm::StringSet<> dispatchFunctions = {builtin::__get_system_best_isa, builtin::__terminate_now};
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

    // We store references to persistent functions in llvm.compiler.used. It is
    // removed later by RemovePersistentFuncsPass to let unused persistent
    // functions to be removed. We do so because calls to persistent functions
    // can be generated by ISPC specific passes ReplacePseudoMemoryOps,
    // ReplaceMaskedMemOps or ImproveMemoryOps later in the middle-end
    // optimization pipeline.
    lCreateLLVMUsed(M, ConstPtrs, "llvm.compiler.used");
}

bool lStartsWithLLVM(llvm::StringRef name) { return name.starts_with("llvm."); }

// Mapping from each target to its parent target
// clang-format off
std::unordered_map<ISPCTarget, ISPCTarget> targetParentMap = {
    {ISPCTarget::neon_i8x16, ISPCTarget::generic_i8x16},
    {ISPCTarget::neon_i8x32, ISPCTarget::generic_i8x32},
    {ISPCTarget::neon_i16x8, ISPCTarget::generic_i16x8},
    {ISPCTarget::neon_i16x16, ISPCTarget::generic_i16x16},
    {ISPCTarget::neon_i32x4, ISPCTarget::generic_i32x4},
    {ISPCTarget::neon_i32x8, ISPCTarget::generic_i32x8},

    {ISPCTarget::rvv_x4, ISPCTarget::generic_i1x4},

#if ISPC_LLVM_VERSION >= ISPC_LLVM_20_0
    {ISPCTarget::avx10_2dmr_x4, ISPCTarget::avx512spr_x4},
    {ISPCTarget::avx10_2dmr_x8, ISPCTarget::avx512spr_x8},
    {ISPCTarget::avx10_2dmr_x16, ISPCTarget::avx512spr_x16},
    {ISPCTarget::avx10_2dmr_x32, ISPCTarget::avx512spr_x32},
    {ISPCTarget::avx10_2dmr_x64, ISPCTarget::avx512spr_x64},
#endif

    {ISPCTarget::avx512spr_x4, ISPCTarget::avx512icl_x4},
    {ISPCTarget::avx512icl_x4, ISPCTarget::avx512skx_x4},
    {ISPCTarget::avx512skx_x4, ISPCTarget::generic_i1x4},

    {ISPCTarget::avx512spr_x8, ISPCTarget::avx512icl_x8},
    {ISPCTarget::avx512icl_x8, ISPCTarget::avx512skx_x8},
    {ISPCTarget::avx512skx_x8, ISPCTarget::generic_i1x8},

    {ISPCTarget::avx512spr_x16, ISPCTarget::avx512icl_x16},
    {ISPCTarget::avx512icl_x16, ISPCTarget::avx512skx_x16},
    {ISPCTarget::avx512skx_x16, ISPCTarget::generic_i1x16},
    
    {ISPCTarget::avx512icl_x16_nozmm, ISPCTarget::avx512skx_x16_nozmm},
    {ISPCTarget::avx512skx_x16_nozmm, ISPCTarget::generic_i1x16},

    {ISPCTarget::avx512spr_x32, ISPCTarget::avx512icl_x32},
    {ISPCTarget::avx512icl_x32, ISPCTarget::avx512skx_x32},
    {ISPCTarget::avx512skx_x32, ISPCTarget::generic_i1x32},

    {ISPCTarget::avx512spr_x64, ISPCTarget::avx512icl_x64},
    {ISPCTarget::avx512icl_x64, ISPCTarget::avx512skx_x64},
    {ISPCTarget::avx512skx_x64, ISPCTarget::generic_i1x64},

    // Several notes here.
    // Even though sse41* targets are aliases for sse4 and avx1_i32x4 is alias for sse4_i32x4,
    // they should be present in this map.
    // Also a reminder that "sse4" in ISPC targets means "sse42".
    {ISPCTarget::avx2vnni_i32x4, ISPCTarget::avx2_i32x4},
    {ISPCTarget::avx2_i32x4, ISPCTarget::avx1_i32x4},
    {ISPCTarget::avx1_i32x4, ISPCTarget::sse4_i32x4},
    {ISPCTarget::sse4_i32x4, ISPCTarget::sse41_i32x4},
    {ISPCTarget::sse41_i32x4, ISPCTarget::sse2_i32x4},
    {ISPCTarget::sse2_i32x4, ISPCTarget::generic_i32x4},

    {ISPCTarget::avx2vnni_i32x8, ISPCTarget::avx2_i32x8},
    {ISPCTarget::avx2_i32x8, ISPCTarget::avx1_i32x8},
    {ISPCTarget::avx1_i32x8, ISPCTarget::sse4_i32x8},
    {ISPCTarget::sse4_i32x8, ISPCTarget::sse41_i32x8},
    {ISPCTarget::sse41_i32x8, ISPCTarget::sse2_i32x8},
    {ISPCTarget::sse2_i32x8, ISPCTarget::generic_i32x8},

    {ISPCTarget::avx2vnni_i32x16, ISPCTarget::avx2_i32x16},
    {ISPCTarget::avx2_i32x16, ISPCTarget::avx1_i32x16},
    {ISPCTarget::avx1_i32x16, ISPCTarget::generic_i32x16},

    {ISPCTarget::avx2_i64x4, ISPCTarget::avx1_i64x4},
    {ISPCTarget::avx1_i64x4, ISPCTarget::generic_i64x4},

    {ISPCTarget::avx2_i16x16, ISPCTarget::generic_i16x16},

    {ISPCTarget::avx2_i8x32, ISPCTarget::generic_i8x32},

    {ISPCTarget::sse4_i8x16, ISPCTarget::sse41_i8x16},
    {ISPCTarget::sse41_i8x16, ISPCTarget::generic_i8x16},

    {ISPCTarget::sse4_i16x8, ISPCTarget::sse41_i16x8},
    {ISPCTarget::sse41_i16x8, ISPCTarget::generic_i16x8},

    {ISPCTarget::wasm_i32x4, ISPCTarget::generic_i32x4}

    /*{ISPCTarget::xelp_x8, ISPCTarget::generic_i1x8},
    {ISPCTarget::xelp_x16, ISPCTarget::generic_i1x16},
    {ISPCTarget::xehpg_x8, ISPCTarget::generic_i1x8},
    {ISPCTarget::xehpg_x16, ISPCTarget::generic_i1x16},
    {ISPCTarget::xehpc_x16, ISPCTarget::generic_i1x16},
    {ISPCTarget::xehpc_x32, ISPCTarget::generic_i1x32},
    {ISPCTarget::xelpg_x8, ISPCTarget::generic_i1x8},
    {ISPCTarget::xelpg_x16, ISPCTarget::generic_i1x16},
    {ISPCTarget::xe2hpg_x16, ISPCTarget::generic_i1x16},
    {ISPCTarget::xe2hpg_x32, ISPCTarget::generic_i1x32},
    {ISPCTarget::xe2lpg_x16, ISPCTarget::generic_i1x16},
    {ISPCTarget::xe2lpg_x32, ISPCTarget::generic_i1x32}*/
};
// clang-format on

// Traverse the target hierarchy to find the parent target.
ISPCTarget GetParentTarget(ISPCTarget target) {
    auto it = targetParentMap.find(target);
    if (it != targetParentMap.end()) {
        return it->second;
    }
    return ISPCTarget::none;
}

// Check if the module has used but unresolved symbols from the given set of symbols.
bool lHasUnresolvedSymbols(llvm::Module *module, const llvm::StringSet<> &symbols, std::string &unresolvedSymbol) {
    for (llvm::Function &F : module->functions()) {
        auto name = F.getName().str();
        if (F.isDeclaration() && !F.use_empty() && symbols.count(name)) {
            unresolvedSymbol = name;
            return true;
        }
    }
    return false;
}

void lFillTargetBuiltins(llvm::Module *bcModule, llvm::StringSet<> &targetBuiltins) {
    for (llvm::Function &F : bcModule->functions()) {
        auto name = F.getName();
        if (!lStartsWithLLVM(name)) {
            targetBuiltins.insert(name);
        }
    }
}

llvm::Module *lGetTargetBCModule(ISPCTarget target) {
    const BitcodeLib *lib = g->target_registry->getISPCTargetLib(target, g->target_os, g->target->getArch());
    if (!lib) {
        Error(SourcePos(), "Failed to get target bitcode library for target %s.", ISPCTargetToString(target).c_str());
        return nullptr;
    }
    return lib->getLLVMModule();
}

void lCollectAllDefinedFunctions(ISPCTarget target, llvm::StringSet<> &definedFunctions) {
    ISPCTarget rootTarget = target;
    // Traverse the target hierarchy to find the root target.
    // If the target is not found in the map then rootTarget is the target itself.
    for (ISPCTarget t = target; t != ISPCTarget::none; t = GetParentTarget(t)) {
        rootTarget = t;
    }
    llvm::Module *m = lGetTargetBCModule(rootTarget);
    for (llvm::Function &F : m->functions()) {
        auto name = F.getName();
        if (!lStartsWithLLVM(name) && !F.isDeclaration()) {
            definedFunctions.insert(name);
        }
    }
}

void lLinkBuitinsForTarget(llvm::Module *m, ISPCTarget target, llvm::StringSet<> &functions) {
    llvm::Module *bc = lGetTargetBCModule(target);
    lFillTargetBuiltins(bc, functions);

    bc->setDataLayout(g->target->getDataLayout()->getStringRepresentation());
    bc->setTargetTriple(m->getTargetTriple());

    lAddBitcodeToModule(bc, m);
}

void lLinkTargetBuiltins(llvm::Module *module, int &debug_num) {
    llvm::StringSet<> targetBuiltins;
    ISPCTarget target = g->target->getISPCTarget();

    // Collect all defined functions in the root target module. We need this
    // list to check for unresolved symbols. To distuinguish between unresolved
    // user symbols and unresolved builtin symbols, we traverse the current
    // target to the most generic one, then we save all defined there functions
    // as symbols that should be resolved.
    // If the target is not found in the targetParentMap then symbolsToResolve
    // is just the list of defined builtins functions, i.e., all of them
    // resolved in the target module, so lHasUnresolvedSymbols() will return
    // false later.
    llvm::StringSet<> symbolsToResolve;
    lCollectAllDefinedFunctions(target, symbolsToResolve);

    // Add the target's custom implementations of the various needed builtin
    // functions (e.g. __masked_store_32(), etc) from builtins module for the
    // current target we are compiling for.
    lLinkBuitinsForTarget(module, target, targetBuiltins);

    // Check for unresolved symbols and hierarchically link with the parent
    // (more generic) target's bitcode if needed.
    std::string unresolvedSymbol;
    while (lHasUnresolvedSymbols(module, symbolsToResolve, unresolvedSymbol)) {
        target = GetParentTarget(target);
        if (target == ISPCTarget::none) {
            Error(SourcePos(), "Unresolved symbol %s in target bitcode.", unresolvedSymbol.c_str());
            break;
        }

        lLinkBuitinsForTarget(module, target, targetBuiltins);
        debugDumpModule(module, "Parent", debug_num++);
    }

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
        debugDumpModule(module, "LinkStdlib", debug_num++);
    }

    lLinkCommonBuiltins(module);
    debugDumpModule(module, "LinkCommonBuiltins", debug_num++);

    lLinkTargetBuiltins(module, debug_num);
    // generic target implementation itself uses some of the pseudo functions
    // and their dependencies that we need to preserve.
    // Here, more code stay in the module in comparison with when this was
    // placed under LinkStdlib. Unfortunately, it is not clear how to reduce
    // having builtin functions implementations in ISPC.
    lAddPersistentToLLVMUsed(*module);

    // Add float16 conversion functions to llvm.used if the command line option is enabled
    if (g->includeFloat16Conversions) {
        lKeepFloat16Conversions(*module);
    }

    lRemoveUnused(module);
    lRemoveUnusedPersistentFunctions(module);
    debugDumpModule(module, "LinkTargetBuiltins", debug_num++);

    lSetInternalLinkageGlobals(module);
    lCheckModuleIntrinsics(module);
}
