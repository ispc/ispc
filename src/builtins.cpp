/*
  Copyright (c) 2010-2024, Intel Corporation

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

#include <llvm/ADT/StringMap.h>
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
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/GlobalDCE.h>
#if ISPC_LLVM_VERSION >= ISPC_LLVM_17_0
#include <llvm/TargetParser/Triple.h>
#else
#include <llvm/ADT/Triple.h>
#endif

#ifdef ISPC_XE_ENABLED
#include <llvm/GenXIntrinsics/GenXIntrinsics.h>
#endif

using namespace ispc;
using namespace ispc::builtin;

extern int yyparse();
struct yy_buffer_state;
extern yy_buffer_state *yy_scan_string(const char *);
typedef struct yy_buffer_state *YY_BUFFER_STATE;
extern void yy_delete_buffer(YY_BUFFER_STATE);

/** Given an LLVM type, try to find the equivalent ispc type.  Note that
    this is an under-constrained problem due to LLVM's type representations
    carrying less information than ispc's.  (For example, LLVM doesn't
    distinguish between signed and unsigned integers in its types.)

    Because this function is only used for generating ispc declarations of
    functions defined in LLVM bitcode in the builtins-*.ll files, in practice
    we can get enough of what we need for the relevant cases to make things
    work, partially with the help of the intAsUnsigned parameter, which
    indicates whether LLVM integer types should be treated as being signed
    or unsigned.

 */
static const Type *lLLVMTypeToISPCType(const llvm::Type *t, bool intAsUnsigned) {
    if (t == LLVMTypes::VoidType)
        return AtomicType::Void;

    // uniform
    else if (t == LLVMTypes::BoolType)
        return AtomicType::UniformBool;
    else if (t == LLVMTypes::Int8Type)
        return intAsUnsigned ? AtomicType::UniformUInt8 : AtomicType::UniformInt8;
    else if (t == LLVMTypes::Int16Type)
        return intAsUnsigned ? AtomicType::UniformUInt16 : AtomicType::UniformInt16;
    else if (t == LLVMTypes::Int32Type)
        return intAsUnsigned ? AtomicType::UniformUInt32 : AtomicType::UniformInt32;
    else if (t == LLVMTypes::Float16Type)
        return AtomicType::UniformFloat16;
    else if (t == LLVMTypes::FloatType)
        return AtomicType::UniformFloat;
    else if (t == LLVMTypes::DoubleType)
        return AtomicType::UniformDouble;
    else if (t == LLVMTypes::Int64Type)
        return intAsUnsigned ? AtomicType::UniformUInt64 : AtomicType::UniformInt64;

    // varying
    if (t == LLVMTypes::Int8VectorType)
        return intAsUnsigned ? AtomicType::VaryingUInt8 : AtomicType::VaryingInt8;
    else if (t == LLVMTypes::Int16VectorType)
        return intAsUnsigned ? AtomicType::VaryingUInt16 : AtomicType::VaryingInt16;
    else if (t == LLVMTypes::Int32VectorType)
        return intAsUnsigned ? AtomicType::VaryingUInt32 : AtomicType::VaryingInt32;
    else if (t == LLVMTypes::Float16VectorType)
        return AtomicType::VaryingFloat16;
    else if (t == LLVMTypes::FloatVectorType)
        return AtomicType::VaryingFloat;
    else if (t == LLVMTypes::DoubleVectorType)
        return AtomicType::VaryingDouble;
    else if (t == LLVMTypes::Int64VectorType)
        return intAsUnsigned ? AtomicType::VaryingUInt64 : AtomicType::VaryingInt64;
    else if (t == LLVMTypes::MaskType)
        return AtomicType::VaryingBool;

    // pointers to uniform
    else if (t == LLVMTypes::Int8PointerType)
        return PointerType::GetUniform(intAsUnsigned ? AtomicType::UniformUInt8 : AtomicType::UniformInt8);
    else if (t == LLVMTypes::Int16PointerType)
        return PointerType::GetUniform(intAsUnsigned ? AtomicType::UniformUInt16 : AtomicType::UniformInt16);
    else if (t == LLVMTypes::Int32PointerType)
        return PointerType::GetUniform(intAsUnsigned ? AtomicType::UniformUInt32 : AtomicType::UniformInt32);
    else if (t == LLVMTypes::Int64PointerType)
        return PointerType::GetUniform(intAsUnsigned ? AtomicType::UniformUInt64 : AtomicType::UniformInt64);
    else if (t == LLVMTypes::Float16PointerType)
        return PointerType::GetUniform(AtomicType::UniformFloat16);
    else if (t == LLVMTypes::FloatPointerType)
        return PointerType::GetUniform(AtomicType::UniformFloat);
    else if (t == LLVMTypes::DoublePointerType)
        return PointerType::GetUniform(AtomicType::UniformDouble);

    // pointers to varying
    else if (t == LLVMTypes::Int8VectorPointerType)
        return PointerType::GetUniform(intAsUnsigned ? AtomicType::VaryingUInt8 : AtomicType::VaryingInt8);
    else if (t == LLVMTypes::Int16VectorPointerType)
        return PointerType::GetUniform(intAsUnsigned ? AtomicType::VaryingUInt16 : AtomicType::VaryingInt16);
    else if (t == LLVMTypes::Int32VectorPointerType)
        return PointerType::GetUniform(intAsUnsigned ? AtomicType::VaryingUInt32 : AtomicType::VaryingInt32);
    else if (t == LLVMTypes::Int64VectorPointerType)
        return PointerType::GetUniform(intAsUnsigned ? AtomicType::VaryingUInt64 : AtomicType::VaryingInt64);
    else if (t == LLVMTypes::Float16VectorPointerType)
        return PointerType::GetUniform(AtomicType::VaryingFloat16);
    else if (t == LLVMTypes::FloatVectorPointerType)
        return PointerType::GetUniform(AtomicType::VaryingFloat);
    else if (t == LLVMTypes::DoubleVectorPointerType)
        return PointerType::GetUniform(AtomicType::VaryingDouble);

    return nullptr;
}

static void lCreateSymbol(const std::string &name, const Type *returnType, llvm::SmallVector<const Type *, 8> &argTypes,
                          const llvm::FunctionType *ftype, llvm::Function *func, SymbolTable *symbolTable) {
    SourcePos noPos;
    noPos.name = "__stdlib";

    FunctionType *funcType = new FunctionType(returnType, argTypes, noPos);

    Debug(noPos, "Created builtin symbol \"%s\" [%s]\n", name.c_str(), funcType->GetString().c_str());

    Symbol *sym = new Symbol(name, noPos, funcType);
    sym->function = func;
    symbolTable->AddFunction(sym);
}

/** Given an LLVM function declaration, synthesize the equivalent ispc
    symbol for the function (if possible).  Returns true on success, false
    on failure.
 */
static bool lCreateISPCSymbol(llvm::Function *func, SymbolTable *symbolTable) {
    SourcePos noPos;
    noPos.name = "__stdlib";

    const llvm::FunctionType *ftype = func->getFunctionType();
    std::string name = std::string(func->getName());

    if (name.size() < 3 || name[0] != '_' || name[1] != '_')
        return false;

    Debug(SourcePos(), "Attempting to create ispc symbol for function \"%s\".", name.c_str());

    // An unfortunate hack: we want this builtin function to have the
    // signature "int __sext_varying_bool(bool)", but the ispc function
    // symbol creation code below assumes that any LLVM vector of i32s is a
    // varying int32.  Here, we need that to be interpreted as a varying
    // bool, so just have a one-off override for that one...
    if (g->target->getMaskBitCount() != 1 && name == builtin::__sext_varying_bool) {
        const Type *returnType = AtomicType::VaryingInt32;
        llvm::SmallVector<const Type *, 8> argTypes;
        argTypes.push_back(AtomicType::VaryingBool);

        FunctionType *funcType = new FunctionType(returnType, argTypes, noPos);

        Symbol *sym = new Symbol(name, noPos, funcType);
        sym->function = func;
        symbolTable->AddFunction(sym);
        return true;
    }

    // If the function has any parameters with integer types, we'll make
    // two Symbols for two overloaded versions of the function, one with
    // all of the integer types treated as signed integers and one with all
    // of them treated as unsigned.
    for (int i = 0; i < 2; ++i) {
        bool intAsUnsigned = (i == 1);

        const Type *returnType = lLLVMTypeToISPCType(ftype->getReturnType(), intAsUnsigned);
        if (returnType == nullptr) {
            Debug(SourcePos(),
                  "Return type not representable for "
                  "builtin %s.",
                  name.c_str());
            // return type not representable in ispc -> not callable from ispc
            return false;
        }

        // Iterate over the arguments and try to find their equivalent ispc
        // types.  Track if any of the arguments has an integer type.
        bool anyIntArgs = false;
        llvm::SmallVector<const Type *, 8> argTypes;
        for (unsigned int j = 0; j < ftype->getNumParams(); ++j) {
            const llvm::Type *llvmArgType = ftype->getParamType(j);
            const Type *type = lLLVMTypeToISPCType(llvmArgType, intAsUnsigned);
            if (type == nullptr) {
                Debug(SourcePos(),
                      "Type of parameter %d not "
                      "representable for builtin %s",
                      j, name.c_str());
                return false;
            }
            anyIntArgs |= (Type::Equal(type, lLLVMTypeToISPCType(llvmArgType, !intAsUnsigned)) == false);
            argTypes.push_back(type);
        }

        // Always create the symbol the first time through, in particular
        // so that we get symbols for things with no integer types!
        if (i == 0 || anyIntArgs == true)
            lCreateSymbol(name, returnType, argTypes, ftype, func, symbolTable);
    }

    return true;
}

Symbol *ispc::CreateISPCSymbolForLLVMIntrinsic(llvm::Function *func, SymbolTable *symbolTable) {
    Symbol *existingSym = symbolTable->LookupIntrinsics(func);
    if (existingSym != nullptr) {
        return existingSym;
    }
    SourcePos noPos;
    noPos.name = "LLVM Intrinsic";
    const llvm::FunctionType *ftype = func->getFunctionType();
    std::string name = std::string(func->getName());
    const Type *returnType = lLLVMTypeToISPCType(ftype->getReturnType(), false);
    if (returnType == nullptr) {
        Error(SourcePos(),
              "Return type not representable for "
              "Intrinsic %s.",
              name.c_str());
        // return type not representable in ispc -> not callable from ispc
        return nullptr;
    }
    llvm::SmallVector<const Type *, 8> argTypes;
    for (unsigned int j = 0; j < ftype->getNumParams(); ++j) {
        const llvm::Type *llvmArgType = ftype->getParamType(j);
        const Type *type = lLLVMTypeToISPCType(llvmArgType, false);
        if (type == nullptr) {
            Error(SourcePos(),
                  "Type of parameter %d not "
                  "representable for Intrinsic %s",
                  j, name.c_str());
            return nullptr;
        }
        argTypes.push_back(type);
    }
    FunctionType *funcType = new FunctionType(returnType, argTypes, noPos);
    Debug(noPos, "Created Intrinsic symbol \"%s\" [%s]\n", name.c_str(), funcType->GetString().c_str());
    Symbol *sym = new Symbol(name, noPos, funcType);
    sym->function = func;
    symbolTable->AddIntrinsics(sym);
    return sym;
}

/** Given an LLVM module, create ispc symbols for the functions in the
    module.
 */
void ispc::AddModuleSymbols(llvm::Module *module, SymbolTable *symbolTable) {
    llvm::Module::iterator iter;
    for (iter = module->begin(); iter != module->end(); ++iter) {
        llvm::Function *func = &*iter;
        lCreateISPCSymbol(func, symbolTable);
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

#if ISPC_LLVM_VERSION >= ISPC_LLVM_16_0
            // ReadNone, ReadOnly and WriteOnly are not supported for intrinsics anymore:
            FixFunctionAttribute(*Fn, llvm::Attribute::ReadNone, llvm::MemoryEffects::none());
            FixFunctionAttribute(*Fn, llvm::Attribute::ReadOnly, llvm::MemoryEffects::readOnly());
            FixFunctionAttribute(*Fn, llvm::Attribute::WriteOnly, llvm::MemoryEffects::writeOnly());
#endif
        }
    }
#endif
}

static void lSetAsInternal(llvm::Module *module, llvm::StringMap<int> &functions) {
    for (llvm::Function &F : module->functions()) {
        if (!F.isDeclaration() && functions.find(F.getName()) != functions.end()) {
            F.setLinkage(llvm::GlobalValue::InternalLinkage);
        }
    }
}

void ispc::AddBitcodeToModule(llvm::Module *bcModule, llvm::Module *module) {
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
                if (f.getNumUses() > 0)
                    continue;
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

void ispc::AddDeclarationsToModule(llvm::Module *bcModule, llvm::Module *module) {
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
            module->getOrInsertFunction(f.getName(), f.getFunctionType(), f.getAttributes());
        }
    }
}

void ispc::removeUnused(llvm::Module *M) {
    llvm::FunctionAnalysisManager FAM;
    llvm::ModuleAnalysisManager MAM;
    llvm::ModulePassManager PM;
    llvm::PassBuilder PB;
    PB.registerModuleAnalyses(MAM);
    PM.addPass(llvm::GlobalDCEPass());
    PM.run(*M, MAM);
}

void ispc::debugDumpModule(llvm::Module *module, std::string name, int stage) {
    if (!(g->off_stages.find(stage) == g->off_stages.end() && g->debug_stages.find(stage) != g->debug_stages.end())) {
        return;
    }

    name = std::string("pre_") + std::to_string(stage) + "_" + name + ".ll";
    if (g->dumpFile && !g->dumpFilePath.empty()) {
        std::error_code EC;
        llvm::SmallString<128> path(g->dumpFilePath);

        if (!llvm::sys::fs::exists(path)) {
            llvm::sys::fs::create_directories(g->dumpFilePath);
        }

        if (!g->singleTargetCompilation) {
            name += std::string("_") + g->target->GetISAString();
        }
        llvm::sys::path::append(path, name);
        llvm::raw_fd_ostream OS(path, EC);

        if (EC) {
            llvm::errs() << "Error: " << EC.message();
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
    AddDeclarationsToModule(dispatchBCModule, module);
    AddBitcodeToModule(dispatchBCModule, module);
}

void ispc::LinkCommonBuiltins(SymbolTable *symbolTable, llvm::Module *module) {
    const BitcodeLib *builtins = g->target_registry->getBuiltinsCLib(g->target_os, g->target->getArch());
    Assert(builtins);
    llvm::Module *builtinsBCModule = builtins->getLLVMModule();

    llvm::StringMap<int> commonBuiltins;
    for (llvm::Function &F : builtinsBCModule->functions()) {
        commonBuiltins[F.getName()] = 1;
    }

    // Unlike regular builtins and dispatch module, which don't care about mangling of external functions,
    // so they only differentiate Windows/Unix and 32/64 bit, builtins-c need to take care about mangling.
    // Hence, different version for all potentially supported OSes.
    AddBitcodeToModule(builtinsBCModule, module);
    lSetAsInternal(module, commonBuiltins);
}

llvm::Constant *lFuncAsConstInt8Ptr(llvm::Module &M, const char *name) {
    llvm::LLVMContext &Context = M.getContext();
    llvm::Function *F = M.getFunction(name);
    if (F) {
        return llvm::ConstantExpr::getBitCast(F, llvm::Type::getInt8PtrTy(Context));
    }
    return nullptr;
}

void ispc::addPersistentToLLVMUsed(llvm::Module &M) {
    llvm::LLVMContext &Context = M.getContext();

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
        // TODO comment that we don't need to preserve unused symbols chains
        if (isGroupUsed) {
            for (auto const &name : functions) {
                if (llvm::Constant *C = lFuncAsConstInt8Ptr(M, name)) {
                    ConstPtrs.push_back(C);
                }
            }
        }
    }

    for (auto const &[name, val] : persistentFuncs) {
        llvm::Function *F = M.getFunction(name);
        if (F) {
            llvm::Constant *FuncAsConst = llvm::ConstantExpr::getBitCast(F, llvm::Type::getInt8PtrTy(Context));
            ConstPtrs.push_back(FuncAsConst);
        }
    }

    if (ConstPtrs.empty()) {
        return;
    }

    // Create the array of i8* that llvm.used will hold
    llvm::ArrayType *ATy = llvm::ArrayType::get(llvm::Type::getInt8PtrTy(Context), ConstPtrs.size());
    llvm::Constant *ArrayInit = llvm::ConstantArray::get(ATy, ConstPtrs);

    // Create llvm.used and initialize it with the functions
    llvm::GlobalVariable *llvmUsed = new llvm::GlobalVariable(
        M, ArrayInit->getType(), false, llvm::GlobalValue::AppendingLinkage, ArrayInit, "llvm.compiler.used");
    llvmUsed->setSection("llvm.metadata");
}

void ispc::LinkTargetBuiltins(SymbolTable *symbolTable, llvm::Module *module) {
    const BitcodeLib *target =
        g->target_registry->getISPCTargetLib(g->target->getISPCTarget(), g->target_os, g->target->getArch());
    Assert(target);
    llvm::Module *targetBCModule = target->getLLVMModule();

    llvm::StringMap<int> targetBuiltins;
    for (llvm::Function &F : targetBCModule->functions()) {
        auto name = F.getName();
        // if (!name.startswith("llvm") && !isPersistent(name.str())) {
        if (!name.startswith("llvm")) {
            targetBuiltins[name] = 1;
        }
    }

    // TODO! it is to suppress warning about mismatch datalayout
    targetBCModule->setDataLayout(g->target->getDataLayout()->getStringRepresentation());

    // Next, add the target's custom implementations of the various needed
    // builtin functions (e.g. __masked_store_32(), etc).
    AddBitcodeToModule(targetBCModule, module);

    AddModuleSymbols(module, symbolTable);
    lSetAsInternal(module, targetBuiltins);
}

void ispc::LinkStdlib(SymbolTable *symbolTable, llvm::Module *module) {
    const BitcodeLib *stdlib =
        g->target_registry->getISPCStdLib(g->target->getISPCTarget(), g->target_os, g->target->getArch());
    Assert(stdlib);
    llvm::Module *stdlibBCModule = stdlib->getLLVMModule();

    if (!g->singleTargetCompilation) {
        for (llvm::Function &F : stdlibBCModule->functions()) {
            if (!F.isDeclaration() && !F.getName().startswith("llvm")) {
                F.setName(F.getName() + "_" + g->target->GetISAString());
            }
        }
    }

    llvm::StringMap<int> stdlibFunctions;
    for (llvm::Function &F : stdlibBCModule->functions()) {
        stdlibFunctions[F.getName()] = 1;
    }

    // TODO! add dump/debug functionality
    AddBitcodeToModule(stdlibBCModule, module);
    lSetAsInternal(module, stdlibFunctions);
}
