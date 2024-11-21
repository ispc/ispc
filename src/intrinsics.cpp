/*
  Copyright (c) 2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include "expr.h"
#include "llvmutil.h"
#include "module.h"
#include "sym.h"
#include "type.h"
#include "util.h"

#include <string>

#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Module.h>
#include <llvm/Target/TargetMachine.h>
#ifdef ISPC_XE_ENABLED
#include <llvm/GenXIntrinsics/GenXIntrinsics.h>
#endif
#include <llvm/Target/TargetIntrinsicInfo.h>

using namespace ispc;

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
    if (t == LLVMTypes::VoidType) {
        return AtomicType::Void;
    }

    // uniform
    else if (t == LLVMTypes::BoolType) {
        return AtomicType::UniformBool;
    } else if (t == LLVMTypes::Int8Type) {
        return intAsUnsigned ? AtomicType::UniformUInt8 : AtomicType::UniformInt8;
    } else if (t == LLVMTypes::Int16Type) {
        return intAsUnsigned ? AtomicType::UniformUInt16 : AtomicType::UniformInt16;
    } else if (t == LLVMTypes::Int32Type) {
        return intAsUnsigned ? AtomicType::UniformUInt32 : AtomicType::UniformInt32;
    } else if (t == LLVMTypes::Float16Type) {
        return AtomicType::UniformFloat16;
    } else if (t == LLVMTypes::FloatType) {
        return AtomicType::UniformFloat;
    } else if (t == LLVMTypes::DoubleType) {
        return AtomicType::UniformDouble;
    } else if (t == LLVMTypes::Int64Type) {
        return intAsUnsigned ? AtomicType::UniformUInt64 : AtomicType::UniformInt64;
    }

    // varying
    if (t == LLVMTypes::Int8VectorType) {
        return intAsUnsigned ? AtomicType::VaryingUInt8 : AtomicType::VaryingInt8;
    } else if (t == LLVMTypes::Int16VectorType) {
        return intAsUnsigned ? AtomicType::VaryingUInt16 : AtomicType::VaryingInt16;
    } else if (t == LLVMTypes::Int32VectorType) {
        return intAsUnsigned ? AtomicType::VaryingUInt32 : AtomicType::VaryingInt32;
    } else if (t == LLVMTypes::Float16VectorType) {
        return AtomicType::VaryingFloat16;
    } else if (t == LLVMTypes::FloatVectorType) {
        return AtomicType::VaryingFloat;
    } else if (t == LLVMTypes::DoubleVectorType) {
        return AtomicType::VaryingDouble;
    } else if (t == LLVMTypes::Int64VectorType) {
        return intAsUnsigned ? AtomicType::VaryingUInt64 : AtomicType::VaryingInt64;
    } else if (t == LLVMTypes::MaskType) {
        return AtomicType::VaryingBool;
    }

    // pointers to uniform
    else if (t == LLVMTypes::Int8PointerType) {
        return PointerType::GetUniform(intAsUnsigned ? AtomicType::UniformUInt8 : AtomicType::UniformInt8);
    } else if (t == LLVMTypes::Int16PointerType) {
        return PointerType::GetUniform(intAsUnsigned ? AtomicType::UniformUInt16 : AtomicType::UniformInt16);
    } else if (t == LLVMTypes::Int32PointerType) {
        return PointerType::GetUniform(intAsUnsigned ? AtomicType::UniformUInt32 : AtomicType::UniformInt32);
    } else if (t == LLVMTypes::Int64PointerType) {
        return PointerType::GetUniform(intAsUnsigned ? AtomicType::UniformUInt64 : AtomicType::UniformInt64);
    } else if (t == LLVMTypes::Float16PointerType) {
        return PointerType::GetUniform(AtomicType::UniformFloat16);
    } else if (t == LLVMTypes::FloatPointerType) {
        return PointerType::GetUniform(AtomicType::UniformFloat);
    } else if (t == LLVMTypes::DoublePointerType) {
        return PointerType::GetUniform(AtomicType::UniformDouble);
    }

    // pointers to varying
    else if (t == LLVMTypes::Int8VectorPointerType) {
        return PointerType::GetUniform(intAsUnsigned ? AtomicType::VaryingUInt8 : AtomicType::VaryingInt8);
    } else if (t == LLVMTypes::Int16VectorPointerType) {
        return PointerType::GetUniform(intAsUnsigned ? AtomicType::VaryingUInt16 : AtomicType::VaryingInt16);
    } else if (t == LLVMTypes::Int32VectorPointerType) {
        return PointerType::GetUniform(intAsUnsigned ? AtomicType::VaryingUInt32 : AtomicType::VaryingInt32);
    } else if (t == LLVMTypes::Int64VectorPointerType) {
        return PointerType::GetUniform(intAsUnsigned ? AtomicType::VaryingUInt64 : AtomicType::VaryingInt64);
    } else if (t == LLVMTypes::Float16VectorPointerType) {
        return PointerType::GetUniform(AtomicType::VaryingFloat16);
    } else if (t == LLVMTypes::FloatVectorPointerType) {
        return PointerType::GetUniform(AtomicType::VaryingFloat);
    } else if (t == LLVMTypes::DoubleVectorPointerType) {
        return PointerType::GetUniform(AtomicType::VaryingDouble);
    }

    return nullptr;
}

/** Create ISPC symbol for LLVM intrinsics and add it to the given module.

    @param func            llvm::Function for the intrinsic to be added
    @param symbolTable     SymbolTable in which to add symbol definitions
    @return                Symbol created for the LLVM::Function
 */
Symbol *lCreateISPCSymbolForLLVMIntrinsic(llvm::Function *func, SymbolTable *symbolTable) {
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
    Symbol *sym = new Symbol(name, noPos, Symbol::SymbolKind::Function, funcType);
    sym->function = func;
    symbolTable->AddIntrinsics(sym);
    return sym;
}

static std::vector<llvm::Type *> lDeductArgTypes(llvm::Intrinsic::ID ID, ExprList *args) {
    std::vector<llvm::Type *> exprType;
    if (llvm::Intrinsic::isOverloaded(ID)) {
        Assert(args);
        int nInits = args->exprs.size();
        for (int i = 0; i < nInits; ++i) {
            const Type *argType = (args->exprs[i])->GetType();
            Assert(argType);
            exprType.push_back(argType->LLVMType(g->ctx));
        }
    }
    return exprType;
}

static llvm::Intrinsic::ID lLookupIntrinsicID(llvm::StringRef name) {
#if ISPC_LLVM_VERSION >= ISPC_LLVM_20_0
    return llvm::Intrinsic::lookupIntrinsicID(llvm::StringRef(name));
#else
    return llvm::Function::lookupIntrinsicID(llvm::StringRef(name));
#endif
}

static llvm::Function *lGetFunctionDeclaration(llvm::Module *module, llvm::Intrinsic::ID ID,
                                               std::vector<llvm::Type *> &exprType) {
#if ISPC_LLVM_VERSION >= ISPC_LLVM_20_0
    return llvm::Intrinsic::getOrInsertDeclaration(module, ID, exprType);
#else
    return llvm::Intrinsic::getDeclaration(module, ID, exprType);
#endif
}

#ifdef ISPC_XE_ENABLED
static llvm::Function *lGetGenXIntrinsicDeclaration(llvm::Module *module, const std::string &name, ExprList *args,
                                                    SourcePos pos) {
    llvm::GenXIntrinsic::ID ID = llvm::GenXIntrinsic::lookupGenXIntrinsicID(name);
    if (ID == llvm::GenXIntrinsic::not_any_intrinsic) {
        Error(pos, "LLVM intrinsic \"%s\" not supported.", name.c_str());
        return nullptr;
    }
    std::vector<llvm::Type *> exprType;
    Assert(args);
    int nInits = args->exprs.size();
    if (llvm::GenXIntrinsic::isOverloadedRet(ID) || llvm::GenXIntrinsic::isOverloadedArg(ID, nInits)) {
        for (int i = 0; i < nInits; ++i) {
            const Type *argType = (args->exprs[i])->GetType();
            Assert(argType);
            exprType.push_back(argType->LLVMType(g->ctx));
        }
    }
    llvm::ArrayRef<llvm::Type *> argArr(exprType);
    llvm::Function *funcDecl = llvm::GenXIntrinsic::getGenXDeclaration(module, ID, argArr);
    if (funcDecl) {
        // ReadNone, ReadOnly and WriteOnly are not supported for intrinsics anymore:
        FixFunctionAttribute(*funcDecl, llvm::Attribute::ReadNone, llvm::MemoryEffects::none());
        FixFunctionAttribute(*funcDecl, llvm::Attribute::ReadOnly, llvm::MemoryEffects::readOnly());
        FixFunctionAttribute(*funcDecl, llvm::Attribute::WriteOnly, llvm::MemoryEffects::writeOnly());
    }
    return funcDecl;
}
#endif // ISPC_XE_ENABLED

static llvm::Function *lGetIntrinsicDeclaration(llvm::Module *module, const std::string &name, ExprList *args,
                                                SourcePos pos) {
#ifdef ISPC_XE_ENABLED
    if (g->target->isXeTarget()) {
        return lGetGenXIntrinsicDeclaration(module, name, args, pos);
    }
#endif // ISPC_XE_ENABLED
    if (!g->target->isXeTarget()) {
        llvm::TargetMachine *targetMachine = g->target->GetTargetMachine();
        const llvm::TargetIntrinsicInfo *TII = targetMachine->getIntrinsicInfo();
        llvm::Intrinsic::ID ID = lLookupIntrinsicID(name);
        if (ID == llvm::Intrinsic::not_intrinsic && TII) {
            ID = static_cast<llvm::Intrinsic::ID>(TII->lookupName(llvm::StringRef(name)));
        }
        if (ID == llvm::Intrinsic::not_intrinsic) {
            Error(pos, "LLVM intrinsic \"%s\" not supported.", name.c_str());
            return nullptr;
        }
        // For LLVM intrinsics, we need to deduce the argument LLVM types.
        auto exprType = lDeductArgTypes(ID, args);
        llvm::Function *funcDecl = lGetFunctionDeclaration(module, ID, exprType);
        llvm::StringRef funcName = funcDecl->getName();
        if (g->target->checkIntrinsticSupport(funcName, pos) == false) {
            return nullptr;
        }
        return funcDecl;
    }
    return nullptr;
}

Symbol *Module::AddLLVMIntrinsicDecl(const std::string &name, ExprList *args, SourcePos pos) {
    if (g->enableLLVMIntrinsics == false) {
        Error(SourcePos(), "Calling LLVM intrinsics from ISPC source code is an experimental feature,"
                           " which can be enabled by passing \"--enable-llvm-intrinsics\" switch to the compiler.\n");
        return nullptr;
    }

    llvm::Function *funcDecl = lGetIntrinsicDeclaration(module, name, args, pos);
    Assert(funcDecl != nullptr);

    Symbol *funcSym = lCreateISPCSymbolForLLVMIntrinsic(funcDecl, symbolTable);
    return funcSym;
}
