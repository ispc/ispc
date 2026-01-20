/*
  Copyright (c) 2024-2025, Intel Corporation

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
#include <llvm/IR/Attributes.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#ifdef ISPC_XE_ENABLED
#include <llvm/GenXIntrinsics/GenXIntrinsics.h>
#endif

using namespace ispc;

using IITDesc = llvm::Intrinsic::IITDescriptor;

enum class ISPCIntrinsics : unsigned {
    not_intrinsic = 0,
    atomicrmw,
    bitcast,
    blend_store,
    concat,
    cmpxchg,
    extract,
    fence,
    insert,
    packmask,
    select,
    stream_load,
    stream_store,
};

static ISPCIntrinsics lLookupISPCInstrinsic(const std::string &name) {
    llvm::StringRef ref(name);
    // These are overloaded intrinsics with suffixes that contain ordering.
    // See src/opt/LowerISPCIntrinsics.cpp for more details.
    if (ref.starts_with("llvm.ispc.atomicrmw.")) {
        return ISPCIntrinsics::atomicrmw;
    } else if (ref.starts_with("llvm.ispc.cmpxchg.")) {
        return ISPCIntrinsics::cmpxchg;
    } else if (ref.starts_with("llvm.ispc.fence.")) {
        return ISPCIntrinsics::fence;
    }

    // These intrinsics are not overloaded.
    else if (name == "llvm.ispc.bitcast") {
        return ISPCIntrinsics::bitcast;
    } else if (name == "llvm.ispc.blend_store") {
        return ISPCIntrinsics::blend_store;
    } else if (name == "llvm.ispc.concat") {
        return ISPCIntrinsics::concat;
    } else if (name == "llvm.ispc.extract") {
        return ISPCIntrinsics::extract;
    } else if (name == "llvm.ispc.insert") {
        return ISPCIntrinsics::insert;
    } else if (name == "llvm.ispc.packmask") {
        return ISPCIntrinsics::packmask;
    } else if (name == "llvm.ispc.select") {
        return ISPCIntrinsics::select;
    } else if (name == "llvm.ispc.stream_load") {
        return ISPCIntrinsics::stream_load;
    } else if (name == "llvm.ispc.stream_store") {
        return ISPCIntrinsics::stream_store;
    } else {
        return ISPCIntrinsics::not_intrinsic;
    }
}

// llvm/lib/IR/Function.cpp:(getMangledTypeStr)
static std::string lGetMangledTypeStr(llvm::Type *Ty, bool &HasUnnamedType) {
    std::string Result;
    if (llvm::PointerType *PTyp = llvm::dyn_cast<llvm::PointerType>(Ty)) {
        Result += "p" + llvm::utostr(PTyp->getAddressSpace());
    } else if (llvm::ArrayType *ATyp = llvm::dyn_cast<llvm::ArrayType>(Ty)) {
        Result +=
            "a" + llvm::utostr(ATyp->getNumElements()) + lGetMangledTypeStr(ATyp->getElementType(), HasUnnamedType);
    } else if (llvm::StructType *STyp = llvm::dyn_cast<llvm::StructType>(Ty)) {
        if (!STyp->isLiteral()) {
            Result += "s_";
            if (STyp->hasName()) {
                Result += STyp->getName();
            } else {
                HasUnnamedType = true;
            }
        } else {
            Result += "sl_";
            for (auto *Elem : STyp->elements()) {
                Result += lGetMangledTypeStr(Elem, HasUnnamedType);
            }
        }
        // Ensure nested structs are distinguishable.
        Result += "s";
    } else if (llvm::FunctionType *FT = llvm::dyn_cast<llvm::FunctionType>(Ty)) {
        Result += "f_" + lGetMangledTypeStr(FT->getReturnType(), HasUnnamedType);
        for (size_t i = 0; i < FT->getNumParams(); i++) {
            Result += lGetMangledTypeStr(FT->getParamType(i), HasUnnamedType);
        }
        if (FT->isVarArg()) {
            Result += "vararg";
        }
        // Ensure nested function types are distinguishable.
        Result += "f";
    } else if (llvm::VectorType *VTy = llvm::dyn_cast<llvm::VectorType>(Ty)) {
        llvm::ElementCount EC = VTy->getElementCount();
        if (EC.isScalable()) {
            Result += "nx";
        }
        Result += "v" + llvm::utostr(EC.getKnownMinValue()) + lGetMangledTypeStr(VTy->getElementType(), HasUnnamedType);
    } else if (llvm::TargetExtType *TETy = llvm::dyn_cast<llvm::TargetExtType>(Ty)) {
        Result += "t";
        Result += TETy->getName();
        for (llvm::Type *ParamTy : TETy->type_params()) {
            Result += "_" + lGetMangledTypeStr(ParamTy, HasUnnamedType);
        }
        for (unsigned IntParam : TETy->int_params()) {
            Result += "_" + llvm::utostr(IntParam);
        }
        // Ensure nested target extension types are distinguishable.
        Result += "t";
    } else if (Ty) {
        switch (Ty->getTypeID()) {
        default:
            UNREACHABLE();
        case llvm::Type::VoidTyID:
            Result += "isVoid";
            break;
        case llvm::Type::MetadataTyID:
            Result += "Metadata";
            break;
        case llvm::Type::HalfTyID:
            Result += "f16";
            break;
        case llvm::Type::BFloatTyID:
            Result += "bf16";
            break;
        case llvm::Type::FloatTyID:
            Result += "f32";
            break;
        case llvm::Type::DoubleTyID:
            Result += "f64";
            break;
        case llvm::Type::X86_FP80TyID:
            Result += "f80";
            break;
        case llvm::Type::FP128TyID:
            Result += "f128";
            break;
        case llvm::Type::PPC_FP128TyID:
            Result += "ppcf128";
            break;

#if ISPC_LLVM_VERSION < ISPC_LLVM_20_0
        case llvm::Type::X86_MMXTyID:
            Result += "x86mmx";
            break;
#endif
        case llvm::Type::X86_AMXTyID:
            Result += "x86amx";
            break;
        case llvm::Type::IntegerTyID:
            Result += "i" + llvm::utostr(llvm::cast<llvm::IntegerType>(Ty)->getBitWidth());
            break;
        }
    }
    return Result;
}

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
    } else if (t == LLVMTypes::Int1VectorType) {
        return AtomicType::VaryingInt1;
    }

    // pointer
    else if (t == LLVMTypes::PtrType) {
        return PointerType::GetUniform(AtomicType::UniformInt8);
    }
    // vector of pointers
    else if (t == LLVMTypes::PtrVectorType) {
        return AtomicType::VaryingUInt64;
    }

    // vector with length different from TARGET_WIDTH can be repsented as uniform TYPE<N>
    else if (const llvm::VectorType *vt = llvm::dyn_cast<llvm::VectorType>(t)) {
        // check if vector length is equal to TARGET_WIDTH
        unsigned int vectorWidth = vt->getElementCount().getKnownMinValue();
        if (vectorWidth == static_cast<unsigned int>(g->target->getVectorWidth())) {
            // we should never hit this case, because it should be handled by the cases above
            return nullptr;
        } else {
            const Type *elementType = lLLVMTypeToISPCType(vt->getElementType(), intAsUnsigned);
            if (elementType == nullptr) {
                return nullptr;
            }
            return new VectorType(elementType, vectorWidth);
        }
    }

    return nullptr;
}

std::string lGetTypeString(const llvm::Type *Ty) {
    std::string TypeStr;
    llvm::raw_string_ostream RSO(TypeStr);
    Ty->print(RSO);
    return RSO.str();
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
        Error(SourcePos(), "Return type \"%s\" not representable as ISPC type for Intrinsic %s.",
              lGetTypeString(ftype->getReturnType()).c_str(), name.c_str());
        // return type not representable in ispc -> not callable from ispc
        return nullptr;
    }
    llvm::SmallVector<const Type *, 8> argTypes;
    for (unsigned int j = 0; j < ftype->getNumParams(); ++j) {
        const llvm::Type *llvmArgType = ftype->getParamType(j);
        const Type *type = lLLVMTypeToISPCType(llvmArgType, false);
        if (type == nullptr) {
            Error(SourcePos(), "Type \"%s\" of parameter %d not representable as ISPC type for Intrinsic %s",
                  lGetTypeString(llvmArgType).c_str(), j, name.c_str());
            return nullptr;
        }
        argTypes.push_back(type);
    }
    FunctionType *funcType = new FunctionType(returnType, argTypes, noPos);
    if (funcType != nullptr) {
        llvm::AttributeList attrs = func->getAttributes();
        unsigned int numParams = func->arg_size();
        for (unsigned int j = 0; j < numParams; ++j) {
            if (attrs.hasParamAttr(j, llvm::Attribute::ImmArg)) {
                funcType->SetParameterImmArg(j, true);
            }
        }
    }
    Debug(noPos, "Created Intrinsic symbol \"%s\" [%s]\n", name.c_str(), funcType->GetString().c_str());
    Symbol *sym = new Symbol(name, noPos, Symbol::SymbolKind::Function, funcType);
    sym->function = func;
    symbolTable->AddIntrinsics(sym);
    return sym;
}

static llvm::VectorType *lGetDoubleWidthVectorType(llvm::Type *type) {
    llvm::VectorType *vt = llvm::dyn_cast<llvm::VectorType>(type);
    if (vt == nullptr) {
        Error(SourcePos(), "Expected vector type.");
        return nullptr;
    }
    unsigned int vectorWidth = vt->getElementCount().getKnownMinValue();
    return llvm::VectorType::get(vt->getElementType(), vectorWidth * 2, false);
}

static llvm::Function *lGetISPCIntrinsicsFuncDecl(llvm::Module *M, std::string name, ISPCIntrinsics ID,
                                                  std::vector<const Type *> &argTypes) {
    llvm::Type *retType = nullptr;
    std::vector<llvm::Type *> TYs;
    for (const Type *type : argTypes) {
        TYs.push_back(type->LLVMType(g->ctx));
    }
    bool hasUnnamedType = false;
    switch (ID) {
    case ISPCIntrinsics::not_intrinsic:
        Error(SourcePos(), "Not ISPC intrinsic.");
        return nullptr;
    case ISPCIntrinsics::atomicrmw: {
        Assert(TYs.size() == 2);
        retType = TYs[1];
        name += "." + lGetMangledTypeStr(TYs[1], hasUnnamedType);
        break;
    }
    case ISPCIntrinsics::bitcast: {
        Assert(TYs.size() == 2 && TYs[0]->getPrimitiveSizeInBits() == TYs[1]->getPrimitiveSizeInBits());
        retType = TYs[1];
        name += "." + lGetMangledTypeStr(TYs[0], hasUnnamedType) + "." + lGetMangledTypeStr(TYs[1], hasUnnamedType);
        break;
    }
    case ISPCIntrinsics::blend_store: {
        Assert(TYs.size() == 3);
        retType = llvm::Type::getVoidTy(*g->ctx);
        name += "." + lGetMangledTypeStr(TYs[0], hasUnnamedType);
        break;
    }
    case ISPCIntrinsics::concat: {
        assert(argTypes.size() == 2 && Type::Equal(argTypes[0], argTypes[1]));
        retType = lGetDoubleWidthVectorType(TYs[0]);
        name += "." + lGetMangledTypeStr(retType, hasUnnamedType) + "." + lGetMangledTypeStr(TYs[1], hasUnnamedType);
        break;
    }
    case ISPCIntrinsics::cmpxchg: {
        Assert(TYs.size() == 3);
        retType = TYs[1];
        name += "." + lGetMangledTypeStr(TYs[1], hasUnnamedType);
        break;
    }
    case ISPCIntrinsics::extract: {
        Assert(TYs.size() == 2 || TYs.size() == 3);
        llvm::VectorType *vt = llvm::dyn_cast<llvm::VectorType>(TYs[0]);
        Assert(vt);
        retType = vt->getElementType();
        name += "." + lGetMangledTypeStr(TYs[0], hasUnnamedType);
        if (argTypes.size() == 3) {
            name += "." + lGetMangledTypeStr(TYs[1], hasUnnamedType);
        }
        break;
    }
    case ISPCIntrinsics::fence: {
        retType = llvm::Type::getVoidTy(*g->ctx);
        break;
    }
    case ISPCIntrinsics::insert: {
        Assert(TYs.size() == 3);
        llvm::VectorType *vt = llvm::dyn_cast<llvm::VectorType>(TYs[0]);
        Assert(vt);
        retType = vt;
        name += "." + lGetMangledTypeStr(TYs[0], hasUnnamedType);
        break;
    }
    case ISPCIntrinsics::packmask: {
        Assert(TYs.size() == 1);
        retType = llvm::Type::getInt64Ty(*g->ctx);
        name += "." + lGetMangledTypeStr(TYs[0], hasUnnamedType);
        break;
    }
    case ISPCIntrinsics::select: {
        Assert(TYs.size() == 3);
        retType = TYs[1];
        name += "." + lGetMangledTypeStr(TYs[1], hasUnnamedType);
        break;
    }
    case ISPCIntrinsics::stream_load: {
        Assert(TYs.size() == 2);
        retType = TYs[1];
        name += "." + lGetMangledTypeStr(TYs[1], hasUnnamedType);
        break;
    }
    case ISPCIntrinsics::stream_store: {
        Assert(TYs.size() == 2);
        retType = llvm::Type::getVoidTy(*g->ctx);
        name += "." + lGetMangledTypeStr(TYs[1], hasUnnamedType);
        break;
    }
    default: {
        Error(SourcePos(), "Unknown ISPC intrinsic ID \"%u\".", (unsigned)ID);
        return nullptr;
    }
    }
    llvm::FunctionType *funcType = llvm::FunctionType::get(retType, TYs, false);
    auto p = llvm::cast<llvm::Function>(M->getOrInsertFunction(name, funcType).getCallee());
    return p;
}

bool lIsAnyArgumentKind(const IITDesc &D) {
    if (D.Kind != IITDesc::Argument) {
        return false;
    }

    unsigned Kind = D.getArgumentKind();
    return Kind == IITDesc::AK_Any || Kind == IITDesc::AK_AnyInteger || Kind == IITDesc::AK_AnyFloat ||
           Kind == IITDesc::AK_AnyVector || Kind == IITDesc::AK_AnyPointer;
}

bool lIsArgDesc(const IITDesc &D) {
    bool IsArgDescriptor = D.Kind == IITDesc::Argument || D.Kind == IITDesc::ExtendArgument ||
                           D.Kind == IITDesc::TruncArgument ||
#if ISPC_LLVM_VERSION < ISPC_LLVM_21_0
                           D.Kind == IITDesc::HalfVecArgument ||
#endif
                           D.Kind == IITDesc::SameVecWidthArgument || D.Kind == IITDesc::VecElementArgument ||
                           D.Kind == IITDesc::Subdivide2Argument || D.Kind == IITDesc::Subdivide4Argument ||
                           D.Kind == IITDesc::VecOfBitcastsToInt;
    return IsArgDescriptor;
}

bool lIsAliased(const IITDesc &L, const IITDesc &R) {
#if ISPC_LLVM_VERSION >= ISPC_LLVM_21_0
    if (R.Kind == IITDesc::OneNthEltsVecArgument) {
        return R.getArgumentNumber() == L.getArgumentNumber();
    }
#endif
    // Check if R is argument with MatchType and has the same argument number as L
    return lIsArgDesc(R) && R.getArgumentKind() == IITDesc::AK_MatchType && R.Kind == IITDesc::Argument &&
           R.getArgumentNumber() == L.getArgumentNumber();
}

size_t lNextIITDescs(const IITDesc &D) {
    switch (D.Kind) {
    case IITDesc::Struct:
        return D.Struct_NumElements + 1;
    case IITDesc::Vector:
    case IITDesc::SameVecWidthArgument:
        return 2;
    default:
        return 1;
    }
}

// The function analyzes an LLVM intrinsic (identified by its ID) and finds
// which argument positions can accept any type. It returns these
// positions in the provided result vector.
void lGetOverloadedArgumentIndices(llvm::Intrinsic::ID ID, std::vector<size_t> &result, SourcePos pos) {
    // The LLVM Intrinsic Information Table (IIT) is a data structure that
    // describes the type system and constraints for LLVM's built-in functions
    // (intrinsics). The table contains a series of descriptors, where the
    // first entry typically describes the return type, followed by entries for
    // each argument (some arguments are represented by multiple entries).
    // These descriptors can express complex relationships like "this argument
    // must match the return type" or "this argument needs to be a vector of
    // the same width as another argument." The table can specify that it works
    // with "any" type, or that multiple arguments need to match each other's
    // types.
    //
    // We need to traverse this table to find the argument positions that can
    // be any type. We need only the first entry and one entry for every unique
    // "any" type.

    llvm::SmallVector<IITDesc, 8> Table;
    llvm::Intrinsic::getIntrinsicInfoTableEntries(ID, Table);

    if (Table.empty()) {
        return;
    }

    // The trick part is that return type often is any. However in ISPC, we
    // have no return type information, so we need to find the argument with
    // the same type as return value.
    size_t argIndex = 0;
    size_t i = lNextIITDescs(Table[0]);
    if (lIsAnyArgumentKind(Table[0])) {
        while (i < Table.size()) {
            if (lIsAliased(Table[0], Table[i])) {
                result.push_back(argIndex);
                break;
            }
            i += lNextIITDescs(Table[i]);
            argIndex++;
        }

        if (result.empty()) {
            // If we didn't find any matching argument, we can create a proper
            // function declaration. I don't know if there are such intrinsics.
            llvm::StringRef name = llvm::Intrinsic::getBaseName(ID);
            Error(pos, "Can't deduce types for function declaration of the overloaded LLVM intrinsic \"%s\".",
                  name.str().c_str());
            return;
        }
    }

    // Find remaining "any" types for arguments.
    argIndex = 0;
    i = lNextIITDescs(Table[0]);
    while (i < Table.size()) {
        if (Table[i].Kind == IITDesc::VecOfAnyPtrsToElt || lIsAnyArgumentKind(Table[i])) {
            result.push_back(argIndex);
        }
        i += lNextIITDescs(Table[i]);
        argIndex++;
    }
}

static std::vector<llvm::Type *> lDeductArgTypes(llvm::Module *module, llvm::Intrinsic::ID ID, ExprList *args,
                                                 SourcePos pos) {
    std::vector<llvm::Type *> exprType;
    if (llvm::Intrinsic::isOverloaded(ID)) {
        Assert(args);

        std::vector<size_t> nInits;
        lGetOverloadedArgumentIndices(ID, nInits, pos);

        Assert(nInits.size() > 0);
        for (size_t i : nInits) {
            Assert(i < args->exprs.size());
            const Type *argType = (args->exprs[i])->GetType();
            Assert(argType);
            if (ID == llvm::Intrinsic::masked_gather && i == 0) {
                // type with <TARGET_WIDTH x ptr> is expected
                exprType.push_back(LLVMTypes::PtrVectorType);
            } else if (ID == llvm::Intrinsic::masked_scatter && i == 1) {
                // type with <TARGET_WIDTH x ptr> is expected
                exprType.push_back(LLVMTypes::PtrVectorType);
            } else {
                exprType.push_back(argType->LLVMType(g->ctx));
            }
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
        llvm::Intrinsic::ID ID = lLookupIntrinsicID(name);

        if (ID == llvm::Intrinsic::not_intrinsic) {
            // Check if it is an ISPC intrinsic
            ISPCIntrinsics IID = lLookupISPCInstrinsic(name);
            if (IID != ISPCIntrinsics::not_intrinsic) {
                if (args) {
                    std::vector<const Type *> argTypes(args->exprs.size(), nullptr);
                    for (size_t i = 0; i < args->exprs.size(); i++) {
                        argTypes[i] = args->exprs[i]->GetType();
                    }
                    return lGetISPCIntrinsicsFuncDecl(module, name, IID, argTypes);
                } else {
                    std::vector<const Type *> argTypes;
                    return lGetISPCIntrinsicsFuncDecl(module, name, IID, argTypes);
                }
            } else {
                Error(pos, "LLVM intrinsic \"%s\" not supported.", name.c_str());
                return nullptr;
            }
        } else {
            // For LLVM intrinsics, we need to deduce the argument LLVM types.
            auto exprType = lDeductArgTypes(module, ID, args, pos);
            llvm::Function *funcDecl = lGetFunctionDeclaration(module, ID, exprType);
            llvm::StringRef funcName = funcDecl->getName();
            if (g->target->checkIntrinsticSupport(funcName, pos) == false) {
                return nullptr;
            }
            return funcDecl;
        }
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
    if (funcDecl == nullptr) {
        return nullptr;
    }

    Symbol *funcSym = lCreateISPCSymbolForLLVMIntrinsic(funcDecl, symbolTable);
    return funcSym;
}
