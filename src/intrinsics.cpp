/*
  Copyright (c) 2024-2026, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include "expr.h"
#include "llvmutil.h"
#include "module.h"
#include "sym.h"
#include "type.h"
#include "util.h"

#include <algorithm>
#include <string>

#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Intrinsics.h>
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
    @param name            Name of the intrinsic as written in the source code
    @param pos             Position of the call, used for error messages
    @return                Symbol created for the LLVM::Function
 */
static Symbol *lCreateISPCSymbolForLLVMIntrinsic(llvm::Function *func, SymbolTable *symbolTable,
                                                 const std::string &name, SourcePos pos) {
    Symbol *existingSym = symbolTable->LookupIntrinsics(func);
    if (existingSym != nullptr) {
        return existingSym;
    }
    SourcePos noPos;
    noPos.name = "LLVM Intrinsic";
    const llvm::FunctionType *ftype = func->getFunctionType();
    std::string funcName = std::string(func->getName());
    const Type *returnType = lLLVMTypeToISPCType(ftype->getReturnType(), false);
    if (returnType == nullptr) {
        // return type not representable in ispc -> not callable from ispc
        Error(pos, "Return type \"%s\" of LLVM intrinsic \"%s\" is not representable as an ISPC type.",
              lGetTypeString(ftype->getReturnType()).c_str(), name.c_str());
        return nullptr;
    }
    llvm::SmallVector<const Type *, 8> argTypes;
    for (unsigned int j = 0; j < ftype->getNumParams(); ++j) {
        const llvm::Type *llvmArgType = ftype->getParamType(j);
        const Type *type = lLLVMTypeToISPCType(llvmArgType, false);
        if (type == nullptr) {
            Error(pos, "Type \"%s\" of parameter %u of LLVM intrinsic \"%s\" is not representable as an ISPC type.",
                  lGetTypeString(llvmArgType).c_str(), j + 1, name.c_str());
            return nullptr;
        }
        argTypes.push_back(type);
    }
    FunctionType *funcType = new FunctionType(returnType, argTypes, noPos);
    Debug(noPos, "Created Intrinsic symbol \"%s\" [%s]\n", funcName.c_str(), funcType->GetString().c_str());
    Symbol *sym = new Symbol(funcName, noPos, Symbol::SymbolKind::Function, funcType);
    sym->function = func;
    symbolTable->AddIntrinsics(sym);
    return sym;
}

static bool lIsMemoryOrdering(llvm::StringRef str) {
    return str == "unordered" || str == "monotonic" || str == "acquire" || str == "release" || str == "acq_rel" ||
           str == "seq_cst";
}

static bool lIsAtomicRMWOp(llvm::StringRef str) {
    return str == "xchg" || str == "add" || str == "sub" || str == "and" || str == "nand" || str == "or" ||
           str == "xor" || str == "max" || str == "min" || str == "umax" || str == "umin" || str == "fadd" ||
           str == "fsub" || str == "fmax" || str == "fmin";
}

// Check the suffix of llvm.ispc.atomicrmw.<op>.<ordering>, llvm.ispc.cmpxchg.<ordering>.<ordering> and
// llvm.ispc.fence.<ordering>, which is parsed by LowerISPCIntrinsicsPass.
static bool lCheckISPCIntrinsicSuffix(const std::string &name, ISPCIntrinsics ID, SourcePos pos) {
    llvm::StringRef ref(name);
    llvm::SmallVector<llvm::StringRef, 2> parts;
    const char *expected = nullptr;
    bool valid = false;
    switch (ID) {
    case ISPCIntrinsics::atomicrmw:
        ref.consume_front("llvm.ispc.atomicrmw.");
        ref.split(parts, '.');
        valid = parts.size() == 2 && lIsAtomicRMWOp(parts[0]) && lIsMemoryOrdering(parts[1]);
        expected = "llvm.ispc.atomicrmw.<operation>.<ordering>";
        break;
    case ISPCIntrinsics::cmpxchg:
        ref.consume_front("llvm.ispc.cmpxchg.");
        ref.split(parts, '.');
        valid = parts.size() == 2 && lIsMemoryOrdering(parts[0]) && lIsMemoryOrdering(parts[1]);
        expected = "llvm.ispc.cmpxchg.<ordering>.<ordering>";
        break;
    case ISPCIntrinsics::fence:
        ref.consume_front("llvm.ispc.fence.");
        valid = lIsMemoryOrdering(ref);
        expected = "llvm.ispc.fence.<ordering>";
        break;
    default:
        return true;
    }
    if (!valid) {
        Error(pos, "Invalid ISPC intrinsic name \"%s\", expected \"%s\".", name.c_str(), expected);
    }
    return valid;
}

static bool lCheckISPCIntrinsicArgCount(const std::string &name, const std::vector<const Type *> &argTypes,
                                        size_t expected, SourcePos pos) {
    if (argTypes.size() != expected) {
        Error(pos, "ISPC intrinsic \"%s\" expects %zu argument%s, %zu provided.", name.c_str(), expected,
              expected == 1 ? "" : "s", argTypes.size());
        return false;
    }
    return true;
}

static bool lCheckISPCIntrinsicArg(const std::string &name, const std::vector<const Type *> &argTypes, size_t i,
                                   bool isValid, const char *expected, SourcePos pos) {
    if (!isValid) {
        Error(pos, "Argument %zu of ISPC intrinsic \"%s\" must be %s, \"%s\" provided.", i + 1, name.c_str(), expected,
              argTypes[i]->GetString().c_str());
    }
    return isValid;
}

static llvm::Function *lGetISPCIntrinsicsFuncDecl(llvm::Module *M, const std::string &srcName, ISPCIntrinsics ID,
                                                  const std::vector<const Type *> &argTypes, SourcePos pos) {
    if (!lCheckISPCIntrinsicSuffix(srcName, ID, pos)) {
        return nullptr;
    }

    std::string name = srcName;
    llvm::Type *retType = nullptr;
    std::vector<llvm::Type *> TYs;
    for (const Type *type : argTypes) {
        llvm::Type *llvmType = type->LLVMType(g->ctx);
        if (llvmType == nullptr || llvmType->isVoidTy()) {
            Error(pos, "Argument of type \"%s\" can't be passed to ISPC intrinsic \"%s\".", type->GetString().c_str(),
                  name.c_str());
            return nullptr;
        }
        TYs.push_back(llvmType);
    }
    auto isPointer = [&TYs](size_t i) { return TYs[i]->isPointerTy(); };
    auto isVector = [&TYs](size_t i) { return llvm::isa<llvm::VectorType>(TYs[i]); };
    auto checkArg = [&](size_t i, bool isValid, const char *expected) {
        return lCheckISPCIntrinsicArg(name, argTypes, i, isValid, expected, pos);
    };
    auto checkSameTypes = [&](size_t i, size_t j) {
        if (TYs[i] != TYs[j]) {
            Error(pos,
                  "Arguments %zu and %zu of ISPC intrinsic \"%s\" must have the same type, \"%s\" and \"%s\" "
                  "provided.",
                  i + 1, j + 1, name.c_str(), argTypes[i]->GetString().c_str(), argTypes[j]->GetString().c_str());
            return false;
        }
        return true;
    };
    const char *pointer = "a uniform pointer";
    const char *vector = "a varying or short vector value";

    bool hasUnnamedType = false;
    switch (ID) {
    case ISPCIntrinsics::not_intrinsic:
        Error(pos, "Not ISPC intrinsic.");
        return nullptr;
    case ISPCIntrinsics::atomicrmw: {
        if (!lCheckISPCIntrinsicArgCount(name, argTypes, 2, pos) || !checkArg(0, isPointer(0), pointer)) {
            return nullptr;
        }
        retType = TYs[1];
        name += "." + lGetMangledTypeStr(TYs[1], hasUnnamedType);
        break;
    }
    case ISPCIntrinsics::bitcast: {
        if (!lCheckISPCIntrinsicArgCount(name, argTypes, 2, pos)) {
            return nullptr;
        }
        llvm::TypeSize size0 = TYs[0]->getPrimitiveSizeInBits();
        llvm::TypeSize size1 = TYs[1]->getPrimitiveSizeInBits();
        if (size0 == 0 || size0 != size1) {
            Error(pos,
                  "Arguments of ISPC intrinsic \"%s\" must be values of the same size in bits, \"%s\" and \"%s\" "
                  "provided.",
                  name.c_str(), argTypes[0]->GetString().c_str(), argTypes[1]->GetString().c_str());
            return nullptr;
        }
        retType = TYs[1];
        name += "." + lGetMangledTypeStr(TYs[0], hasUnnamedType) + "." + lGetMangledTypeStr(TYs[1], hasUnnamedType);
        break;
    }
    case ISPCIntrinsics::blend_store: {
        if (!lCheckISPCIntrinsicArgCount(name, argTypes, 3, pos) || !checkArg(1, isPointer(1), pointer)) {
            return nullptr;
        }
        retType = llvm::Type::getVoidTy(*g->ctx);
        name += "." + lGetMangledTypeStr(TYs[0], hasUnnamedType);
        break;
    }
    case ISPCIntrinsics::concat: {
        if (!lCheckISPCIntrinsicArgCount(name, argTypes, 2, pos) || !checkArg(0, isVector(0), vector) ||
            !checkSameTypes(0, 1)) {
            return nullptr;
        }
        llvm::VectorType *vt = llvm::cast<llvm::VectorType>(TYs[0]);
        retType = llvm::VectorType::get(vt->getElementType(), vt->getElementCount().getKnownMinValue() * 2, false);
        name += "." + lGetMangledTypeStr(retType, hasUnnamedType) + "." + lGetMangledTypeStr(TYs[1], hasUnnamedType);
        break;
    }
    case ISPCIntrinsics::cmpxchg: {
        if (!lCheckISPCIntrinsicArgCount(name, argTypes, 3, pos) || !checkArg(0, isPointer(0), pointer) ||
            !checkSameTypes(1, 2)) {
            return nullptr;
        }
        retType = TYs[1];
        name += "." + lGetMangledTypeStr(TYs[1], hasUnnamedType);
        break;
    }
    case ISPCIntrinsics::extract: {
        if (argTypes.size() != 2 && argTypes.size() != 3) {
            Error(pos, "ISPC intrinsic \"%s\" expects 2 or 3 arguments, %zu provided.", name.c_str(), argTypes.size());
            return nullptr;
        }
        if (!checkArg(0, isVector(0), vector) || (argTypes.size() == 3 && !checkSameTypes(0, 1))) {
            return nullptr;
        }
        retType = llvm::cast<llvm::VectorType>(TYs[0])->getElementType();
        name += "." + lGetMangledTypeStr(TYs[0], hasUnnamedType);
        if (argTypes.size() == 3) {
            name += "." + lGetMangledTypeStr(TYs[1], hasUnnamedType);
        }
        break;
    }
    case ISPCIntrinsics::fence: {
        if (!lCheckISPCIntrinsicArgCount(name, argTypes, 0, pos)) {
            return nullptr;
        }
        retType = llvm::Type::getVoidTy(*g->ctx);
        break;
    }
    case ISPCIntrinsics::insert: {
        if (!lCheckISPCIntrinsicArgCount(name, argTypes, 3, pos) || !checkArg(0, isVector(0), vector)) {
            return nullptr;
        }
        retType = TYs[0];
        name += "." + lGetMangledTypeStr(TYs[0], hasUnnamedType);
        break;
    }
    case ISPCIntrinsics::packmask: {
        if (!lCheckISPCIntrinsicArgCount(name, argTypes, 1, pos) || !checkArg(0, isVector(0), vector)) {
            return nullptr;
        }
        retType = llvm::Type::getInt64Ty(*g->ctx);
        name += "." + lGetMangledTypeStr(TYs[0], hasUnnamedType);
        break;
    }
    case ISPCIntrinsics::select: {
        if (!lCheckISPCIntrinsicArgCount(name, argTypes, 3, pos) || !checkSameTypes(1, 2)) {
            return nullptr;
        }
        retType = TYs[1];
        name += "." + lGetMangledTypeStr(TYs[1], hasUnnamedType);
        break;
    }
    case ISPCIntrinsics::stream_load: {
        if (!lCheckISPCIntrinsicArgCount(name, argTypes, 2, pos) || !checkArg(0, isPointer(0), pointer)) {
            return nullptr;
        }
        retType = TYs[1];
        name += "." + lGetMangledTypeStr(TYs[1], hasUnnamedType);
        break;
    }
    case ISPCIntrinsics::stream_store: {
        if (!lCheckISPCIntrinsicArgCount(name, argTypes, 2, pos) || !checkArg(0, isPointer(0), pointer)) {
            return nullptr;
        }
        retType = llvm::Type::getVoidTy(*g->ctx);
        name += "." + lGetMangledTypeStr(TYs[1], hasUnnamedType);
        break;
    }
    default: {
        Error(pos, "Unknown ISPC intrinsic ID \"%u\".", (unsigned)ID);
        return nullptr;
    }
    }
    llvm::FunctionType *funcType = llvm::FunctionType::get(retType, TYs, false);
    return llvm::cast<llvm::Function>(M->getOrInsertFunction(name, funcType).getCallee());
}

static bool lIsAnyArgumentKind(const IITDesc &D) {
#if ISPC_LLVM_VERSION >= ISPC_LLVM_23_0
    return D.Kind == IITDesc::Overloaded;
#else
    if (D.Kind != IITDesc::Argument) {
        return false;
    }

    unsigned Kind = D.getArgumentKind();
    return Kind == IITDesc::AK_Any || Kind == IITDesc::AK_AnyInteger || Kind == IITDesc::AK_AnyFloat ||
           Kind == IITDesc::AK_AnyVector || Kind == IITDesc::AK_AnyPointer;
#endif
}

// Returns the overload index of the overloaded type that D introduces, or -1
// if D doesn't introduce an overloaded type.
static int lGetDefinedOverloadIndex(const IITDesc &D) {
#if ISPC_LLVM_VERSION >= ISPC_LLVM_23_0
    if (lIsAnyArgumentKind(D) || D.Kind == IITDesc::VecOfAnyPtrsToElt) {
        return D.getOverloadIndex();
    }
#else
    if (lIsAnyArgumentKind(D)) {
        return D.getArgumentNumber();
    }
    if (D.Kind == IITDesc::VecOfAnyPtrsToElt) {
        return D.getOverloadArgNumber();
    }
#endif
    return -1;
}

// Returns the overload index of the overloaded type that D is the same type
// as (LLVMMatchType), or -1.
static int lGetMatchedOverloadIndex(const IITDesc &D) {
#if ISPC_LLVM_VERSION >= ISPC_LLVM_23_0
    if (D.Kind == IITDesc::Match) {
        return D.getOverloadIndex();
    }
#else
    if (D.Kind == IITDesc::Argument && D.getArgumentKind() == IITDesc::AK_MatchType) {
        return D.getArgumentNumber();
    }
#endif
    return -1;
}

// LLVM computes the types derived from an overloaded type (for example, the
// element type of an overloaded vector type) without checking that the
// overloaded type has the expected kind, and fails with llvm_unreachable or
// an assertion if it doesn't. Returns the overload index of the overloaded
// type that D can't be derived from, or -1 if D can be computed. Extended and
// truncated types are conservatively accepted only for integer types.
static int lCheckDerivedType(const IITDesc &D, const std::vector<llvm::Type *> &overloadTys) {
    unsigned index = 0;
    bool needVector = false;
    bool needInteger = false;
    unsigned divisor = 1;
    switch (D.Kind) {
#if ISPC_LLVM_VERSION >= ISPC_LLVM_23_0
    case IITDesc::VecElement:
    case IITDesc::VecOfBitcastsToInt:
        index = D.getOverloadIndex();
        needVector = true;
        break;
    case IITDesc::OneNthEltsVec:
        index = D.getOverloadIndex();
        needVector = true;
        divisor = D.getVectorDivisor();
        break;
    case IITDesc::Subdivide2:
    case IITDesc::Subdivide4:
        index = D.getOverloadIndex();
        needVector = true;
        needInteger = true;
        break;
    case IITDesc::Extend:
    case IITDesc::Trunc:
        index = D.getOverloadIndex();
        needInteger = true;
        break;
#else
    case IITDesc::VecElementArgument:
    case IITDesc::VecOfBitcastsToInt:
        index = D.getArgumentNumber();
        needVector = true;
        break;
#if ISPC_LLVM_VERSION >= ISPC_LLVM_21_0
    case IITDesc::OneNthEltsVecArgument:
        index = D.getRefArgNumber();
        needVector = true;
        divisor = D.getVectorDivisor();
        break;
#else
    case IITDesc::HalfVecArgument:
        index = D.getArgumentNumber();
        needVector = true;
        divisor = 2;
        break;
#endif
    case IITDesc::Subdivide2Argument:
    case IITDesc::Subdivide4Argument:
        index = D.getArgumentNumber();
        needVector = true;
        needInteger = true;
        break;
    case IITDesc::ExtendArgument:
    case IITDesc::TruncArgument:
        index = D.getArgumentNumber();
        needInteger = true;
        break;
#endif
    default:
        return -1;
    }
    if (index >= overloadTys.size()) {
        return index;
    }
    llvm::Type *type = overloadTys[index];
    llvm::VectorType *vt = llvm::dyn_cast<llvm::VectorType>(type);
    if (needVector && (vt == nullptr || (divisor > 1 && vt->getElementCount().getKnownMinValue() % divisor != 0))) {
        return index;
    }
    if (needInteger && (!type->isIntOrIntVectorTy() || type->getScalarSizeInBits() < 8)) {
        return index;
    }
    return -1;
}

// Returns the index of the IIT descriptor that follows the type that starts
// at Table[i].
static size_t lSkipIITType(llvm::ArrayRef<IITDesc> Table, size_t i) {
    if (i >= Table.size()) {
        return i;
    }
    const IITDesc &D = Table[i++];
    switch (D.Kind) {
    case IITDesc::Vector:
#if ISPC_LLVM_VERSION >= ISPC_LLVM_23_0
    case IITDesc::SameVecWidth:
#else
    case IITDesc::SameVecWidthArgument:
#endif
        // The element type follows.
        return lSkipIITType(Table, i);
    case IITDesc::Struct: {
#if ISPC_LLVM_VERSION >= ISPC_LLVM_23_0
        unsigned numElements = D.StructNumElements;
#else
        unsigned numElements = D.Struct_NumElements;
#endif
        for (unsigned j = 0; j < numElements; ++j) {
            i = lSkipIITType(Table, i);
        }
        return i;
    }
    default:
        return i;
    }
}

static std::string lGetArgTypesString(ExprList *args) {
    std::string result;
    if (args != nullptr) {
        for (size_t i = 0; i < args->exprs.size(); ++i) {
            result += (i > 0 ? ", " : "") + args->exprs[i]->GetType()->GetString();
        }
    }
    return result;
}

// Declare the LLVM intrinsic ID called as "name" with the given arguments.
//
// The types that an overloaded intrinsic is overloaded on (for example, the
// type of x in llvm.ctpop(x)) are deduced from the arguments. LLVM numbers
// these types in the order in which they appear in the intrinsic info table,
// and each of them has to be the type of an argument or of an argument
// declared as LLVMMatchType of it. The type suffix in the name is ignored.
static llvm::Function *lGetLLVMIntrinsicDeclaration(llvm::Module *module, llvm::Intrinsic::ID ID,
                                                    const std::string &name, ExprList *args, SourcePos pos) {
    size_t numArgs = args ? args->exprs.size() : 0;

    llvm::SmallVector<IITDesc, 8> TableStorage;
#if ISPC_LLVM_VERSION >= ISPC_LLVM_23_0
    auto [Table, numTableParams, isVarArg] = llvm::Intrinsic::getIntrinsicInfoTableEntries(ID, TableStorage);
    (void)numTableParams;
#else
    llvm::Intrinsic::getIntrinsicInfoTableEntries(ID, TableStorage);
    llvm::ArrayRef<IITDesc> Table(TableStorage);
    bool isVarArg = false;
#endif

    // The first type in the table is the return type, the others are the
    // types of the parameters.
    std::vector<size_t> typeStarts;
    for (size_t i = 0; i < Table.size(); i = lSkipIITType(Table, i)) {
        if (Table[i].Kind == IITDesc::VarArg) {
            isVarArg = true;
            break;
        }
        typeStarts.push_back(i);
    }
    if (isVarArg || typeStarts.empty()) {
        Error(pos, "LLVM intrinsic \"%s\" with a variable number of arguments is not supported.", name.c_str());
        return nullptr;
    }
    size_t numParams = typeStarts.size() - 1;
    if (numArgs != numParams) {
        Error(pos, "LLVM intrinsic \"%s\" expects %zu argument%s, %zu provided.", name.c_str(), numParams,
              numParams == 1 ? "" : "s", numArgs);
        return nullptr;
    }

    int numOverloads = 0;
    for (const IITDesc &D : Table) {
        numOverloads = std::max(numOverloads, lGetDefinedOverloadIndex(D) + 1);
    }
    std::vector<llvm::Type *> overloadTys(numOverloads, nullptr);
    std::vector<size_t> overloadArgs(numOverloads, 0);
    for (size_t i = 0; i < numParams; ++i) {
        const IITDesc &D = Table[typeStarts[i + 1]];
        int index = lGetDefinedOverloadIndex(D);
        if (index < 0) {
            index = lGetMatchedOverloadIndex(D);
        }
        if (index < 0 || index >= numOverloads || overloadTys[index] != nullptr) {
            continue;
        }
        const Type *argType = args->exprs[i]->GetType();
        if ((ID == llvm::Intrinsic::masked_gather || ID == llvm::Intrinsic::masked_scatter) &&
            D.Kind == IITDesc::VecOfAnyPtrsToElt) {
            // The addresses are passed as varying integers and converted to
            // <TARGET_WIDTH x ptr> in FunctionCallExpr::GetValue().
            overloadTys[index] = LLVMTypes::PtrVectorType;
        } else {
            overloadTys[index] = argType->LLVMType(g->ctx);
        }
        overloadArgs[index] = i;
        if (overloadTys[index] == nullptr || overloadTys[index]->isVoidTy()) {
            Error(pos, "Argument %zu of type \"%s\" can't be passed to LLVM intrinsic \"%s\".", i + 1,
                  argType->GetString().c_str(), name.c_str());
            return nullptr;
        }
    }
    for (llvm::Type *type : overloadTys) {
        if (type == nullptr) {
            Error(pos, "Can't deduce the overloaded types of LLVM intrinsic \"%s\" from its arguments.", name.c_str());
            return nullptr;
        }
    }
    for (const IITDesc &D : Table) {
        int index = lCheckDerivedType(D, overloadTys);
        if (index >= numOverloads) {
            Error(pos, "Can't deduce the overloaded types of LLVM intrinsic \"%s\" from its arguments.", name.c_str());
            return nullptr;
        } else if (index >= 0) {
            size_t argIndex = overloadArgs[index];
            Error(pos, "Type \"%s\" of argument %zu is not valid for LLVM intrinsic \"%s\".",
                  args->exprs[argIndex]->GetType()->GetString().c_str(), argIndex + 1, name.c_str());
            return nullptr;
        }
    }

    llvm::FunctionType *funcType = llvm::Intrinsic::getType(*g->ctx, ID, overloadTys);
    llvm::SmallVector<llvm::Type *, 4> deducedTys;
#if ISPC_LLVM_VERSION >= ISPC_LLVM_23_0
    bool isValid = llvm::Intrinsic::isSignatureValid(ID, funcType, deducedTys);
#else
    bool isValid = llvm::Intrinsic::getIntrinsicSignature(ID, funcType, deducedTys);
#endif
    if (!isValid) {
        Error(pos, "Arguments of types (%s) are not valid for LLVM intrinsic \"%s\".", lGetArgTypesString(args).c_str(),
              name.c_str());
        return nullptr;
    }

    return llvm::Intrinsic::getOrInsertDeclaration(module, ID, overloadTys);
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
    int nInits = args ? args->exprs.size() : 0;
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
    llvm::Intrinsic::ID ID = llvm::Intrinsic::lookupIntrinsicID(name);
    if (ID != llvm::Intrinsic::not_intrinsic) {
        if (!g->target->checkIntrinsticSupport(name, pos)) {
            return nullptr;
        }
        return lGetLLVMIntrinsicDeclaration(module, ID, name, args, pos);
    }

    // Check if it is an ISPC intrinsic
    ISPCIntrinsics IID = lLookupISPCInstrinsic(name);
    if (IID == ISPCIntrinsics::not_intrinsic) {
        Error(pos, "LLVM intrinsic \"%s\" not supported.", name.c_str());
        return nullptr;
    }
    std::vector<const Type *> argTypes;
    if (args) {
        for (const Expr *arg : args->exprs) {
            argTypes.push_back(arg->GetType());
        }
    }
    return lGetISPCIntrinsicsFuncDecl(module, name, IID, argTypes, pos);
}

bool Module::CheckLLVMIntrinsicName(const std::string &name, SourcePos pos) {
#ifdef ISPC_XE_ENABLED
    if (g->target->isXeTarget()) {
        if (llvm::GenXIntrinsic::lookupGenXIntrinsicID(name) == llvm::GenXIntrinsic::not_any_intrinsic) {
            Error(pos, "LLVM intrinsic \"%s\" not supported.", name.c_str());
            return false;
        }
        return true;
    }
#endif // ISPC_XE_ENABLED
    if (llvm::Intrinsic::lookupIntrinsicID(name) == llvm::Intrinsic::not_intrinsic &&
        lLookupISPCInstrinsic(name) == ISPCIntrinsics::not_intrinsic) {
        Error(pos, "LLVM intrinsic \"%s\" not supported.", name.c_str());
        return false;
    }
    return true;
}

Symbol *Module::AddLLVMIntrinsicDecl(const std::string &name, ExprList *args, SourcePos pos) {
    if (args != nullptr) {
        for (const Expr *arg : args->exprs) {
            // An error has already been reported for the argument, and the
            // intrinsic can't be declared without its type.
            if (arg == nullptr || arg->GetType() == nullptr) {
                return nullptr;
            }
        }
    }

    llvm::Function *funcDecl = lGetIntrinsicDeclaration(module, name, args, pos);
    if (funcDecl == nullptr) {
        return nullptr;
    }

    return lCreateISPCSymbolForLLVMIntrinsic(funcDecl, symbolTable, name, pos);
}
