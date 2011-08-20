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

/** @file builtins.cpp
    @brief Definitions of functions related to setting up the standard library 
           and other builtins.
*/

#include "builtins.h"
#include "type.h"
#include "util.h"
#include "sym.h"
#include "expr.h"
#include "llvmutil.h"
#include "module.h"
#include "ctx.h"

#include <math.h>
#include <stdlib.h>
#include <llvm/LLVMContext.h>
#include <llvm/Module.h>
#include <llvm/Type.h>
#include <llvm/DerivedTypes.h>
#include <llvm/Instructions.h>
#include <llvm/Intrinsics.h>
#include <llvm/Linker.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/ADT/Triple.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Bitcode/ReaderWriter.h>

extern int yyparse();
struct yy_buffer_state;
extern yy_buffer_state *yy_scan_string(const char *);


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
static const Type *
lLLVMTypeToISPCType(const llvm::Type *t, bool intAsUnsigned) {
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
    else if (t == LLVMTypes::FloatType)
        return AtomicType::UniformFloat;
    else if (t == LLVMTypes::DoubleType)
        return AtomicType::UniformDouble;
    else if (t == LLVMTypes::Int64Type)
        return intAsUnsigned ? AtomicType::UniformUInt64 : AtomicType::UniformInt64;

    // varying
    else if (t == LLVMTypes::Int8VectorType)
        return intAsUnsigned ? AtomicType::VaryingUInt8 : AtomicType::VaryingInt8;
    else if (t == LLVMTypes::Int16VectorType)
        return intAsUnsigned ? AtomicType::VaryingUInt16 : AtomicType::VaryingInt16;
    else if (t == LLVMTypes::Int32VectorType)
        return intAsUnsigned ? AtomicType::VaryingUInt32 : AtomicType::VaryingInt32;
    else if (t == LLVMTypes::FloatVectorType)
        return AtomicType::VaryingFloat;
    else if (t == LLVMTypes::DoubleVectorType)
        return AtomicType::VaryingDouble;
    else if (t == LLVMTypes::Int64VectorType)
        return intAsUnsigned ? AtomicType::VaryingUInt64 : AtomicType::VaryingInt64;

    // pointers to uniform
    else if (t == LLVMTypes::Int8PointerType)
        return new ReferenceType(intAsUnsigned ? AtomicType::UniformUInt8 :
                                                 AtomicType::UniformInt8, false);
    else if (t == LLVMTypes::Int16PointerType)
        return new ReferenceType(intAsUnsigned ? AtomicType::UniformUInt16 :
                                                 AtomicType::UniformInt16, false);
    else if (t == LLVMTypes::Int32PointerType)
        return new ReferenceType(intAsUnsigned ? AtomicType::UniformUInt32 :
                                                 AtomicType::UniformInt32, false);
    else if (t == LLVMTypes::Int64PointerType)
        return new ReferenceType(intAsUnsigned ? AtomicType::UniformUInt64 :
                                                 AtomicType::UniformInt64, false);
    else if (t == LLVMTypes::FloatPointerType)
        return new ReferenceType(AtomicType::UniformFloat, false);
    else if (t == LLVMTypes::DoublePointerType)
        return new ReferenceType(AtomicType::UniformDouble, false);

    // pointers to varying
    else if (t == LLVMTypes::Int8VectorPointerType)
        return new ReferenceType(intAsUnsigned ? AtomicType::VaryingUInt8 :
                                                 AtomicType::VaryingInt8, false);
    else if (t == LLVMTypes::Int16VectorPointerType)
        return new ReferenceType(intAsUnsigned ? AtomicType::VaryingUInt16 :
                                                 AtomicType::VaryingInt16, false);
    else if (t == LLVMTypes::Int32VectorPointerType)
        return new ReferenceType(intAsUnsigned ? AtomicType::VaryingUInt32 :
                                                 AtomicType::VaryingInt32, false);
    else if (t == LLVMTypes::Int64VectorPointerType)
        return new ReferenceType(intAsUnsigned ? AtomicType::VaryingUInt64 :
                                                 AtomicType::VaryingInt64, false);
    else if (t == LLVMTypes::FloatVectorPointerType)
        return new ReferenceType(AtomicType::VaryingFloat, false);
    else if (t == LLVMTypes::DoubleVectorPointerType)
        return new ReferenceType(AtomicType::VaryingDouble, false);

    // arrays
    else if (llvm::isa<const llvm::PointerType>(t)) {
        const llvm::PointerType *pt = llvm::dyn_cast<const llvm::PointerType>(t);

        // Is it a pointer to an unsized array of objects?  If so, then
        // create the equivalent ispc type.  Note that it has to be a
        // reference to an array, since ispc passes arrays to functions by
        // reference.
        const llvm::ArrayType *at = 
            llvm::dyn_cast<const llvm::ArrayType>(pt->getElementType());
        if (at != NULL) {
            const Type *eltType = lLLVMTypeToISPCType(at->getElementType(),
                                                      intAsUnsigned);
            if (eltType == NULL)
                return NULL;
            return new ReferenceType(new ArrayType(eltType, at->getNumElements()),
                                     false);
        }
    }

    return NULL;
}


static void
lCreateSymbol(const std::string &name, const Type *returnType, 
              const std::vector<const Type *> &argTypes, 
              const llvm::FunctionType *ftype, llvm::Function *func, 
              SymbolTable *symbolTable) {
    SourcePos noPos;
    noPos.name = "__stdlib";

    FunctionType *funcType = new FunctionType(returnType, argTypes, noPos);
    // set NULL default arguments
    std::vector<ConstExpr *> defaults;
    for (unsigned int j = 0; j < ftype->getNumParams(); ++j)
        defaults.push_back(NULL);
    funcType->SetArgumentDefaults(defaults);

    Symbol *sym = new Symbol(name, noPos, funcType);
    sym->function = func;
    symbolTable->AddFunction(sym);
}


/** Given an LLVM function declaration, synthesize the equivalent ispc
    symbol for the function (if possible).  Returns true on success, false
    on failure.
 */
static bool
lCreateISPCSymbol(llvm::Function *func, SymbolTable *symbolTable) {
    SourcePos noPos;
    noPos.name = "__stdlib";

    const llvm::FunctionType *ftype = func->getFunctionType();
    std::string name = func->getName();

    if (name.size() < 3 || name[0] != '_' || name[1] != '_')
        return false;

    // An unfortunate hack: we want this builtin function to have the
    // signature "int __sext_varying_bool(bool)", but the ispc function
    // symbol creation code below assumes that any LLVM vector of i32s is a
    // varying int32.  Here, we need that to be interpreted as a varying
    // bool, so just have a one-off override for that one...
    if (name == "__sext_varying_bool") {
        const Type *returnType = AtomicType::VaryingInt32;
        std::vector<const Type *> argTypes;
        argTypes.push_back(AtomicType::VaryingBool);
        std::vector<ConstExpr *> defaults;
        defaults.push_back(NULL);

        FunctionType *funcType = new FunctionType(returnType, argTypes, noPos);
        funcType->SetArgumentDefaults(defaults);

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

        const Type *returnType = lLLVMTypeToISPCType(ftype->getReturnType(),
                                                     intAsUnsigned);
        if (!returnType)
            // return type not representable in ispc -> not callable from ispc
            return false;

        // Iterate over the arguments and try to find their equivalent ispc
        // types.  Track if any of the arguments has an integer type.
        bool anyIntArgs = false, anyReferenceArgs = false;
        std::vector<const Type *> argTypes;
        for (unsigned int j = 0; j < ftype->getNumParams(); ++j) {
            const llvm::Type *llvmArgType = ftype->getParamType(j);
            const Type *type = lLLVMTypeToISPCType(llvmArgType, intAsUnsigned);
            if (type == NULL)
                return false;
            anyIntArgs |= 
                (Type::Equal(type, lLLVMTypeToISPCType(llvmArgType, !intAsUnsigned)) == false);
            anyReferenceArgs |= (dynamic_cast<const ReferenceType *>(type) != NULL);
            argTypes.push_back(type);
        }

        // Always create the symbol the first time through, in particular
        // so that we get symbols for things with no integer types!
        if (i == 0 || anyIntArgs == true)
            lCreateSymbol(name, returnType, argTypes, ftype, func, symbolTable);

        // If there are any reference types, also make a variant of the
        // symbol that has them as const references.  This obviously
        // doesn't make sense for many builtins, but we'll give the stdlib
        // the option to call one if it needs one.
        if (anyReferenceArgs == true) {
            for (unsigned int j = 0; j < argTypes.size(); ++j) {
                if (dynamic_cast<const ReferenceType *>(argTypes[j]) != NULL)
                    argTypes[j] = argTypes[j]->GetAsConstType();
                lCreateSymbol(name + "_refsconst", returnType, argTypes, 
                              ftype, func, symbolTable);
            }
        }
    }

    return true;
}


/** Given an LLVM module, create ispc symbols for the functions in the
    module.
 */
static void
lAddModuleSymbols(llvm::Module *module, SymbolTable *symbolTable) {
#if 0
    // FIXME: handle globals?
    assert(module->global_empty());
#endif

    llvm::Module::iterator iter;
    for (iter = module->begin(); iter != module->end(); ++iter) {
        llvm::Function *func = iter;
        lCreateISPCSymbol(func, symbolTable);
    }
}


/** In many of the builtins-*.ll files, we have declarations of various LLVM
    intrinsics that are then used in the implementation of various target-
    specific functions.  This function loops over all of the intrinsic 
    declarations and makes sure that the signature we have in our .ll file
    matches the signature of the actual intrinsic.
*/
static void
lCheckModuleIntrinsics(llvm::Module *module) {
    llvm::Module::iterator iter;
    for (iter = module->begin(); iter != module->end(); ++iter) {
        llvm::Function *func = iter;
        if (!func->isIntrinsic())
            continue;

        const std::string funcName = func->getName().str();
        // Work around http://llvm.org/bugs/show_bug.cgi?id=10438; only
        // check the llvm.x86.* intrinsics for now...
        if (!strncmp(funcName.c_str(), "llvm.x86.", 9)) {
            llvm::Intrinsic::ID id = (llvm::Intrinsic::ID)func->getIntrinsicID();
            assert(id != 0);
            LLVM_TYPE_CONST llvm::Type *intrinsicType = 
                llvm::Intrinsic::getType(*g->ctx, id);
            intrinsicType = llvm::PointerType::get(intrinsicType, 0);
            assert(func->getType() == intrinsicType);
        }
    }
}


/** This utility function takes serialized binary LLVM bitcode and adds its
    definitions to the given module.  Functions in the bitcode that can be
    mapped to ispc functions are also added to the symbol table.

    @param bitcode     Binary LLVM bitcode (e.g. the contents of a *.bc file)
    @param length      Length of the bitcode buffer
    @param module      Module to link the bitcode into
    @param symbolTable Symbol table to add definitions to
 */
static void
lAddBitcode(const unsigned char *bitcode, int length,
            llvm::Module *module, SymbolTable *symbolTable) {
    std::string bcErr;
    llvm::StringRef sb = llvm::StringRef((char *)bitcode, length);
    llvm::MemoryBuffer *bcBuf = llvm::MemoryBuffer::getMemBuffer(sb);
    llvm::Module *bcModule = llvm::ParseBitcodeFile(bcBuf, *g->ctx, &bcErr);
    if (!bcModule)
        Error(SourcePos(), "Error parsing stdlib bitcode: %s", bcErr.c_str());
    else {
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
        assert(bcTriple.getArch() == llvm::Triple::UnknownArch ||
               mTriple.getArch() == bcTriple.getArch());
        assert(bcTriple.getVendor() == llvm::Triple::UnknownVendor ||
               mTriple.getVendor() == bcTriple.getVendor());
        bcModule->setTargetTriple(mTriple.str());

        std::string(linkError);
        if (llvm::Linker::LinkModules(module, bcModule, &linkError))
            Error(SourcePos(), "Error linking stdlib bitcode: %s", linkError.c_str());
        lAddModuleSymbols(module, symbolTable);
        lCheckModuleIntrinsics(module);
    }
}


/** Utility routine that defines a constant int32 with given value, adding
    the symbol to both the ispc symbol table and the given LLVM module.
 */
static void
lDefineConstantInt(const char *name, int val, llvm::Module *module,
                   SymbolTable *symbolTable) {
    Symbol *pw = new Symbol(name, SourcePos(), AtomicType::UniformConstInt32);
    pw->isStatic = true;
    pw->constValue = new ConstExpr(pw->type, val, SourcePos());
    LLVM_TYPE_CONST llvm::Type *ltype = LLVMTypes::Int32Type;
    llvm::Constant *linit = LLVMInt32(val);
    pw->storagePtr = new llvm::GlobalVariable(*module, ltype, true, 
                                              llvm::GlobalValue::InternalLinkage,
                                              linit, pw->name.c_str());
    symbolTable->AddVariable(pw);
}


static void
lDefineProgramIndex(llvm::Module *module, SymbolTable *symbolTable) {
    Symbol *pidx = new Symbol("programIndex", SourcePos(), 
                              AtomicType::VaryingConstInt32);
    pidx->isStatic = true;

    int pi[ISPC_MAX_NVEC];
    for (int i = 0; i < g->target.vectorWidth; ++i)
        pi[i] = i;
    pidx->constValue = new ConstExpr(pidx->type, pi, SourcePos());

    LLVM_TYPE_CONST llvm::Type *ltype = LLVMTypes::Int32VectorType;
    llvm::Constant *linit = LLVMInt32Vector(pi);
    pidx->storagePtr = new llvm::GlobalVariable(*module, ltype, true, 
                                                llvm::GlobalValue::InternalLinkage, linit, 
                                                pidx->name.c_str());
    symbolTable->AddVariable(pidx);
}


void
DefineStdlib(SymbolTable *symbolTable, llvm::LLVMContext *ctx, llvm::Module *module,
             bool includeStdlibISPC) {
    // Add the definitions from the compiled builtins-c.c file
    if (g->target.is32bit) {
        extern unsigned char builtins_bitcode_c_32[];
        extern int builtins_bitcode_c_32_length;
        lAddBitcode(builtins_bitcode_c_32, builtins_bitcode_c_32_length, 
                    module, symbolTable);
    }
    else {
        extern unsigned char builtins_bitcode_c_64[];
        extern int builtins_bitcode_c_64_length;
        lAddBitcode(builtins_bitcode_c_64, builtins_bitcode_c_64_length, 
                    module, symbolTable);
    }

    // Next, add the target's custom implementations of the various needed
    // builtin functions (e.g. __masked_store_32(), etc).
    switch (g->target.isa) {
    case Target::SSE2:
        extern unsigned char builtins_bitcode_sse2[];
        extern int builtins_bitcode_sse2_length;
        lAddBitcode(builtins_bitcode_sse2, builtins_bitcode_sse2_length, module,
                    symbolTable);
        break;
    case Target::SSE4:
        extern unsigned char builtins_bitcode_sse4[];
        extern int builtins_bitcode_sse4_length;
        extern unsigned char builtins_bitcode_sse4x2[];
        extern int builtins_bitcode_sse4x2_length;
        switch (g->target.vectorWidth) {
        case 4: 
            lAddBitcode(builtins_bitcode_sse4, builtins_bitcode_sse4_length, 
                        module, symbolTable);
            break;
        case 8:
            lAddBitcode(builtins_bitcode_sse4x2, builtins_bitcode_sse4x2_length, 
                        module, symbolTable);
            break;
        default:
            FATAL("logic error in DefineStdlib");
        }
        break;
    case Target::AVX:
        switch (g->target.vectorWidth) {
        case 8:
            extern unsigned char builtins_bitcode_avx[];
            extern int builtins_bitcode_avx_length;
            lAddBitcode(builtins_bitcode_avx, builtins_bitcode_avx_length, module, 
                        symbolTable);
            break;
        case 16:
            extern unsigned char builtins_bitcode_avx_x2[];
            extern int builtins_bitcode_avx_x2_length;
            lAddBitcode(builtins_bitcode_avx_x2, builtins_bitcode_avx_x2_length,
                        module,  symbolTable);
            break;
        default:
            FATAL("logic error in DefineStdlib");
        }
        break;
    default:
        FATAL("logic error");
    }

    // define the 'programCount' builtin variable
    lDefineConstantInt("programCount", g->target.vectorWidth, module, symbolTable);

    // define the 'programIndex' builtin
    lDefineProgramIndex(module, symbolTable);

    // Define __math_lib stuff.  This is used by stdlib.ispc, for example, to
    // figure out which math routines to end up calling...
    lDefineConstantInt("__math_lib", (int)g->mathLib, module, symbolTable);
    lDefineConstantInt("__math_lib_ispc", (int)Globals::Math_ISPC, module,
                       symbolTable);
    lDefineConstantInt("__math_lib_ispc_fast", (int)Globals::Math_ISPCFast, 
                       module, symbolTable);
    lDefineConstantInt("__math_lib_svml", (int)Globals::Math_SVML, module,
                       symbolTable);
    lDefineConstantInt("__math_lib_system", (int)Globals::Math_System, module,
                       symbolTable);

    if (includeStdlibISPC) {
        // If the user wants the standard library to be included, parse the
        // serialized version of the stdlib.ispc file to get its
        // definitions added.  Disable emission of performance warnings for
        // now, since the user doesn't care about any of that in the stdlib
        // implementation...
        bool epf = g->emitPerfWarnings;
        g->emitPerfWarnings = false;
        extern char stdlib_code[];
        yy_scan_string(stdlib_code);
        yyparse();
        g->emitPerfWarnings = epf;
    }
}
