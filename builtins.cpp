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
    if (LLVMTypes::MaskType != LLVMTypes::Int32VectorType &&
        t == LLVMTypes::MaskType)
        return AtomicType::VaryingBool;
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
        return PointerType::GetUniform(intAsUnsigned ? AtomicType::UniformUInt8 :
                                       AtomicType::UniformInt8);
    else if (t == LLVMTypes::Int16PointerType)
        return PointerType::GetUniform(intAsUnsigned ? AtomicType::UniformUInt16 :
                                       AtomicType::UniformInt16);
    else if (t == LLVMTypes::Int32PointerType)
        return PointerType::GetUniform(intAsUnsigned ? AtomicType::UniformUInt32 :
                                       AtomicType::UniformInt32);
    else if (t == LLVMTypes::Int64PointerType)
        return PointerType::GetUniform(intAsUnsigned ? AtomicType::UniformUInt64 :
                                       AtomicType::UniformInt64);
    else if (t == LLVMTypes::FloatPointerType)
        return PointerType::GetUniform(AtomicType::UniformFloat);
    else if (t == LLVMTypes::DoublePointerType)
        return PointerType::GetUniform(AtomicType::UniformDouble);

    // pointers to varying
    else if (t == LLVMTypes::Int8VectorPointerType)
        return PointerType::GetUniform(intAsUnsigned ? AtomicType::VaryingUInt8 :
                                       AtomicType::VaryingInt8);
    else if (t == LLVMTypes::Int16VectorPointerType)
        return PointerType::GetUniform(intAsUnsigned ? AtomicType::VaryingUInt16 :
                                       AtomicType::VaryingInt16);
    else if (t == LLVMTypes::Int32VectorPointerType)
        return PointerType::GetUniform(intAsUnsigned ? AtomicType::VaryingUInt32 :
                                       AtomicType::VaryingInt32);
    else if (t == LLVMTypes::Int64VectorPointerType)
        return PointerType::GetUniform(intAsUnsigned ? AtomicType::VaryingUInt64 :
                                       AtomicType::VaryingInt64);
    else if (t == LLVMTypes::FloatVectorPointerType)
        return PointerType::GetUniform(AtomicType::VaryingFloat);
    else if (t == LLVMTypes::DoubleVectorPointerType)
        return PointerType::GetUniform(AtomicType::VaryingDouble);

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

    Debug(noPos, "Created builtin symbol \"%s\" [%s]\n", name.c_str(),
          funcType->GetString().c_str());

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

    Debug(SourcePos(), "Attempting to create ispc symbol for function \"%s\".",
          name.c_str());

    // An unfortunate hack: we want this builtin function to have the
    // signature "int __sext_varying_bool(bool)", but the ispc function
    // symbol creation code below assumes that any LLVM vector of i32s is a
    // varying int32.  Here, we need that to be interpreted as a varying
    // bool, so just have a one-off override for that one...
    if (g->target.maskBitCount != 1 && name == "__sext_varying_bool") {
        const Type *returnType = AtomicType::VaryingInt32;
        std::vector<const Type *> argTypes;
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

        const Type *returnType = lLLVMTypeToISPCType(ftype->getReturnType(),
                                                     intAsUnsigned);
        if (returnType == NULL) {
            Debug(SourcePos(), "Failed: return type not representable for "
                  "builtin %s.", name.c_str());
            // return type not representable in ispc -> not callable from ispc
            return false;
        }

        // Iterate over the arguments and try to find their equivalent ispc
        // types.  Track if any of the arguments has an integer type.
        bool anyIntArgs = false;
        std::vector<const Type *> argTypes;
        for (unsigned int j = 0; j < ftype->getNumParams(); ++j) {
            const llvm::Type *llvmArgType = ftype->getParamType(j);
            const Type *type = lLLVMTypeToISPCType(llvmArgType, intAsUnsigned);
            if (type == NULL) {
                Debug(SourcePos(), "Failed: type of parameter %d not "
                      "representable for builtin %s", j, name.c_str());
                return false;
            }
            anyIntArgs |= 
                (Type::Equal(type, lLLVMTypeToISPCType(llvmArgType, !intAsUnsigned)) == false);
            argTypes.push_back(type);
        }

        // Always create the symbol the first time through, in particular
        // so that we get symbols for things with no integer types!
        if (i == 0 || anyIntArgs == true)
            lCreateSymbol(name, returnType, argTypes, ftype, func, symbolTable);
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
    Assert(module->global_empty());
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
            Assert(id != 0);
            LLVM_TYPE_CONST llvm::Type *intrinsicType = 
                llvm::Intrinsic::getType(*g->ctx, id);
            intrinsicType = llvm::PointerType::get(intrinsicType, 0);
            Assert(func->getType() == intrinsicType);
        }
    }
}


/** We'd like to have all of these functions declared as 'internal' in
    their respective bitcode files so that if they aren't needed by the
    user's program they are elimiated from the final output.  However, if
    we do so, then they aren't brought in by the LinkModules() call below
    since they aren't yet used by anything in the module they're being
    linked with (in LLVM 3.1, at least).

    Therefore, we don't declare them as internal when we first define them,
    but instead mark them as internal after they've been linked in.  This
    is admittedly a kludge.
 */
static void
lSetInternalFunctions(llvm::Module *module) {
    const char *names[] = {
        "__add_float",
        "__add_int32",
        "__add_uniform_double",
        "__add_uniform_int32",
        "__add_uniform_int64",
        "__add_varying_double",
        "__add_varying_int32",
        "__add_varying_int64",
        "__aos_to_soa3_float",
        "__aos_to_soa3_float16",
        "__aos_to_soa3_float4",
        "__aos_to_soa3_float8",
        "__aos_to_soa3_int32",
        "__aos_to_soa4_float",
        "__aos_to_soa4_float16",
        "__aos_to_soa4_float4",
        "__aos_to_soa4_float8",
        "__aos_to_soa4_int32",
        "__atomic_add_int32_global",
        "__atomic_add_int64_global",
        "__atomic_add_uniform_int32_global",
        "__atomic_add_uniform_int64_global",
        "__atomic_and_int32_global",
        "__atomic_and_int64_global",
        "__atomic_and_uniform_int32_global",
        "__atomic_and_uniform_int64_global",
        "__atomic_compare_exchange_double_global",
        "__atomic_compare_exchange_float_global",
        "__atomic_compare_exchange_int32_global",
        "__atomic_compare_exchange_int64_global",
        "__atomic_compare_exchange_uniform_double_global",
        "__atomic_compare_exchange_uniform_float_global",
        "__atomic_compare_exchange_uniform_int32_global",
        "__atomic_compare_exchange_uniform_int64_global",
        "__atomic_max_uniform_int32_global",
        "__atomic_max_uniform_int64_global",
        "__atomic_min_uniform_int32_global",
        "__atomic_min_uniform_int64_global",
        "__atomic_or_int32_global",
        "__atomic_or_int64_global",
        "__atomic_or_uniform_int32_global",
        "__atomic_or_uniform_int64_global",
        "__atomic_sub_int32_global",
        "__atomic_sub_int64_global",
        "__atomic_sub_uniform_int32_global",
        "__atomic_sub_uniform_int64_global",
        "__atomic_swap_double_global",
        "__atomic_swap_float_global",
        "__atomic_swap_int32_global",
        "__atomic_swap_int64_global",
        "__atomic_swap_uniform_double_global",
        "__atomic_swap_uniform_float_global",
        "__atomic_swap_uniform_int32_global",
        "__atomic_swap_uniform_int64_global",
        "__atomic_umax_uniform_uint32_global",
        "__atomic_umax_uniform_uint64_global",
        "__atomic_umin_uniform_uint32_global",
        "__atomic_umin_uniform_uint64_global",
        "__atomic_xor_int32_global",
        "__atomic_xor_int64_global",
        "__atomic_xor_uniform_int32_global",
        "__atomic_xor_uniform_int64_global",
        "__broadcast_double",
        "__broadcast_float",
        "__broadcast_i16",
        "__broadcast_i32",
        "__broadcast_i64",
        "__broadcast_i8",
        "__ceil_uniform_double",
        "__ceil_uniform_float",
        "__ceil_varying_double",
        "__ceil_varying_float",
        "__clock",
        "__count_trailing_zeros_i32",
        "__count_trailing_zeros_i64",
        "__count_leading_zeros_i32",
        "__count_leading_zeros_i64",
        "__delete_uniform",
        "__delete_varying",
        "__do_assert_uniform",
        "__do_assert_varying",
        "__do_print", 
        "__doublebits_uniform_int64",
        "__doublebits_varying_int64",
        "__exclusive_scan_add_double",
        "__exclusive_scan_add_float",
        "__exclusive_scan_add_i32",
        "__exclusive_scan_add_i64",
        "__exclusive_scan_and_i32",
        "__exclusive_scan_and_i64",
        "__exclusive_scan_or_i32",
        "__exclusive_scan_or_i64",
        "__extract_int16",
        "__extract_int32",
        "__extract_int64",
        "__extract_int8",
        "__fastmath",
        "__floatbits_uniform_int32",
        "__floatbits_varying_int32",
        "__floor_uniform_double",
        "__floor_uniform_float",
        "__floor_varying_double",
        "__floor_varying_float",
        "__insert_int16",
        "__insert_int32",
        "__insert_int64",
        "__insert_int8",
        "__intbits_uniform_double",
        "__intbits_uniform_float",
        "__intbits_varying_double",
        "__intbits_varying_float",
        "__max_uniform_double",
        "__max_uniform_float",
        "__max_uniform_int32",
        "__max_uniform_int64",
        "__max_uniform_uint32",
        "__max_uniform_uint64",
        "__max_varying_double",
        "__max_varying_float",
        "__max_varying_int32",
        "__max_varying_int64",
        "__max_varying_uint32",
        "__max_varying_uint64",
        "__memory_barrier",
        "__min_uniform_double",
        "__min_uniform_float",
        "__min_uniform_int32",
        "__min_uniform_int64",
        "__min_uniform_uint32",
        "__min_uniform_uint64",
        "__min_varying_double",
        "__min_varying_float",
        "__min_varying_int32",
        "__min_varying_int64",
        "__min_varying_uint32",
        "__min_varying_uint64",
        "__movmsk",
        "__new_uniform",
        "__new_varying32",
        "__new_varying64",
        "__num_cores",
        "__packed_load_active",
        "__packed_store_active",
        "__popcnt_int32",
        "__popcnt_int64",
        "__prefetch_read_uniform_1",
        "__prefetch_read_uniform_2",
        "__prefetch_read_uniform_3",
        "__prefetch_read_uniform_nt",
        "__rcp_uniform_float",
        "__rcp_varying_float",
        "__reduce_add_double",
        "__reduce_add_float",
        "__reduce_add_int32",
        "__reduce_add_int64",
        "__reduce_add_uint32",
        "__reduce_add_uint64",
        "__reduce_equal_double",
        "__reduce_equal_float",
        "__reduce_equal_int32",
        "__reduce_equal_int64",
        "__reduce_max_double",
        "__reduce_max_float",
        "__reduce_max_int32",
        "__reduce_max_int64",
        "__reduce_max_uint32",
        "__reduce_max_uint64",
        "__reduce_min_double",
        "__reduce_min_float",
        "__reduce_min_int32",
        "__reduce_min_int64",
        "__reduce_min_uint32",
        "__reduce_min_uint64",
        "__rotate_double",
        "__rotate_float",
        "__rotate_i16",
        "__rotate_i32",
        "__rotate_i64",
        "__rotate_i8",
        "__round_uniform_double",
        "__round_uniform_float",
        "__round_varying_double",
        "__round_varying_float",
        "__rsqrt_uniform_float",
        "__rsqrt_varying_float",
        "__sext_uniform_bool",
        "__sext_varying_bool",
        "__shuffle2_double",
        "__shuffle2_float",
        "__shuffle2_i16",
        "__shuffle2_i32",
        "__shuffle2_i64",
        "__shuffle2_i8",
        "__shuffle_double",
        "__shuffle_float",
        "__shuffle_i16",
        "__shuffle_i32",
        "__shuffle_i64",
        "__shuffle_i8",
        "__soa_to_aos3_float",
        "__soa_to_aos3_float16",
        "__soa_to_aos3_float4",
        "__soa_to_aos3_float8",
        "__soa_to_aos3_int32",
        "__soa_to_aos4_float",
        "__soa_to_aos4_float16",
        "__soa_to_aos4_float4",
        "__soa_to_aos4_float8",
        "__soa_to_aos4_int32",
        "__sqrt_uniform_double",
        "__sqrt_uniform_float",
        "__sqrt_varying_double",
        "__sqrt_varying_float",
        "__stdlib_atan",
        "__stdlib_atan2",
        "__stdlib_atan2f",
        "__stdlib_atanf",
        "__stdlib_cos",
        "__stdlib_cosf",
        "__stdlib_exp",
        "__stdlib_expf",
        "__stdlib_log",
        "__stdlib_logf",
        "__stdlib_pow",
        "__stdlib_powf",
        "__stdlib_sin",
        "__stdlib_sincos",
        "__stdlib_sincosf",
        "__stdlib_sinf",
        "__stdlib_tan",
        "__stdlib_tanf",
        "__svml_sin",
        "__svml_cos",
        "__svml_sincos",
        "__svml_tan",
        "__svml_atan",
        "__svml_atan2",
        "__svml_exp",
        "__svml_log",
        "__svml_pow",
        "__undef_uniform",
        "__undef_varying",
        "__vec4_add_float",
        "__vec4_add_int32",
        "__vselect_float",
        "__vselect_i32",
    };

    int count = sizeof(names) / sizeof(names[0]);
    for (int i = 0; i < count; ++i) {
        llvm::Function *f = module->getFunction(names[i]);
        if (f != NULL && f->empty() == false)
            f->setLinkage(llvm::GlobalValue::InternalLinkage);
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
void
AddBitcodeToModule(const unsigned char *bitcode, int length,
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
        Assert(bcTriple.getArch() == llvm::Triple::UnknownArch ||
               mTriple.getArch() == bcTriple.getArch());
        Assert(bcTriple.getVendor() == llvm::Triple::UnknownVendor ||
               mTriple.getVendor() == bcTriple.getVendor());
        bcModule->setTargetTriple(mTriple.str());

        std::string(linkError);
        if (llvm::Linker::LinkModules(module, bcModule, 
#if defined(LLVM_3_0) || defined(LLVM_3_0svn) || defined(LLVM_3_1svn)
                                      llvm::Linker::DestroySource,
#endif // LLVM_3_0
                                      &linkError))
            Error(SourcePos(), "Error linking stdlib bitcode: %s", linkError.c_str());
        lSetInternalFunctions(module);
        if (symbolTable != NULL)
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
    Symbol *pw = new Symbol(name, SourcePos(), AtomicType::UniformConstInt32,
                            SC_STATIC);
    pw->constValue = new ConstExpr(pw->type, val, SourcePos());
    LLVM_TYPE_CONST llvm::Type *ltype = LLVMTypes::Int32Type;
    llvm::Constant *linit = LLVMInt32(val);
    pw->storagePtr = new llvm::GlobalVariable(*module, ltype, true, 
                                              llvm::GlobalValue::InternalLinkage,
                                              linit, pw->name.c_str());
    symbolTable->AddVariable(pw);
}



static void
lDefineConstantIntFunc(const char *name, int val, llvm::Module *module,
                       SymbolTable *symbolTable) {
    std::vector<const Type *> args;
    FunctionType *ft = new FunctionType(AtomicType::UniformInt32, args, SourcePos());
    Symbol *sym = new Symbol(name, SourcePos(), ft, SC_STATIC);

    llvm::Function *func = module->getFunction(name);
    Assert(func != NULL); // it should be declared already...
    func->addFnAttr(llvm::Attribute::AlwaysInline);
    llvm::BasicBlock *bblock = llvm::BasicBlock::Create(*g->ctx, "entry", func, 0);
    llvm::ReturnInst::Create(*g->ctx, LLVMInt32(val), bblock);

    sym->function = func;
    symbolTable->AddVariable(sym);
}



static void
lDefineProgramIndex(llvm::Module *module, SymbolTable *symbolTable) {
    Symbol *pidx = new Symbol("programIndex", SourcePos(), 
                              AtomicType::VaryingConstInt32, SC_STATIC);

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
    if (g->target.is32Bit) {
        extern unsigned char builtins_bitcode_c_32[];
        extern int builtins_bitcode_c_32_length;
        AddBitcodeToModule(builtins_bitcode_c_32, builtins_bitcode_c_32_length, 
                           module, symbolTable);
    }
    else {
        extern unsigned char builtins_bitcode_c_64[];
        extern int builtins_bitcode_c_64_length;
        AddBitcodeToModule(builtins_bitcode_c_64, builtins_bitcode_c_64_length, 
                           module, symbolTable);
    }

    // Next, add the target's custom implementations of the various needed
    // builtin functions (e.g. __masked_store_32(), etc).
    switch (g->target.isa) {
    case Target::SSE2:
        extern unsigned char builtins_bitcode_sse2[];
        extern int builtins_bitcode_sse2_length;
        extern unsigned char builtins_bitcode_sse2_x2[];
        extern int builtins_bitcode_sse2_x2_length;
        switch (g->target.vectorWidth) {
        case 4: 
            AddBitcodeToModule(builtins_bitcode_sse2, builtins_bitcode_sse2_length, 
                               module, symbolTable);
            break;
        case 8:
            AddBitcodeToModule(builtins_bitcode_sse2_x2, builtins_bitcode_sse2_x2_length, 
                               module, symbolTable);
            break;
        default:
            FATAL("logic error in DefineStdlib");
        }
        break;
    case Target::SSE4:
        extern unsigned char builtins_bitcode_sse4[];
        extern int builtins_bitcode_sse4_length;
        extern unsigned char builtins_bitcode_sse4_x2[];
        extern int builtins_bitcode_sse4_x2_length;
        switch (g->target.vectorWidth) {
        case 4: 
            AddBitcodeToModule(builtins_bitcode_sse4,
                               builtins_bitcode_sse4_length, 
                               module, symbolTable);
            break;
        case 8:
            AddBitcodeToModule(builtins_bitcode_sse4_x2, 
                               builtins_bitcode_sse4_x2_length, 
                               module, symbolTable);
            break;
        default:
            FATAL("logic error in DefineStdlib");
        }
        break;
    case Target::AVX:
        switch (g->target.vectorWidth) {
        case 8:
            extern unsigned char builtins_bitcode_avx1[];
            extern int builtins_bitcode_avx1_length;
            AddBitcodeToModule(builtins_bitcode_avx1, 
                               builtins_bitcode_avx1_length, 
                               module, symbolTable);
            break;
        case 16:
            extern unsigned char builtins_bitcode_avx1_x2[];
            extern int builtins_bitcode_avx1_x2_length;
            AddBitcodeToModule(builtins_bitcode_avx1_x2, 
                               builtins_bitcode_avx1_x2_length,
                               module,  symbolTable);
            break;
        default:
            FATAL("logic error in DefineStdlib");
        }
        break;
    case Target::AVX2:
        switch (g->target.vectorWidth) {
        case 8:
            extern unsigned char builtins_bitcode_avx2[];
            extern int builtins_bitcode_avx2_length;
            AddBitcodeToModule(builtins_bitcode_avx2, 
                               builtins_bitcode_avx2_length, 
                               module, symbolTable);
            break;
        case 16:
            extern unsigned char builtins_bitcode_avx2_x2[];
            extern int builtins_bitcode_avx2_x2_length;
            AddBitcodeToModule(builtins_bitcode_avx2_x2, 
                               builtins_bitcode_avx2_x2_length,
                               module,  symbolTable);
            break;
        default:
            FATAL("logic error in DefineStdlib");
        }
        break;
    case Target::GENERIC:
        switch (g->target.vectorWidth) {
        case 4:
            extern unsigned char builtins_bitcode_generic_4[];
            extern int builtins_bitcode_generic_4_length;
            AddBitcodeToModule(builtins_bitcode_generic_4, 
                               builtins_bitcode_generic_4_length, 
                               module, symbolTable);
            break;
        case 8:
            extern unsigned char builtins_bitcode_generic_8[];
            extern int builtins_bitcode_generic_8_length;
            AddBitcodeToModule(builtins_bitcode_generic_8, 
                               builtins_bitcode_generic_8_length, 
                               module, symbolTable);
            break;
        case 16:
            extern unsigned char builtins_bitcode_generic_16[];
            extern int builtins_bitcode_generic_16_length;
            AddBitcodeToModule(builtins_bitcode_generic_16, 
                               builtins_bitcode_generic_16_length, 
                               module, symbolTable);
            break;
	case 1:
            extern unsigned char builtins_bitcode_generic_1[];
            extern int builtins_bitcode_generic_1_length;
            AddBitcodeToModule(builtins_bitcode_generic_1, 
                               builtins_bitcode_generic_1_length, 
                               module, symbolTable);
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
    lDefineConstantIntFunc("__fast_masked_vload", (int)g->opt.fastMaskedVload, module,
                           symbolTable);

    lDefineConstantInt("__have_native_half", (g->target.isa == Target::AVX2),
                       module, symbolTable);

    if (includeStdlibISPC) {
        // If the user wants the standard library to be included, parse the
        // serialized version of the stdlib.ispc file to get its
        // definitions added.
      if (g->target.isa == Target::GENERIC&&g->target.vectorWidth!=1) { // 1 wide uses x86 stdlib
            extern char stdlib_generic_code[];
            yy_scan_string(stdlib_generic_code);
            yyparse();
        }
        else {
            extern char stdlib_x86_code[];
            yy_scan_string(stdlib_x86_code);
            yyparse();
        }
    }
}
