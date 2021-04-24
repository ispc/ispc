/*
  Copyright (c) 2010-2021, Intel Corporation
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
#include "ctx.h"
#include "expr.h"
#include "llvmutil.h"
#include "module.h"
#include "sym.h"
#include "type.h"
#include "util.h"

#include <math.h>
#include <stdlib.h>

#include <llvm/ADT/Triple.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Target/TargetMachine.h>

#ifdef ISPC_GENX_ENABLED
#include <llvm/GenXIntrinsics/GenXIntrinsics.h>
#endif

using namespace ispc;

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
    else if (t == LLVMTypes::FloatVectorPointerType)
        return PointerType::GetUniform(AtomicType::VaryingFloat);
    else if (t == LLVMTypes::DoubleVectorPointerType)
        return PointerType::GetUniform(AtomicType::VaryingDouble);

    return NULL;
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
    if (g->target->getMaskBitCount() != 1 && name == "__sext_varying_bool") {
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
        if (returnType == NULL) {
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
            if (type == NULL) {
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
    if (existingSym != NULL) {
        return existingSym;
    }
    SourcePos noPos;
    noPos.name = "LLVM Intrinsic";
    const llvm::FunctionType *ftype = func->getFunctionType();
    std::string name = std::string(func->getName());
    const Type *returnType = lLLVMTypeToISPCType(ftype->getReturnType(), false);
    if (returnType == NULL) {
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
        if (type == NULL) {
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
static void lAddModuleSymbols(llvm::Module *module, SymbolTable *symbolTable) {
#if 0
    // FIXME: handle globals?
    Assert(module->global_empty());
#endif

    llvm::Module::iterator iter;
    for (iter = module->begin(); iter != module->end(); ++iter) {
        llvm::Function *func = &*iter;
        lCreateISPCSymbol(func, symbolTable);
    }
}

static void lUpdateIntrinsicsAttributes(llvm::Module *module) {
#ifdef ISPC_GENX_ENABLED
    for (auto F = module->begin(), E = module->end(); F != E; ++F) {
        llvm::Function *Fn = &*F;

        if (Fn && llvm::GenXIntrinsic::isGenXIntrinsic(Fn)) {
            Fn->setAttributes(
                llvm::GenXIntrinsic::getAttributes(Fn->getContext(), llvm::GenXIntrinsic::getGenXIntrinsicID(Fn)));
        }
    }
#endif
}

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
        if (!func->isIntrinsic())
            continue;

        const std::string funcName = func->getName().str();
        // Work around http://llvm.org/bugs/show_bug.cgi?id=10438; only
        // check the llvm.x86.* intrinsics for now...
        if (!strncmp(funcName.c_str(), "llvm.x86.", 9)) {
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
static void lSetInternalFunctions(llvm::Module *module) {
    // clang-format off
    const char *names[] = {
        "__add_float",
        "__add_int32",
        "__add_uniform_double",
        "__add_uniform_int32",
        "__add_uniform_int64",
        "__add_varying_double",
        "__add_varying_int32",
        "__add_varying_int64",
        "__all",
        "__any",
        "__aos_to_soa2_double",
        "__aos_to_soa2_double1",
        "__aos_to_soa2_double16",
        "__aos_to_soa2_double32",
        "__aos_to_soa2_double4",
        "__aos_to_soa2_double64",
        "__aos_to_soa2_double8",
        "__aos_to_soa2_float",
        "__aos_to_soa2_float1",
        "__aos_to_soa2_float16",
        "__aos_to_soa2_float32",
        "__aos_to_soa2_float4",
        "__aos_to_soa2_float64",
        "__aos_to_soa2_float8",
        "__aos_to_soa3_double",
        "__aos_to_soa3_double1",
        "__aos_to_soa3_double16",
        "__aos_to_soa3_double32",
        "__aos_to_soa3_double4",
        "__aos_to_soa3_double64",
        "__aos_to_soa3_double8",
        "__aos_to_soa3_float",
        "__aos_to_soa3_float1",
        "__aos_to_soa3_float16",
        "__aos_to_soa3_float32",
        "__aos_to_soa3_float4",
        "__aos_to_soa3_float64",
        "__aos_to_soa3_float8",
        "__aos_to_soa4_double",
        "__aos_to_soa4_double1",
        "__aos_to_soa4_double16",
        "__aos_to_soa4_double32",
        "__aos_to_soa4_double4",
        "__aos_to_soa4_double64",
        "__aos_to_soa4_double8",
        "__aos_to_soa4_float",
        "__aos_to_soa4_float1",
        "__aos_to_soa4_float16",
        "__aos_to_soa4_float32",
        "__aos_to_soa4_float4",
        "__aos_to_soa4_float64",
        "__aos_to_soa4_float8",
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
        "__cast_mask_to_i1",
        "__cast_mask_to_i8",
        "__cast_mask_to_i16",
        "__ceil_uniform_double",
        "__ceil_uniform_float",
        "__ceil_varying_double",
        "__ceil_varying_float",
        "__clock",
        "__count_trailing_zeros_i32",
        "__count_trailing_zeros_i64",
        "__count_leading_zeros_i32",
        "__count_leading_zeros_i64",
        "__delete_uniform_32rt",
        "__delete_uniform_64rt",
        "__delete_varying_32rt",
        "__delete_varying_64rt",
        "__divs_ui64",
        "__divs_vi64",
        "__divus_ui64",
        "__divus_vi64",
        "__do_assume_uniform",
        "__do_assert_uniform",
        "__do_assert_varying",
        "__do_print",
#ifdef ISPC_GENX_ENABLED
        "__do_print_cm",
        "__do_print_lz",
        "__do_print_cm_str",
        "__send_eot",
#endif //ISPC_GENX_ENABLED
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
        "__extract_bool",
        "__extract_int16",
        "__extract_int32",
        "__extract_int64",
        "__extract_int8",
        "__extract_mask_low",
        "__extract_mask_hi",
        "__fastmath",
        "__float_to_half_uniform",
        "__float_to_half_varying",
        "__floatbits_uniform_int32",
        "__floatbits_varying_int32",
        "__floor_uniform_double",
        "__floor_uniform_float",
        "__floor_varying_double",
        "__floor_varying_float",
        "__get_system_isa",
        "__half_to_float_uniform",
        "__half_to_float_varying",
        "__idiv_uint8",
        "__idiv_uint16",
        "__idiv_uint32",
        "__idiv_int8",
        "__idiv_int16",
        "__idiv_int32",
        "__insert_bool",
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
        "__memcpy32",
        "__memcpy64",
        "__memmove32",
        "__memmove64",
        "__memset32",
        "__memset64",
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
        "__new_uniform_32rt",
        "__new_uniform_64rt",
        "__new_varying32_32rt",
        "__new_varying32_64rt",
        "__new_varying64_64rt",
        "__none",
        "__num_cores",
        "__packed_load_activei32",
        "__packed_load_activei64",
        "__packed_store_activei32",
        "__packed_store_activei64",
        "__packed_store_active2i32",
        "__packed_store_active2i64",
        "__padds_ui8",
        "__padds_ui16",
        "__padds_ui32",
        "__padds_ui64",
        "__padds_vi8",
        "__padds_vi16",
        "__padds_vi32",
        "__padds_vi64",
        "__paddus_ui8",
        "__paddus_ui16",
        "__paddus_ui32",
        "__paddus_ui64",
        "__paddus_vi8",
        "__paddus_vi16",
        "__paddus_vi32",
        "__paddus_vi64",
        "__pmuls_ui8",
        "__pmuls_ui16",
        "__pmuls_ui32",
        "__pmuls_vi8",
        "__pmuls_vi16",
        "__pmuls_vi32",
        "__pmulus_ui8",
        "__pmulus_ui16",
        "__pmulus_ui32",
        "__pmulus_vi8",
        "__pmulus_vi16",
        "__pmulus_vi32",
        "__popcnt_int32",
        "__popcnt_int64",
        "__prefetch_read_uniform_1",
        "__prefetch_read_uniform_2",
        "__prefetch_read_uniform_3",
        "__prefetch_read_uniform_nt",
        "__pseudo_prefetch_read_varying_1",
        "__pseudo_prefetch_read_varying_2",
        "__pseudo_prefetch_read_varying_3",
        "__pseudo_prefetch_read_varying_nt",
        "__psubs_ui8",
        "__psubs_ui16",
        "__psubs_ui32",
        "__psubs_ui64",
        "__psubs_vi8",
        "__psubs_vi16",
        "__psubs_vi32",
        "__psubs_vi64",
        "__psubus_ui8",
        "__psubus_ui16",
        "__psubus_ui32",
        "__psubus_ui64",
        "__psubus_vi8",
        "__psubus_vi16",
        "__psubus_vi32",
        "__psubus_vi64",
        "__rcp_fast_uniform_float",
        "__rcp_uniform_float",
        "__rcp_fast_varying_float",
        "__rcp_varying_float",
        "__rcp_uniform_double",
        "__rcp_varying_double",
        "__rdrand_i16",
        "__rdrand_i32",
        "__rdrand_i64",
        "__reduce_add_double",
        "__reduce_add_float",
        "__reduce_add_int8",
        "__reduce_add_int16",
        "__reduce_add_int32",
        "__reduce_add_int64",
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
        "__rems_ui64",
        "__rems_vi64",
        "__remus_ui64",
        "__remus_vi64",
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
        "__rsqrt_fast_varying_float",
        "__rsqrt_uniform_float",
        "__rsqrt_fast_uniform_float",
        "__rsqrt_varying_float",
        "__rsqrt_uniform_double",
        "__rsqrt_varying_double",
        "__saturating_add_i8",
        "__saturating_add_i16",
        "__saturating_add_i32",
        "__saturating_add_i64",
        "__saturating_add_ui8",
        "__saturating_add_ui16",
        "__saturating_add_ui32",
        "__saturating_add_ui64",
        "__saturating_mul_i8",
        "__saturating_mul_i16",
        "__saturating_mul_i32",
        "__saturating_mul_ui8",
        "__saturating_mul_ui16",
        "__saturating_mul_ui32",
        "__set_system_isa",
        "__sext_uniform_bool",
        "__sext_varying_bool",
        "__shift_double",
        "__shift_float",
        "__shift_i16",
        "__shift_i32",
        "__shift_i64",
        "__shift_i8",
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
        "__soa_to_aos2_double",
        "__soa_to_aos2_double1",
        "__soa_to_aos2_double16",
        "__soa_to_aos2_double32",
        "__soa_to_aos2_double4",
        "__soa_to_aos2_double64",
        "__soa_to_aos2_double8",
        "__soa_to_aos2_float",
        "__soa_to_aos2_float1",
        "__soa_to_aos2_float16",
        "__soa_to_aos2_float32",
        "__soa_to_aos2_float4",
        "__soa_to_aos2_float64",
        "__soa_to_aos2_float8",
        "__soa_to_aos3_double",
        "__soa_to_aos3_double1",
        "__soa_to_aos3_double16",
        "__soa_to_aos3_double32",
        "__soa_to_aos3_double4",
        "__soa_to_aos3_double64",
        "__soa_to_aos3_double8",
        "__soa_to_aos3_float",
        "__soa_to_aos3_float1",
        "__soa_to_aos3_float16",
        "__soa_to_aos3_float32",
        "__soa_to_aos3_float4",
        "__soa_to_aos3_float64",
        "__soa_to_aos3_float8",
        "__soa_to_aos4_double",
        "__soa_to_aos4_double1",
        "__soa_to_aos4_double16",
        "__soa_to_aos4_double32",
        "__soa_to_aos4_double4",
        "__soa_to_aos4_double64",
        "__soa_to_aos4_double8",
        "__soa_to_aos4_float",
        "__soa_to_aos4_float1",
        "__soa_to_aos4_float16",
        "__soa_to_aos4_float32",
        "__soa_to_aos4_float4",
        "__soa_to_aos4_float64",
        "__soa_to_aos4_float8",
        "__sqrt_uniform_double",
        "__sqrt_uniform_float",
        "__sqrt_varying_double",
        "__sqrt_varying_float",
        "__stdlib_acosf",
        "__stdlib_asinf",
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
        "__stdlib_asin",
        "__stdlib_sincos",
        "__stdlib_sincosf",
        "__stdlib_sinf",
        "__stdlib_tan",
        "__stdlib_tanf",
        "__streaming_load_uniform_double",
        "__streaming_load_uniform_float",
        "__streaming_load_uniform_i8",
        "__streaming_load_uniform_i16",
        "__streaming_load_uniform_i32",
        "__streaming_load_uniform_i64",
        "__streaming_load_varying_double",
        "__streaming_load_varying_float",
        "__streaming_load_varying_i8",
        "__streaming_load_varying_i16",
        "__streaming_load_varying_i32",
        "__streaming_load_varying_i64",
        "__streaming_store_uniform_double",
        "__streaming_store_uniform_float",
        "__streaming_store_uniform_i8",
        "__streaming_store_uniform_i16",
        "__streaming_store_uniform_i32",
        "__streaming_store_uniform_i64",
        "__streaming_store_varying_double",
        "__streaming_store_varying_float",
        "__streaming_store_varying_i8",
        "__streaming_store_varying_i16",
        "__streaming_store_varying_i32",
        "__streaming_store_varying_i64",
        "__svml_sind",
        "__svml_asind",
        "__svml_cosd",
        "__svml_acosd",
        "__svml_sincosd",
        "__svml_tand",
        "__svml_atand",
        "__svml_atan2d",
        "__svml_expd",
        "__svml_logd",
        "__svml_powd",
        "__svml_sinf",
        "__svml_asinf",
        "__svml_cosf",
        "__svml_acosf",
        "__svml_sincosf",
        "__svml_tanf",
        "__svml_atanf",
        "__svml_atan2f",
        "__svml_expf",
        "__svml_logf",
        "__svml_powf",
        "__trunc_uniform_double",
        "__trunc_uniform_float",
        "__trunc_varying_double",
        "__trunc_varying_float",
        "__log_uniform_float",
        "__log_varying_float",
        "__exp_uniform_float",
        "__exp_varying_float",
        "__pow_uniform_float",
        "__pow_varying_float",
        "__log_uniform_double",
        "__log_varying_double",
        "__exp_uniform_double",
        "__exp_varying_double",
        "__pow_uniform_double",
        "__pow_varying_double",
        "__sin_varying_float",
        "__asin_varying_float",
        "__cos_varying_float",
        "__acos_varying_float",
        "__sincos_varying_float",
        "__tan_varying_float",
        "__atan_varying_float",
        "__atan2_varying_float",
        "__sin_uniform_float",
        "__asin_uniform_float",
        "__cos_uniform_float",
        "__acos_uniform_float",
        "__sincos_uniform_float",
        "__tan_uniform_float",
        "__atan_uniform_float",
        "__atan2_uniform_float",
        "__sin_varying_double",
        "__asin_varying_double",
        "__cos_varying_double",
        "__acos_varying_double",
        "__sincos_varying_double",
        "__tan_varying_double",
        "__atan_varying_double",
        "__atan2_varying_double",
        "__sin_uniform_double",
        "__asin_uniform_double",
        "__cos_uniform_double",
        "__acos_uniform_double",
        "__sincos_uniform_double",
        "__tan_uniform_double",
        "__atan_uniform_double",
        "__atan2_uniform_double",
        "__undef_uniform",
        "__undef_varying",
        "__vec4_add_float",
        "__vec4_add_int32",
        "__vselect_float",
        "__vselect_i32",
        "ISPCAlloc",
        "ISPCLaunch",
        "ISPCSync",
// ISPC_GENX_ENABLED
        "__task_index0",
        "__task_index1",
        "__task_index2",
        "__task_index",
        "__task_count0",
        "__task_count1",
        "__task_count2",
        "__task_count",
    };
    // clang-format on
    for (auto name : names) {
        llvm::Function *f = module->getFunction(name);
        if (f != NULL && f->empty() == false) {
            f->setLinkage(llvm::GlobalValue::InternalLinkage);
            // TO-DO : Revisit adding this back for ARM support.
            // g->target->markFuncWithTargetAttr(f);
        }
    }
}

static void lSetAlwaysInlineFunctions(llvm::Module *module) {
    std::vector<const char *> names = {
#ifdef ISPC_GENX_ENABLED
        "__do_print_cm", "__do_print_lz"
#endif /* ISPC_GENX_ENABLED */
    };
    for (auto name : names) {
        llvm::Function *f = module->getFunction(name);
        if (f) {
            if (f->hasFnAttribute(llvm::Attribute::NoInline)) {
                f->removeFnAttr(llvm::Attribute::NoInline);
            }
            f->addFnAttr(llvm::Attribute::AlwaysInline);
        }
    }
}

/** This utility function takes serialized binary LLVM bitcode and adds its
    definitions to the given module.  Functions in the bitcode that can be
    mapped to ispc functions are also added to the symbol table.

    @param lib         Pointer to BitcodeLib class representing LLVM bitcode (e.g. the contents of a *.bc file)
    @param module      Module to link the bitcode into
    @param symbolTable Symbol table to add definitions to
 */
void ispc::AddBitcodeToModule(const BitcodeLib *lib, llvm::Module *module, SymbolTable *symbolTable) {
    llvm::StringRef sb = llvm::StringRef((const char *)lib->getLib(), lib->getSize());
    llvm::MemoryBufferRef bcBuf = llvm::MemoryBuffer::getMemBuffer(sb)->getMemBufferRef();

    llvm::Expected<std::unique_ptr<llvm::Module>> ModuleOrErr = llvm::parseBitcodeFile(bcBuf, *g->ctx);
    if (!ModuleOrErr) {
        Error(SourcePos(), "Error parsing stdlib bitcode: %s", toString(ModuleOrErr.takeError()).c_str());
    } else {
        llvm::Module *bcModule = ModuleOrErr.get().release();
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

        // Disable this code for cross compilation
#if 0
            {
                Assert(bcTriple.getArch() == llvm::Triple::UnknownArch || mTriple.getArch() == bcTriple.getArch());
                Assert(bcTriple.getVendor() == llvm::Triple::UnknownVendor ||
                       mTriple.getVendor() == bcTriple.getVendor());

                // We unconditionally set module DataLayout to library, but we must
                // ensure that library and module DataLayouts are compatible.
                // If they are not, we should recompile the library for problematic
                // architecture and investigate what happened.
                // Generally we allow library DataLayout to be subset of module
                // DataLayout or library DataLayout to be empty.
                if (!VerifyDataLayoutCompatibility(module->getDataLayoutStr(), bcModule->getDataLayoutStr())) {
                    Warning(SourcePos(),
                            "Module DataLayout is incompatible with "
                            "library DataLayout:\n"
                            "Module  DL: %s\n"
                            "Library DL: %s\n",
                            module->getDataLayoutStr().c_str(), bcModule->getDataLayoutStr().c_str());
                }
            }
#endif

        bcModule->setTargetTriple(mTriple.str());
        bcModule->setDataLayout(module->getDataLayout());

        if (g->target->isGenXTarget()) {
            // Maybe we will use it for other targets in future,
            // but now it is needed only by GenX. We need
            // to update attributes because GenX intrinsics are
            // separated from the others and it is not done by default
            lUpdateIntrinsicsAttributes(bcModule);
        }

        // A hack to move over declaration, which have no definition.
        // New linker is kind of smart and think it knows better what to do, so
        // it removes unused declarations without definitions.
        // This trick should be legal, as both modules use the same LLVMContext.
        for (llvm::Function &f : *bcModule) {
            if (f.isDeclaration()) {
                // Declarations with uses will be moved by Linker.
                if (f.getNumUses() > 0)
                    continue;
                module->getOrInsertFunction(f.getName(), f.getFunctionType(), f.getAttributes());
            }
        }

        std::unique_ptr<llvm::Module> M(bcModule);
        if (llvm::Linker::linkModules(*module, std::move(M))) {
            Error(SourcePos(), "Error linking stdlib bitcode.");
        }

        lSetInternalFunctions(module);
        if (g->target->isGenXTarget()) {
            // For now this function is used for gen target only
            // TODO: check if its usage affects CPU targets
            lSetAlwaysInlineFunctions(module);
        }
        if (symbolTable != NULL)
            lAddModuleSymbols(module, symbolTable);
        lCheckModuleIntrinsics(module);
    }
}

/** Utility routine that defines a constant int32 with given value, adding
    the symbol to both the ispc symbol table and the given LLVM module.
 */
static void lDefineConstantInt(const char *name, int val, llvm::Module *module, SymbolTable *symbolTable,
                               std::vector<llvm::Constant *> &dbg_sym) {
    Symbol *sym = new Symbol(name, SourcePos(), AtomicType::UniformInt32->GetAsConstType(), SC_STATIC);
    sym->constValue = new ConstExpr(sym->type, val, SourcePos());
    llvm::Type *ltype = LLVMTypes::Int32Type;
    llvm::Constant *linit = LLVMInt32(val);
    auto GV = new llvm::GlobalVariable(*module, ltype, true, llvm::GlobalValue::InternalLinkage, linit, name);
    dbg_sym.push_back(GV);
    sym->storagePtr = GV;
    symbolTable->AddVariable(sym);

    if (m->diBuilder != NULL) {
        llvm::DIFile *file = m->diCompileUnit->getFile();
        llvm::DICompileUnit *cu = m->diCompileUnit;
        llvm::DIType *diType = sym->type->GetDIType(file);
        // FIXME? DWARF says that this (and programIndex below) should
        // have the DW_AT_artifical attribute.  It's not clear if this
        // matters for anything though.
        llvm::GlobalVariable *sym_GV_storagePtr = llvm::dyn_cast<llvm::GlobalVariable>(sym->storagePtr);
        Assert(sym_GV_storagePtr);
        llvm::DIGlobalVariableExpression *var =
            m->diBuilder->createGlobalVariableExpression(cu, name, name, file, 0 /* line */, diType, true /* static */);
        sym_GV_storagePtr->addDebugInfo(var);
        /*#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_6
                Assert(var.Verify());
        #else // LLVM 3.7+
              // coming soon
        #endif*/
    }
}

static void lDefineConstantIntFunc(const char *name, int val, llvm::Module *module, SymbolTable *symbolTable,
                                   std::vector<llvm::Constant *> &dbg_sym) {
    llvm::SmallVector<const Type *, 8> args;
    FunctionType *ft = new FunctionType(AtomicType::UniformInt32, args, SourcePos());
    Symbol *sym = new Symbol(name, SourcePos(), ft, SC_STATIC);

    llvm::Function *func = module->getFunction(name);
    dbg_sym.push_back(func);
    Assert(func != NULL); // it should be declared already...
    func->addFnAttr(llvm::Attribute::AlwaysInline);
    llvm::BasicBlock *bblock = llvm::BasicBlock::Create(*g->ctx, "entry", func, 0);
    llvm::ReturnInst::Create(*g->ctx, LLVMInt32(val), bblock);

    sym->function = func;
    symbolTable->AddVariable(sym);
}

static void lDefineProgramIndex(llvm::Module *module, SymbolTable *symbolTable,
                                std::vector<llvm::Constant *> &dbg_sym) {
    Symbol *sym = new Symbol("programIndex", SourcePos(), AtomicType::VaryingInt32->GetAsConstType(), SC_STATIC);

    int pi[ISPC_MAX_NVEC];
    for (int i = 0; i < g->target->getVectorWidth(); ++i)
        pi[i] = i;
    sym->constValue = new ConstExpr(sym->type, pi, SourcePos());

    llvm::Type *ltype = LLVMTypes::Int32VectorType;
    llvm::Constant *linit = LLVMInt32Vector(pi);

    auto GV =
        new llvm::GlobalVariable(*module, ltype, true, llvm::GlobalValue::InternalLinkage, linit, sym->name.c_str());
    dbg_sym.push_back(GV);
    sym->storagePtr = GV;
    symbolTable->AddVariable(sym);

    if (m->diBuilder != NULL) {
        llvm::DIFile *file = m->diCompileUnit->getFile();
        llvm::DICompileUnit *cu = m->diCompileUnit;
        llvm::DIType *diType = sym->type->GetDIType(file);
        llvm::GlobalVariable *sym_GV_storagePtr = llvm::dyn_cast<llvm::GlobalVariable>(sym->storagePtr);
        Assert(sym_GV_storagePtr);
        llvm::DIGlobalVariableExpression *var = m->diBuilder->createGlobalVariableExpression(
            cu, sym->name.c_str(), sym->name.c_str(), file, 0 /* line */, diType, false /* static */);
        sym_GV_storagePtr->addDebugInfo(var);
        /*#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_6
                Assert(var.Verify());
        #else // LLVM 3.7+
              // coming soon
        #endif*/
    }
}

static void emitLLVMUsed(llvm::Module &module, std::vector<llvm::Constant *> &list) {
    // Convert list to what ConstantArray needs.
    llvm::SmallVector<llvm::Constant *, 8> UsedArray;
    UsedArray.reserve(list.size());
    for (auto c : list) {
        UsedArray.push_back(llvm::ConstantExpr::getPointerBitCastOrAddrSpaceCast(llvm::cast<llvm::Constant>(c),
                                                                                 LLVMTypes::Int8PointerType));
    }

    llvm::ArrayType *ATy = llvm::ArrayType::get(LLVMTypes::Int8PointerType, UsedArray.size());

    auto *GV = new llvm::GlobalVariable(module, ATy, false, llvm::GlobalValue::AppendingLinkage,
                                        llvm::ConstantArray::get(ATy, UsedArray), "llvm.used");

    GV->setSection("llvm.metadata");
}

void ispc::DefineStdlib(SymbolTable *symbolTable, llvm::LLVMContext *ctx, llvm::Module *module,
                        bool includeStdlibISPC) {
    // debug_symbols are symbols that supposed to be preserved in debug information.
    // They will be referenced in llvm.used intrinsic to prevent they removal from
    // the object file.
    std::vector<llvm::Constant *> debug_symbols;

    // Unlike regular builtins and dispatch module, which don't care about mangling of external functions,
    // so they only differentiate Windows/Unix and 32/64 bit, builtins-c need to take care about mangling.
    // Hence, different version for all potentially supported OSes.
    const BitcodeLib *builtins = g->target_registry->getBuiltinsCLib(g->target_os, g->target->getArch());
    Assert(builtins);
    AddBitcodeToModule(builtins, module, symbolTable);

    // Next, add the target's custom implementations of the various needed
    // builtin functions (e.g. __masked_store_32(), etc).
    const BitcodeLib *target =
        g->target_registry->getISPCTargetLib(g->target->getISPCTarget(), g->target_os, g->target->getArch());
    Assert(target);
    AddBitcodeToModule(target, module, symbolTable);

    // define the 'programCount' builtin variable
    lDefineConstantInt("programCount", g->target->getVectorWidth(), module, symbolTable, debug_symbols);

    // define the 'programIndex' builtin
    lDefineProgramIndex(module, symbolTable, debug_symbols);

    // Define __math_lib stuff.  This is used by stdlib.ispc, for example, to
    // figure out which math routines to end up calling...
    lDefineConstantInt("__math_lib", (int)g->mathLib, module, symbolTable, debug_symbols);
    lDefineConstantInt("__math_lib_ispc", (int)Globals::Math_ISPC, module, symbolTable, debug_symbols);
    lDefineConstantInt("__math_lib_ispc_fast", (int)Globals::Math_ISPCFast, module, symbolTable, debug_symbols);
    lDefineConstantInt("__math_lib_svml", (int)Globals::Math_SVML, module, symbolTable, debug_symbols);
    lDefineConstantInt("__math_lib_system", (int)Globals::Math_System, module, symbolTable, debug_symbols);
    lDefineConstantIntFunc("__fast_masked_vload", (int)g->opt.fastMaskedVload, module, symbolTable, debug_symbols);

    lDefineConstantInt("__have_native_half", g->target->hasHalf(), module, symbolTable, debug_symbols);
    lDefineConstantInt("__have_native_rand", g->target->hasRand(), module, symbolTable, debug_symbols);
    lDefineConstantInt("__have_native_transcendentals", g->target->hasTranscendentals(), module, symbolTable,
                       debug_symbols);
    lDefineConstantInt("__have_native_trigonometry", g->target->hasTrigonometry(), module, symbolTable, debug_symbols);
    lDefineConstantInt("__have_native_rsqrtd", g->target->hasRsqrtd(), module, symbolTable, debug_symbols);
    lDefineConstantInt("__have_native_rcpd", g->target->hasRcpd(), module, symbolTable, debug_symbols);
    lDefineConstantInt("__have_saturating_arithmetic", g->target->hasSatArith(), module, symbolTable, debug_symbols);

    lDefineConstantInt("__is_genx_target", (int)(g->target->isGenXTarget()), module, symbolTable, debug_symbols);

    if (g->forceAlignment != -1) {
        llvm::GlobalVariable *alignment = module->getGlobalVariable("memory_alignment", true);
        alignment->setInitializer(LLVMInt32(g->forceAlignment));
    }

    if (g->generateDebuggingSymbols) {
        emitLLVMUsed(*module, debug_symbols);
    }

    if (includeStdlibISPC) {
        // If the user wants the standard library to be included, parse the
        // serialized version of the stdlib.ispc file to get its
        // definitions added.
        extern const char stdlib_mask1_code[], stdlib_mask8_code[];
        extern const char stdlib_mask16_code[], stdlib_mask32_code[], stdlib_mask64_code[];
        switch (g->target->getMaskBitCount()) {
        case 1:
            yy_scan_string(stdlib_mask1_code);
            break;
        case 8:
            yy_scan_string(stdlib_mask8_code);
            break;
        case 16:
            yy_scan_string(stdlib_mask16_code);
            break;
        case 32:
            yy_scan_string(stdlib_mask32_code);
            break;
        case 64:
            yy_scan_string(stdlib_mask64_code);
            break;
        default:
            FATAL("Unhandled mask bit size for stdlib.ispc");
        }
        yyparse();
    }
}
