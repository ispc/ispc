/*
  Copyright (c) 2022, Intel Corporation
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

#include "MakeInternalFuncsStatic.h"

namespace ispc {

char MakeInternalFuncsStaticPass::ID = 0;

bool MakeInternalFuncsStaticPass::runOnModule(llvm::Module &module) {
    const char *names[] = {
        "__avg_up_uint8",
        "__avg_up_int8",
        "__avg_up_uint16",
        "__avg_up_int16",
        "__avg_down_uint8",
        "__avg_down_int8",
        "__avg_down_uint16",
        "__avg_down_int16",
        "__fast_masked_vload",
        "__gather_factored_base_offsets32_i8",
        "__gather_factored_base_offsets32_i16",
        "__gather_factored_base_offsets32_i32",
        "__gather_factored_base_offsets32_i64",
        "__gather_factored_base_offsets32_half",
        "__gather_factored_base_offsets32_float",
        "__gather_factored_base_offsets32_double",
        "__gather_factored_base_offsets64_i8",
        "__gather_factored_base_offsets64_i16",
        "__gather_factored_base_offsets64_i32",
        "__gather_factored_base_offsets64_i64",
        "__gather_factored_base_offsets64_half",
        "__gather_factored_base_offsets64_float",
        "__gather_factored_base_offsets64_double",
        "__gather_base_offsets32_i8",
        "__gather_base_offsets32_i16",
        "__gather_base_offsets32_i32",
        "__gather_base_offsets32_i64",
        "__gather_base_offsets32_half",
        "__gather_base_offsets32_float",
        "__gather_base_offsets32_double",
        "__gather_base_offsets64_i8",
        "__gather_base_offsets64_i16",
        "__gather_base_offsets64_i32",
        "__gather_base_offsets64_i64",
        "__gather_base_offsets64_half",
        "__gather_base_offsets64_float",
        "__gather_base_offsets64_double",
        "__gather32_i8",
        "__gather32_i16",
        "__gather32_i32",
        "__gather32_i64",
        "__gather32_half",
        "__gather32_float",
        "__gather32_double",
        "__gather64_i8",
        "__gather64_i16",
        "__gather64_i32",
        "__gather64_i64",
        "__gather64_half",
        "__gather64_float",
        "__gather64_double",
        "__gather_elt32_i8",
        "__gather_elt32_i16",
        "__gather_elt32_i32",
        "__gather_elt32_i64",
        "__gather_elt32_half",
        "__gather_elt32_float",
        "__gather_elt32_double",
        "__gather_elt64_i8",
        "__gather_elt64_i16",
        "__gather_elt64_i32",
        "__gather_elt64_i64",
        "__gather_elt64_half",
        "__gather_elt64_float",
        "__gather_elt64_double",
        "__masked_load_i8",
        "__masked_load_i16",
        "__masked_load_i32",
        "__masked_load_i64",
        "__masked_load_half",
        "__masked_load_float",
        "__masked_load_double",
        "__masked_store_i8",
        "__masked_store_i16",
        "__masked_store_i32",
        "__masked_store_i64",
        "__masked_store_half",
        "__masked_store_float",
        "__masked_store_double",
        "__masked_store_blend_i8",
        "__masked_store_blend_i16",
        "__masked_store_blend_i32",
        "__masked_store_blend_i64",
        "__masked_store_blend_half",
        "__masked_store_blend_float",
        "__masked_store_blend_double",
        "__scatter_factored_base_offsets32_i8",
        "__scatter_factored_base_offsets32_i16",
        "__scatter_factored_base_offsets32_i32",
        "__scatter_factored_base_offsets32_i64",
        "__scatter_factored_base_offsets32_half",
        "__scatter_factored_base_offsets32_float",
        "__scatter_factored_base_offsets32_double",
        "__scatter_factored_base_offsets64_i8",
        "__scatter_factored_base_offsets64_i16",
        "__scatter_factored_base_offsets64_i32",
        "__scatter_factored_base_offsets64_i64",
        "__scatter_factored_base_offsets64_half",
        "__scatter_factored_base_offsets64_float",
        "__scatter_factored_base_offsets64_double",
        "__scatter_base_offsets32_i8",
        "__scatter_base_offsets32_i16",
        "__scatter_base_offsets32_i32",
        "__scatter_base_offsets32_i64",
        "__scatter_base_offsets32_half",
        "__scatter_base_offsets32_float",
        "__scatter_base_offsets32_double",
        "__scatter_base_offsets64_i8",
        "__scatter_base_offsets64_i16",
        "__scatter_base_offsets64_i32",
        "__scatter_base_offsets64_i64",
        "__scatter_base_offsets64_half",
        "__scatter_base_offsets64_float",
        "__scatter_base_offsets64_double",
        "__scatter_elt32_i8",
        "__scatter_elt32_i16",
        "__scatter_elt32_i32",
        "__scatter_elt32_i64",
        "__scatter_elt32_half",
        "__scatter_elt32_float",
        "__scatter_elt32_double",
        "__scatter_elt64_i8",
        "__scatter_elt64_i16",
        "__scatter_elt64_i32",
        "__scatter_elt64_i64",
        "__scatter_elt64_half",
        "__scatter_elt64_float",
        "__scatter_elt64_double",
        "__scatter32_i8",
        "__scatter32_i16",
        "__scatter32_i32",
        "__scatter32_i64",
        "__scatter32_half",
        "__scatter32_float",
        "__scatter32_double",
        "__scatter64_i8",
        "__scatter64_i16",
        "__scatter64_i32",
        "__scatter64_i64",
        "__scatter64_half",
        "__scatter64_float",
        "__scatter64_double",
        "__prefetch_read_varying_1",
        "__prefetch_read_varying_2",
        "__prefetch_read_varying_3",
        "__prefetch_read_varying_nt",
        "__prefetch_write_varying_1",
        "__prefetch_write_varying_2",
        "__prefetch_write_varying_3",
        "__keep_funcs_live",
#ifdef ISPC_XE_ENABLED
        "__masked_load_blend_i8",
        "__masked_load_blend_i16",
        "__masked_load_blend_i32",
        "__masked_load_blend_i64",
        "__masked_load_blend_half",
        "__masked_load_blend_float",
        "__masked_load_blend_double",
#endif
    };

    bool modifiedAny = false;
    int count = sizeof(names) / sizeof(names[0]);
    for (int i = 0; i < count; ++i) {
        llvm::Function *f = m->module->getFunction(names[i]);
        if (f != NULL && f->empty() == false) {
            f->setLinkage(llvm::GlobalValue::InternalLinkage);
            modifiedAny = true;
        }
    }

    return modifiedAny;
}

llvm::Pass *CreateMakeInternalFuncsStaticPass() { return new MakeInternalFuncsStaticPass; }

} // namespace ispc