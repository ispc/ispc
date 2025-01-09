/*
  Copyright (c) 2024-2025, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

// Builtins are target-specific functions implemented using LLVM IR. They are
// located in builtins directory. Their names typically start with double
// underscores. Unlike Standard Library functions, they are not guaranteed to
// be stable as compiler evolves. But they are used to implement the Standard
// Library.
//
// During compilation flow ISPC compiler sometimes needs to treat builtins
// differently than user functions or stdlib functions, or does optimizations
// that recognize certain builtins (like gather/scatter optimization flow).
// Addressing builtins by string representation of the name in the source code
// is error prone, so this file solves this problem by defining the pointer
// with builtin name pointing to a C-string representing the name. For example,
// instead of using "__any" identifier, it's available as builtin::__any.
//
// This file has to be the only place to declare builtin names that used across
// ISPC code base. We suppose that only functions defined in core.isph can be
// used in the ISPC code. The list is alphabetically sorted for convenience.

#include <string>
#include <unordered_map>
#include <vector>

namespace ispc {

namespace builtin {

// Groups of persistent functions.
enum class PersistentGroup {
    GATHER_DOUBLE = 0,
    GATHER_FLOAT,
    GATHER_HALF,
    GATHER_I16,
    GATHER_I32,
    GATHER_I64,
    GATHER_I8,
    PREFETCH_READ,
    PREFETCH_WRITE,
    SCATTER_DOUBLE,
    SCATTER_FLOAT,
    SCATTER_HALF,
    SCATTER_I16,
    SCATTER_I32,
    SCATTER_I64,
    SCATTER_I8,
};

// Functions that should be preserved across optimization pipeline unconditionally.
extern std::unordered_map<std::string, int> persistentFuncs;

// Groups of function that should be preserved across optimization pipeline.
// The logic is following: the whole group is to be preserved if any of
// function from the group is actually used.
extern std::unordered_map<PersistentGroup, std::vector<const char *>> persistentGroups;

// Only functions from core.isph should be listed here. As we suppose,
// that only they can be used in the code.
extern const char *const __all;
extern const char *const __any;
extern const char *const __avg_down_int16;
extern const char *const __avg_down_int8;
extern const char *const __avg_down_uint16;
extern const char *const __avg_down_uint8;
extern const char *const __avg_up_int16;
extern const char *const __avg_up_int8;
extern const char *const __avg_up_uint16;
extern const char *const __avg_up_uint8;
extern const char *const __count_leading_zeros_i32;
extern const char *const __count_leading_zeros_i64;
extern const char *const __count_trailing_zeros_i32;
extern const char *const __count_trailing_zeros_i64;
extern const char *const __delete_uniform_32rt;
extern const char *const __delete_uniform_64rt;
extern const char *const __delete_varying_32rt;
extern const char *const __delete_varying_64rt;
extern const char *const __do_assert_uniform;
extern const char *const __do_assert_varying;
extern const char *const __do_assume_uniform;
extern const char *const __do_print;
extern const char *const __fast_masked_vload;
extern const char *const __gather32_double;
extern const char *const __gather32_float;
extern const char *const __gather32_generic_double;
extern const char *const __gather32_generic_float;
extern const char *const __gather32_generic_half;
extern const char *const __gather32_generic_i16;
extern const char *const __gather32_generic_i32;
extern const char *const __gather32_generic_i64;
extern const char *const __gather32_generic_i8;
extern const char *const __gather32_half;
extern const char *const __gather32_i16;
extern const char *const __gather32_i32;
extern const char *const __gather32_i64;
extern const char *const __gather32_i8;
extern const char *const __gather64_double;
extern const char *const __gather64_float;
extern const char *const __gather64_generic_double;
extern const char *const __gather64_generic_float;
extern const char *const __gather64_generic_half;
extern const char *const __gather64_generic_i16;
extern const char *const __gather64_generic_i32;
extern const char *const __gather64_generic_i64;
extern const char *const __gather64_generic_i8;
extern const char *const __gather64_half;
extern const char *const __gather64_i16;
extern const char *const __gather64_i32;
extern const char *const __gather64_i64;
extern const char *const __gather64_i8;
extern const char *const __gather_base_offsets32_double;
extern const char *const __gather_base_offsets32_float;
extern const char *const __gather_base_offsets32_half;
extern const char *const __gather_base_offsets32_i16;
extern const char *const __gather_base_offsets32_i32;
extern const char *const __gather_base_offsets32_i64;
extern const char *const __gather_base_offsets32_i8;
extern const char *const __gather_base_offsets64_double;
extern const char *const __gather_base_offsets64_float;
extern const char *const __gather_base_offsets64_half;
extern const char *const __gather_base_offsets64_i16;
extern const char *const __gather_base_offsets64_i32;
extern const char *const __gather_base_offsets64_i64;
extern const char *const __gather_base_offsets64_i8;
extern const char *const __gather_factored_base_offsets32_double;
extern const char *const __gather_factored_base_offsets32_float;
extern const char *const __gather_factored_base_offsets32_half;
extern const char *const __gather_factored_base_offsets32_i16;
extern const char *const __gather_factored_base_offsets32_i32;
extern const char *const __gather_factored_base_offsets32_i64;
extern const char *const __gather_factored_base_offsets32_i8;
extern const char *const __gather_factored_base_offsets64_double;
extern const char *const __gather_factored_base_offsets64_float;
extern const char *const __gather_factored_base_offsets64_half;
extern const char *const __gather_factored_base_offsets64_i16;
extern const char *const __gather_factored_base_offsets64_i32;
extern const char *const __gather_factored_base_offsets64_i64;
extern const char *const __gather_factored_base_offsets64_i8;
extern const char *const __is_compile_time_constant_mask;
extern const char *const __is_compile_time_constant_uniform_int32;
extern const char *const __is_compile_time_constant_varying_int32;
extern const char *const ISPCAlloc;
extern const char *const ISPCLaunch;
extern const char *const ISPCSync;
extern const char *const ISPCInstrument;
extern const char *const __masked_load_blend_double;
extern const char *const __masked_load_blend_float;
extern const char *const __masked_load_blend_half;
extern const char *const __masked_load_blend_i16;
extern const char *const __masked_load_blend_i32;
extern const char *const __masked_load_blend_i64;
extern const char *const __masked_load_blend_i8;
extern const char *const __masked_load_double;
extern const char *const __masked_load_float;
extern const char *const __masked_load_half;
extern const char *const __masked_load_i16;
extern const char *const __masked_load_i32;
extern const char *const __masked_load_i64;
extern const char *const __masked_load_i8;
extern const char *const __masked_store_blend_double;
extern const char *const __masked_store_blend_float;
extern const char *const __masked_store_blend_half;
extern const char *const __masked_store_blend_i16;
extern const char *const __masked_store_blend_i32;
extern const char *const __masked_store_blend_i64;
extern const char *const __masked_store_blend_i8;
extern const char *const __masked_store_double;
extern const char *const __masked_store_float;
extern const char *const __masked_store_half;
extern const char *const __masked_store_i16;
extern const char *const __masked_store_i32;
extern const char *const __masked_store_i64;
extern const char *const __masked_store_i8;
extern const char *const __movmsk;
extern const char *const __new_uniform_32rt;
extern const char *const __new_uniform_64rt;
extern const char *const __new_varying32_32rt;
extern const char *const __new_varying32_64rt;
extern const char *const __new_varying64_64rt;
extern const char *const __none;
extern const char *const __num_cores;
extern const char *const __prefetch_read_sized_uniform_1;
extern const char *const __prefetch_read_sized_uniform_2;
extern const char *const __prefetch_read_sized_uniform_3;
extern const char *const __prefetch_read_sized_uniform_nt;
extern const char *const __prefetch_read_sized_varying_1;
extern const char *const __prefetch_read_sized_varying_2;
extern const char *const __prefetch_read_sized_varying_3;
extern const char *const __prefetch_read_sized_varying_nt;
extern const char *const __prefetch_read_uniform_1;
extern const char *const __prefetch_read_uniform_2;
extern const char *const __prefetch_read_uniform_3;
extern const char *const __prefetch_read_uniform_nt;
extern const char *const __prefetch_read_varying_1;
extern const char *const __prefetch_read_varying_1_native;
extern const char *const __prefetch_read_varying_2;
extern const char *const __prefetch_read_varying_2_native;
extern const char *const __prefetch_read_varying_3;
extern const char *const __prefetch_read_varying_3_native;
extern const char *const __prefetch_read_varying_nt;
extern const char *const __prefetch_read_varying_nt_native;
extern const char *const __prefetch_write_uniform_1;
extern const char *const __prefetch_write_uniform_2;
extern const char *const __prefetch_write_uniform_3;
extern const char *const __prefetch_write_varying_1;
extern const char *const __prefetch_write_varying_1_native;
extern const char *const __prefetch_write_varying_2;
extern const char *const __prefetch_write_varying_2_native;
extern const char *const __prefetch_write_varying_3;
extern const char *const __prefetch_write_varying_3_native;
extern const char *const __pseudo_gather32_double;
extern const char *const __pseudo_gather32_float;
extern const char *const __pseudo_gather32_half;
extern const char *const __pseudo_gather32_i16;
extern const char *const __pseudo_gather32_i32;
extern const char *const __pseudo_gather32_i64;
extern const char *const __pseudo_gather32_i8;
extern const char *const __pseudo_gather64_double;
extern const char *const __pseudo_gather64_float;
extern const char *const __pseudo_gather64_half;
extern const char *const __pseudo_gather64_i16;
extern const char *const __pseudo_gather64_i32;
extern const char *const __pseudo_gather64_i64;
extern const char *const __pseudo_gather64_i8;
extern const char *const __pseudo_gather_base_offsets32_double;
extern const char *const __pseudo_gather_base_offsets32_float;
extern const char *const __pseudo_gather_base_offsets32_half;
extern const char *const __pseudo_gather_base_offsets32_i16;
extern const char *const __pseudo_gather_base_offsets32_i32;
extern const char *const __pseudo_gather_base_offsets32_i64;
extern const char *const __pseudo_gather_base_offsets32_i8;
extern const char *const __pseudo_gather_base_offsets64_double;
extern const char *const __pseudo_gather_base_offsets64_float;
extern const char *const __pseudo_gather_base_offsets64_half;
extern const char *const __pseudo_gather_base_offsets64_i16;
extern const char *const __pseudo_gather_base_offsets64_i32;
extern const char *const __pseudo_gather_base_offsets64_i64;
extern const char *const __pseudo_gather_base_offsets64_i8;
extern const char *const __pseudo_gather_factored_base_offsets32_double;
extern const char *const __pseudo_gather_factored_base_offsets32_float;
extern const char *const __pseudo_gather_factored_base_offsets32_half;
extern const char *const __pseudo_gather_factored_base_offsets32_i16;
extern const char *const __pseudo_gather_factored_base_offsets32_i32;
extern const char *const __pseudo_gather_factored_base_offsets32_i64;
extern const char *const __pseudo_gather_factored_base_offsets32_i8;
extern const char *const __pseudo_gather_factored_base_offsets64_double;
extern const char *const __pseudo_gather_factored_base_offsets64_float;
extern const char *const __pseudo_gather_factored_base_offsets64_half;
extern const char *const __pseudo_gather_factored_base_offsets64_i16;
extern const char *const __pseudo_gather_factored_base_offsets64_i32;
extern const char *const __pseudo_gather_factored_base_offsets64_i64;
extern const char *const __pseudo_gather_factored_base_offsets64_i8;
extern const char *const __pseudo_masked_store_double;
extern const char *const __pseudo_masked_store_float;
extern const char *const __pseudo_masked_store_half;
extern const char *const __pseudo_masked_store_i16;
extern const char *const __pseudo_masked_store_i32;
extern const char *const __pseudo_masked_store_i64;
extern const char *const __pseudo_masked_store_i8;
extern const char *const __pseudo_prefetch_read_varying_1;
extern const char *const __pseudo_prefetch_read_varying_1_native;
extern const char *const __pseudo_prefetch_read_varying_2;
extern const char *const __pseudo_prefetch_read_varying_2_native;
extern const char *const __pseudo_prefetch_read_varying_3;
extern const char *const __pseudo_prefetch_read_varying_3_native;
extern const char *const __pseudo_prefetch_read_varying_nt;
extern const char *const __pseudo_prefetch_read_varying_nt_native;
extern const char *const __pseudo_prefetch_write_varying_1;
extern const char *const __pseudo_prefetch_write_varying_1_native;
extern const char *const __pseudo_prefetch_write_varying_2;
extern const char *const __pseudo_prefetch_write_varying_2_native;
extern const char *const __pseudo_prefetch_write_varying_3;
extern const char *const __pseudo_prefetch_write_varying_3_native;
extern const char *const __pseudo_scatter32_double;
extern const char *const __pseudo_scatter32_float;
extern const char *const __pseudo_scatter32_half;
extern const char *const __pseudo_scatter32_i16;
extern const char *const __pseudo_scatter32_i32;
extern const char *const __pseudo_scatter32_i64;
extern const char *const __pseudo_scatter32_i8;
extern const char *const __pseudo_scatter64_double;
extern const char *const __pseudo_scatter64_float;
extern const char *const __pseudo_scatter64_half;
extern const char *const __pseudo_scatter64_i16;
extern const char *const __pseudo_scatter64_i32;
extern const char *const __pseudo_scatter64_i64;
extern const char *const __pseudo_scatter64_i8;
extern const char *const __pseudo_scatter_base_offsets32_double;
extern const char *const __pseudo_scatter_base_offsets32_float;
extern const char *const __pseudo_scatter_base_offsets32_half;
extern const char *const __pseudo_scatter_base_offsets32_i16;
extern const char *const __pseudo_scatter_base_offsets32_i32;
extern const char *const __pseudo_scatter_base_offsets32_i64;
extern const char *const __pseudo_scatter_base_offsets32_i8;
extern const char *const __pseudo_scatter_base_offsets64_double;
extern const char *const __pseudo_scatter_base_offsets64_float;
extern const char *const __pseudo_scatter_base_offsets64_half;
extern const char *const __pseudo_scatter_base_offsets64_i16;
extern const char *const __pseudo_scatter_base_offsets64_i32;
extern const char *const __pseudo_scatter_base_offsets64_i64;
extern const char *const __pseudo_scatter_base_offsets64_i8;
extern const char *const __pseudo_scatter_factored_base_offsets32_double;
extern const char *const __pseudo_scatter_factored_base_offsets32_float;
extern const char *const __pseudo_scatter_factored_base_offsets32_half;
extern const char *const __pseudo_scatter_factored_base_offsets32_i16;
extern const char *const __pseudo_scatter_factored_base_offsets32_i32;
extern const char *const __pseudo_scatter_factored_base_offsets32_i64;
extern const char *const __pseudo_scatter_factored_base_offsets32_i8;
extern const char *const __pseudo_scatter_factored_base_offsets64_double;
extern const char *const __pseudo_scatter_factored_base_offsets64_float;
extern const char *const __pseudo_scatter_factored_base_offsets64_half;
extern const char *const __pseudo_scatter_factored_base_offsets64_i16;
extern const char *const __pseudo_scatter_factored_base_offsets64_i32;
extern const char *const __pseudo_scatter_factored_base_offsets64_i64;
extern const char *const __pseudo_scatter_factored_base_offsets64_i8;
extern const char *const __restore_ftz_daz_flags;
extern const char *const __scatter32_double;
extern const char *const __scatter32_float;
extern const char *const __scatter32_generic_double;
extern const char *const __scatter32_generic_float;
extern const char *const __scatter32_generic_half;
extern const char *const __scatter32_generic_i16;
extern const char *const __scatter32_generic_i32;
extern const char *const __scatter32_generic_i64;
extern const char *const __scatter32_generic_i8;
extern const char *const __scatter32_half;
extern const char *const __scatter32_i16;
extern const char *const __scatter32_i32;
extern const char *const __scatter32_i64;
extern const char *const __scatter32_i8;
extern const char *const __scatter64_double;
extern const char *const __scatter64_float;
extern const char *const __scatter64_generic_double;
extern const char *const __scatter64_generic_float;
extern const char *const __scatter64_generic_half;
extern const char *const __scatter64_generic_i16;
extern const char *const __scatter64_generic_i32;
extern const char *const __scatter64_generic_i64;
extern const char *const __scatter64_generic_i8;
extern const char *const __scatter64_half;
extern const char *const __scatter64_i16;
extern const char *const __scatter64_i32;
extern const char *const __scatter64_i64;
extern const char *const __scatter64_i8;
extern const char *const __scatter_base_offsets32_double;
extern const char *const __scatter_base_offsets32_float;
extern const char *const __scatter_base_offsets32_half;
extern const char *const __scatter_base_offsets32_i16;
extern const char *const __scatter_base_offsets32_i32;
extern const char *const __scatter_base_offsets32_i64;
extern const char *const __scatter_base_offsets32_i8;
extern const char *const __scatter_base_offsets64_double;
extern const char *const __scatter_base_offsets64_float;
extern const char *const __scatter_base_offsets64_half;
extern const char *const __scatter_base_offsets64_i16;
extern const char *const __scatter_base_offsets64_i32;
extern const char *const __scatter_base_offsets64_i64;
extern const char *const __scatter_base_offsets64_i8;
extern const char *const __scatter_elt32_double;
extern const char *const __scatter_elt32_float;
extern const char *const __scatter_elt32_half;
extern const char *const __scatter_elt32_i16;
extern const char *const __scatter_elt32_i32;
extern const char *const __scatter_elt32_i64;
extern const char *const __scatter_elt32_i8;
extern const char *const __scatter_elt64_double;
extern const char *const __scatter_elt64_float;
extern const char *const __scatter_elt64_half;
extern const char *const __scatter_elt64_i16;
extern const char *const __scatter_elt64_i32;
extern const char *const __scatter_elt64_i64;
extern const char *const __scatter_elt64_i8;
extern const char *const __scatter_factored_base_offsets32_double;
extern const char *const __scatter_factored_base_offsets32_float;
extern const char *const __scatter_factored_base_offsets32_half;
extern const char *const __scatter_factored_base_offsets32_i16;
extern const char *const __scatter_factored_base_offsets32_i32;
extern const char *const __scatter_factored_base_offsets32_i64;
extern const char *const __scatter_factored_base_offsets32_i8;
extern const char *const __scatter_factored_base_offsets64_double;
extern const char *const __scatter_factored_base_offsets64_float;
extern const char *const __scatter_factored_base_offsets64_half;
extern const char *const __scatter_factored_base_offsets64_i16;
extern const char *const __scatter_factored_base_offsets64_i32;
extern const char *const __scatter_factored_base_offsets64_i64;
extern const char *const __scatter_factored_base_offsets64_i8;
extern const char *const __set_ftz_daz_flags;
extern const char *const __set_system_isa;
extern const char *const __task_count;
extern const char *const __task_count0;
extern const char *const __task_count1;
extern const char *const __task_count2;
extern const char *const __task_index;
extern const char *const __task_index0;
extern const char *const __task_index1;
extern const char *const __task_index2;
extern const char *const __terminate_now;
extern const char *const __wasm_cmp_msk_eq;

} // namespace builtin

} // namespace ispc
