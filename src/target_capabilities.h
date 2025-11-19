/*
  Copyright (c) 2025, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file target_capabilities.h
    @brief Defines target capability enums and related functionality.
*/

#pragma once

namespace ispc {

/** @brief Enumeration of target-specific capabilities
 *
 * This enum represents hardware/ISA capabilities that can be queried
 * to determine what features are available on a compilation target.
 * Used with std::bitset for efficient storage and querying.
 *
 */
enum class TargetCapability {
    // Indicates whether the CPU has ARM dot product instructions (SDOT/UDOT).
    // Enables accelerated dot product operations on:
    //  - signedxsigned 8-bit integers operations only (SDOT)
    //  - unsignedxunsigned 8-bit integers operations only (UDOT)
    //  - No support for mixed sign operations
    ArmDotProduct,
    // Indicates whether the CPU supports ARM I8MM instructions for 8-bit integers matrix multiplication.
    // Provides capability for mixed-sign int8 operations not covered by basic ARM dot product.
    ArmI8MM,
    // Indicates whether the target has conflict detection-based run-Length encoding (avx512cd).
    ConflictDetection,
    // Indicates whether the target has FP16 support.
    Fp16Support,
    // Indicates whether the target has FP64 support.
    Fp64Support,
    // Indicates whether the target has native support for float/half conversions.
    HalfConverts,
    // Indicates whether the target has full native support for float16 type, i.e.
    // arithmetic operations, rsqrt, rcp, etc.
    // TODO: this needs to be merged with m_hasFp16Support eventually, but we need to
    // define proper ARM targets with and without FP16 support first.
    HalfFullSupport,
    // Indicates whether the CPU has Intel VNNI (Vector Neural Network Instructions) support.
    // Enables accelerated dot product operations on:
    //  - 8-bit integers (mixed sign only)
    //  - 16-bit integers (mixed sign only)
    //  - With optional saturation arithmetic
    IntelVNNI,
    // Indicates whether the CPU supports 16-bit integer VNNI operations specifically.
    // Enables all combinations of signed/unsigned int16 dot products with optional saturation.
    IntelVNNI_Int16,
    // Indicates whether the CPU supports 8-bit integer VNNI operations specifically.
    // Enables all combinations of signed/unsigned int8 dot products with optional saturation.
    IntelVNNI_Int8,
    // Indicates whether there is an ISA random number instruction.
    Rand,
    // Indicates whether there is an ISA double precision rcp.
    Rcpd,
    // Indicates whether there is an ISA double precision rsqrt.
    Rsqrtd,
    // Indicates whether the target has special saturating arithmetic instructions.
    SaturatingArithmetic,
    // Indicates whether the target has support for transcendentals (beyond
    // sqrt, which we assume that all of them handle).
    Transcendentals,
    // Indicates whether the target has ISA support for trigonometry.
    Trigonometry,
    // Indicates whether it's Xe target with prefetch capabilities.
    XePrefetch,
    // Must be last - used for bitset sizing.
    COUNT
};

/** @brief Metadata for each target capability
 *
 * Contains the preprocessor macro name and global variable name
 * associated with each capability. This centralized structure is used
 * by both the preprocessor (for defining macros) and the builtins
 * linker (for setting internal linkage on globals).
 */
struct CapabilityMetadata {
    TargetCapability capability;
    const char *macroName;     // Preprocessor macro (e.g., "ISPC_TARGET_HAS_HALF")
    const char *globalVarName; // Global variable name (e.g., "__have_native_half_converts")
};

/** @brief Central table of all capability metadata
 *
 * This table maps each TargetCapability enum value to its associated
 * preprocessor macro and global variable names. By keeping this in one
 * place, we ensure consistency across the codebase and make it easier
 * to add new capabilities.
 *
 */
static constexpr CapabilityMetadata g_capabilityMetadata[] = {
    {TargetCapability::ArmDotProduct, "ISPC_TARGET_HAS_ARM_DOT_PRODUCT", "__have_arm_dot_product"},
    {TargetCapability::ArmI8MM, "ISPC_TARGET_HAS_ARM_I8MM", "__have_arm_i8mm"},
    {TargetCapability::ConflictDetection, "ISPC_TARGET_HAS_CONFLICT_DETECTION", "__have_conflict_detection"},
    // TODO: Rename/alias these to ISPC_TARGET_HAS_FP16_SUPPORT and ISPC_TARGET_HAS_FP64_SUPPORT
    {TargetCapability::Fp16Support, "ISPC_FP16_SUPPORTED", nullptr},
    {TargetCapability::Fp64Support, "ISPC_FP64_SUPPORTED", nullptr},
    {TargetCapability::HalfConverts, "ISPC_TARGET_HAS_HALF", "__have_native_half_converts"},
    {TargetCapability::HalfFullSupport, "ISPC_TARGET_HAS_HALF_FULL_SUPPORT", "__have_native_half_full_support"},
    {TargetCapability::IntelVNNI, "ISPC_TARGET_HAS_INTEL_VNNI", "__have_intel_vnni"},
    {TargetCapability::IntelVNNI_Int16, "ISPC_TARGET_HAS_INTEL_VNNI_INT16", "__have_intel_vnni_int16"},
    {TargetCapability::IntelVNNI_Int8, "ISPC_TARGET_HAS_INTEL_VNNI_INT8", "__have_intel_vnni_int8"},
    {TargetCapability::Rand, "ISPC_TARGET_HAS_RAND", "__have_native_rand"},
    {TargetCapability::Rcpd, "ISPC_TARGET_HAS_RCPD", "__have_native_rcpd"},
    {TargetCapability::Rsqrtd, "ISPC_TARGET_HAS_RSQRTD", "__have_native_rsqrtd"},
    {TargetCapability::SaturatingArithmetic, "ISPC_TARGET_HAS_SATURATING_ARITHMETIC", "__have_saturating_arithmetic"},
    {TargetCapability::Transcendentals, "ISPC_TARGET_HAS_TRANSCENDENTALS", "__have_native_transcendentals"},
    {TargetCapability::Trigonometry, "ISPC_TARGET_HAS_TRIGONOMETRY", "__have_native_trigonometry"},
    {TargetCapability::XePrefetch, "ISPC_TARGET_HAS_XE_PREFETCH", "__have_xe_prefetch"},
};

/** @brief List of non-capability global variable names
 *
 * These globals represent compiler/target configuration settings rather than
 * hardware capabilities. They need internal linkage set in builtins.cpp.
 * This list corresponds to ISPC_CONFIG_CONSTANTS in stdlib/include/core.isph.
 *
 * Note: Entries are sorted alphabetically.
 */
static constexpr const char *g_configGlobalNames[] = {
    "__fast_masked_vload",
    "__have_rotate_via_shuffle_16",
    "__have_rotate_via_shuffle_32",
    "__have_rotate_via_shuffle_64",
    "__have_rotate_via_shuffle_8",
    "__have_shift_via_shuffle_16",
    "__have_shift_via_shuffle_32",
    "__have_shift_via_shuffle_64",
    "__have_shift_via_shuffle_8",
    "__is_xe_target",
    "__is_avx512_target",
    "__math_lib",
    "__memory_alignment",
};

} // namespace ispc
