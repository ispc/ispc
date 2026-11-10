/*
  Copyright (c) 2026, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include "MemoryOpRegistry.h"
#include "builtins-decl.h"
#include "ispc.h"
#include "llvmutil.h"
#include "util.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Module.h>

#include <array>
#include <string>
#include <vector>

namespace ispc {

using namespace builtin;

namespace {

/** The element types of the gather, scatter and masked builtins, in the
    order of Family::names: the name suffix and the LLVM type of each. The
    types are LLVMTypes members, which are set for the current target after
    static initialization, so the table holds their addresses. */
struct ElementInfo {
    MemoryElement element;
    const char *suffix;
    llvm::Type *const *type;
};
constexpr std::array<ElementInfo, 7> kElements = {{
    {MemoryElement::I8, "_i8", &LLVMTypes::Int8Type},
    {MemoryElement::I16, "_i16", &LLVMTypes::Int16Type},
    {MemoryElement::Half, "_half", &LLVMTypes::Float16Type},
    {MemoryElement::I32, "_i32", &LLVMTypes::Int32Type},
    {MemoryElement::Float, "_float", &LLVMTypes::FloatType},
    {MemoryElement::I64, "_i64", &LLVMTypes::Int64Type},
    {MemoryElement::Double, "_double", &LLVMTypes::DoubleType},
}};

/** One builtin for each element type: everything but the element is
    shared. */
struct Family {
    struct {
        MemoryOpKind kind;
        MemoryOpForm form;
        MemoryOpFlavor flavor;
        uint8_t offsetBits;
    } key;
    std::array<const char *, kElements.size()> names;
};

struct PrefetchInfo {
    MemoryOpFlavor flavor;
    PrefetchHint hint;
    const char *name;
};

/** The parameters of each form, in the order core.isph declares them. The
    stored value is a parameter of scatters and masked stores only. */
enum class Role : uint8_t { Addresses, Base, Scale, Offsets, Delta, Value, Mask };
constexpr Role kGeneralParams[] = {Role::Addresses, Role::Value, Role::Mask};
constexpr Role kBaseOffsetsParams[] = {Role::Base, Role::Scale, Role::Offsets, Role::Value, Role::Mask};
constexpr Role kFactoredParams[] = {Role::Base, Role::Offsets, Role::Scale, Role::Delta, Role::Value, Role::Mask};
constexpr Role kPointerParams[] = {Role::Base, Role::Value, Role::Mask};

/** The MemoryOpLayout field of each Role, indexed by Role. */
constexpr int8_t MemoryOpLayout::*kRoleField[] = {
    &MemoryOpLayout::addresses, &MemoryOpLayout::base,  &MemoryOpLayout::scale, &MemoryOpLayout::offsets,
    &MemoryOpLayout::delta,     &MemoryOpLayout::value, &MemoryOpLayout::mask,
};
static_assert(std::size(kRoleField) == static_cast<size_t>(Role::Mask) + 1, "one field per Role");

MemoryOpLayout lLayout(MemoryOpKind kind, MemoryOpForm form) {
    llvm::ArrayRef<Role> params;
    switch (form) {
    case MemoryOpForm::General:
        params = kGeneralParams;
        break;
    case MemoryOpForm::BaseOffsets:
        params = kBaseOffsetsParams;
        break;
    case MemoryOpForm::Factored:
        params = kFactoredParams;
        break;
    case MemoryOpForm::Pointer:
        params = kPointerParams;
        break;
    }
    const bool stores = kind == MemoryOpKind::Scatter || kind == MemoryOpKind::MaskedStore;
    MemoryOpLayout layout;
    for (Role role : params) {
        if (role == Role::Value && !stores) {
            continue;
        }
        layout.*kRoleField[static_cast<size_t>(role)] = static_cast<int8_t>(layout.numArgs++);
    }
    return layout;
}

/** A fixed vector of lanes integer elements that are bits wide (any width if
    bits is 0). */
bool lIsIntVector(llvm::Type *t, unsigned lanes, unsigned bits) {
    llvm::FixedVectorType *vt = llvm::dyn_cast<llvm::FixedVectorType>(t);
    return vt != nullptr && vt->getNumElements() == lanes && vt->getElementType()->isIntegerTy() &&
           (bits == 0 || vt->getElementType()->getIntegerBitWidth() == bits);
}

/** A fixed vector of lanes elements of type elem. */
bool lIsElementVector(llvm::Type *t, unsigned lanes, llvm::Type *elem) {
    llvm::FixedVectorType *vt = llvm::dyn_cast<llvm::FixedVectorType>(t);
    return vt != nullptr && vt->getNumElements() == lanes && vt->getElementType() == elem;
}

uint32_t lPack(const MemoryOpKey &k) {
    return uint32_t(k.kind) | uint32_t(k.form) << 4 | uint32_t(k.element) << 8 | uint32_t(k.flavor) << 12 |
           uint32_t(k.offsetBits) << 16 | uint32_t(k.hint) << 24;
}

struct Registry {
    std::vector<MemoryOp> ops;
    llvm::StringMap<const MemoryOp *> byName;
    llvm::DenseMap<uint32_t, const MemoryOp *> byKey;

    Registry() {
        // The builtin:: names are defined in another translation unit, so
        // the tables are built on first use rather than at static
        // initialization.
        const Family families[] = {
            // Pseudo gathers and scatters, emitted by the frontend and
            // ImproveMemoryOps.
            {{MemoryOpKind::Gather, MemoryOpForm::General, MemoryOpFlavor::Pseudo, 32},
             {__pseudo_gather32_i8, __pseudo_gather32_i16, __pseudo_gather32_half, __pseudo_gather32_i32,
              __pseudo_gather32_float, __pseudo_gather32_i64, __pseudo_gather32_double}},
            {{MemoryOpKind::Gather, MemoryOpForm::General, MemoryOpFlavor::Pseudo, 64},
             {__pseudo_gather64_i8, __pseudo_gather64_i16, __pseudo_gather64_half, __pseudo_gather64_i32,
              __pseudo_gather64_float, __pseudo_gather64_i64, __pseudo_gather64_double}},
            {{MemoryOpKind::Gather, MemoryOpForm::BaseOffsets, MemoryOpFlavor::Pseudo, 32},
             {__pseudo_gather_base_offsets32_i8, __pseudo_gather_base_offsets32_i16,
              __pseudo_gather_base_offsets32_half, __pseudo_gather_base_offsets32_i32,
              __pseudo_gather_base_offsets32_float, __pseudo_gather_base_offsets32_i64,
              __pseudo_gather_base_offsets32_double}},
            {{MemoryOpKind::Gather, MemoryOpForm::BaseOffsets, MemoryOpFlavor::Pseudo, 64},
             {__pseudo_gather_base_offsets64_i8, __pseudo_gather_base_offsets64_i16,
              __pseudo_gather_base_offsets64_half, __pseudo_gather_base_offsets64_i32,
              __pseudo_gather_base_offsets64_float, __pseudo_gather_base_offsets64_i64,
              __pseudo_gather_base_offsets64_double}},
            {{MemoryOpKind::Gather, MemoryOpForm::Factored, MemoryOpFlavor::Pseudo, 32},
             {__pseudo_gather_factored_base_offsets32_i8, __pseudo_gather_factored_base_offsets32_i16,
              __pseudo_gather_factored_base_offsets32_half, __pseudo_gather_factored_base_offsets32_i32,
              __pseudo_gather_factored_base_offsets32_float, __pseudo_gather_factored_base_offsets32_i64,
              __pseudo_gather_factored_base_offsets32_double}},
            {{MemoryOpKind::Gather, MemoryOpForm::Factored, MemoryOpFlavor::Pseudo, 64},
             {__pseudo_gather_factored_base_offsets64_i8, __pseudo_gather_factored_base_offsets64_i16,
              __pseudo_gather_factored_base_offsets64_half, __pseudo_gather_factored_base_offsets64_i32,
              __pseudo_gather_factored_base_offsets64_float, __pseudo_gather_factored_base_offsets64_i64,
              __pseudo_gather_factored_base_offsets64_double}},
            {{MemoryOpKind::Scatter, MemoryOpForm::General, MemoryOpFlavor::Pseudo, 32},
             {__pseudo_scatter32_i8, __pseudo_scatter32_i16, __pseudo_scatter32_half, __pseudo_scatter32_i32,
              __pseudo_scatter32_float, __pseudo_scatter32_i64, __pseudo_scatter32_double}},
            {{MemoryOpKind::Scatter, MemoryOpForm::General, MemoryOpFlavor::Pseudo, 64},
             {__pseudo_scatter64_i8, __pseudo_scatter64_i16, __pseudo_scatter64_half, __pseudo_scatter64_i32,
              __pseudo_scatter64_float, __pseudo_scatter64_i64, __pseudo_scatter64_double}},
            {{MemoryOpKind::Scatter, MemoryOpForm::BaseOffsets, MemoryOpFlavor::Pseudo, 32},
             {__pseudo_scatter_base_offsets32_i8, __pseudo_scatter_base_offsets32_i16,
              __pseudo_scatter_base_offsets32_half, __pseudo_scatter_base_offsets32_i32,
              __pseudo_scatter_base_offsets32_float, __pseudo_scatter_base_offsets32_i64,
              __pseudo_scatter_base_offsets32_double}},
            {{MemoryOpKind::Scatter, MemoryOpForm::BaseOffsets, MemoryOpFlavor::Pseudo, 64},
             {__pseudo_scatter_base_offsets64_i8, __pseudo_scatter_base_offsets64_i16,
              __pseudo_scatter_base_offsets64_half, __pseudo_scatter_base_offsets64_i32,
              __pseudo_scatter_base_offsets64_float, __pseudo_scatter_base_offsets64_i64,
              __pseudo_scatter_base_offsets64_double}},
            {{MemoryOpKind::Scatter, MemoryOpForm::Factored, MemoryOpFlavor::Pseudo, 32},
             {__pseudo_scatter_factored_base_offsets32_i8, __pseudo_scatter_factored_base_offsets32_i16,
              __pseudo_scatter_factored_base_offsets32_half, __pseudo_scatter_factored_base_offsets32_i32,
              __pseudo_scatter_factored_base_offsets32_float, __pseudo_scatter_factored_base_offsets32_i64,
              __pseudo_scatter_factored_base_offsets32_double}},
            {{MemoryOpKind::Scatter, MemoryOpForm::Factored, MemoryOpFlavor::Pseudo, 64},
             {__pseudo_scatter_factored_base_offsets64_i8, __pseudo_scatter_factored_base_offsets64_i16,
              __pseudo_scatter_factored_base_offsets64_half, __pseudo_scatter_factored_base_offsets64_i32,
              __pseudo_scatter_factored_base_offsets64_float, __pseudo_scatter_factored_base_offsets64_i64,
              __pseudo_scatter_factored_base_offsets64_double}},

            // Their lowerings, selected by ReplacePseudoMemoryOps.
            {{MemoryOpKind::Gather, MemoryOpForm::General, MemoryOpFlavor::Lowered, 32},
             {__gather32_i8, __gather32_i16, __gather32_half, __gather32_i32, __gather32_float, __gather32_i64,
              __gather32_double}},
            {{MemoryOpKind::Gather, MemoryOpForm::General, MemoryOpFlavor::Lowered, 64},
             {__gather64_i8, __gather64_i16, __gather64_half, __gather64_i32, __gather64_float, __gather64_i64,
              __gather64_double}},
            {{MemoryOpKind::Gather, MemoryOpForm::General, MemoryOpFlavor::Generic, 32},
             {__gather32_generic_i8, __gather32_generic_i16, __gather32_generic_half, __gather32_generic_i32,
              __gather32_generic_float, __gather32_generic_i64, __gather32_generic_double}},
            {{MemoryOpKind::Gather, MemoryOpForm::General, MemoryOpFlavor::Generic, 64},
             {__gather64_generic_i8, __gather64_generic_i16, __gather64_generic_half, __gather64_generic_i32,
              __gather64_generic_float, __gather64_generic_i64, __gather64_generic_double}},
            {{MemoryOpKind::Gather, MemoryOpForm::BaseOffsets, MemoryOpFlavor::Lowered, 32},
             {__gather_base_offsets32_i8, __gather_base_offsets32_i16, __gather_base_offsets32_half,
              __gather_base_offsets32_i32, __gather_base_offsets32_float, __gather_base_offsets32_i64,
              __gather_base_offsets32_double}},
            {{MemoryOpKind::Gather, MemoryOpForm::BaseOffsets, MemoryOpFlavor::Lowered, 64},
             {__gather_base_offsets64_i8, __gather_base_offsets64_i16, __gather_base_offsets64_half,
              __gather_base_offsets64_i32, __gather_base_offsets64_float, __gather_base_offsets64_i64,
              __gather_base_offsets64_double}},
            {{MemoryOpKind::Gather, MemoryOpForm::Factored, MemoryOpFlavor::Lowered, 32},
             {__gather_factored_base_offsets32_i8, __gather_factored_base_offsets32_i16,
              __gather_factored_base_offsets32_half, __gather_factored_base_offsets32_i32,
              __gather_factored_base_offsets32_float, __gather_factored_base_offsets32_i64,
              __gather_factored_base_offsets32_double}},
            {{MemoryOpKind::Gather, MemoryOpForm::Factored, MemoryOpFlavor::Lowered, 64},
             {__gather_factored_base_offsets64_i8, __gather_factored_base_offsets64_i16,
              __gather_factored_base_offsets64_half, __gather_factored_base_offsets64_i32,
              __gather_factored_base_offsets64_float, __gather_factored_base_offsets64_i64,
              __gather_factored_base_offsets64_double}},
            {{MemoryOpKind::Scatter, MemoryOpForm::General, MemoryOpFlavor::Lowered, 32},
             {__scatter32_i8, __scatter32_i16, __scatter32_half, __scatter32_i32, __scatter32_float, __scatter32_i64,
              __scatter32_double}},
            {{MemoryOpKind::Scatter, MemoryOpForm::General, MemoryOpFlavor::Lowered, 64},
             {__scatter64_i8, __scatter64_i16, __scatter64_half, __scatter64_i32, __scatter64_float, __scatter64_i64,
              __scatter64_double}},
            {{MemoryOpKind::Scatter, MemoryOpForm::General, MemoryOpFlavor::Generic, 32},
             {__scatter32_generic_i8, __scatter32_generic_i16, __scatter32_generic_half, __scatter32_generic_i32,
              __scatter32_generic_float, __scatter32_generic_i64, __scatter32_generic_double}},
            {{MemoryOpKind::Scatter, MemoryOpForm::General, MemoryOpFlavor::Generic, 64},
             {__scatter64_generic_i8, __scatter64_generic_i16, __scatter64_generic_half, __scatter64_generic_i32,
              __scatter64_generic_float, __scatter64_generic_i64, __scatter64_generic_double}},
            {{MemoryOpKind::Scatter, MemoryOpForm::BaseOffsets, MemoryOpFlavor::Lowered, 32},
             {__scatter_base_offsets32_i8, __scatter_base_offsets32_i16, __scatter_base_offsets32_half,
              __scatter_base_offsets32_i32, __scatter_base_offsets32_float, __scatter_base_offsets32_i64,
              __scatter_base_offsets32_double}},
            {{MemoryOpKind::Scatter, MemoryOpForm::BaseOffsets, MemoryOpFlavor::Lowered, 64},
             {__scatter_base_offsets64_i8, __scatter_base_offsets64_i16, __scatter_base_offsets64_half,
              __scatter_base_offsets64_i32, __scatter_base_offsets64_float, __scatter_base_offsets64_i64,
              __scatter_base_offsets64_double}},
            {{MemoryOpKind::Scatter, MemoryOpForm::Factored, MemoryOpFlavor::Lowered, 32},
             {__scatter_factored_base_offsets32_i8, __scatter_factored_base_offsets32_i16,
              __scatter_factored_base_offsets32_half, __scatter_factored_base_offsets32_i32,
              __scatter_factored_base_offsets32_float, __scatter_factored_base_offsets32_i64,
              __scatter_factored_base_offsets32_double}},
            {{MemoryOpKind::Scatter, MemoryOpForm::Factored, MemoryOpFlavor::Lowered, 64},
             {__scatter_factored_base_offsets64_i8, __scatter_factored_base_offsets64_i16,
              __scatter_factored_base_offsets64_half, __scatter_factored_base_offsets64_i32,
              __scatter_factored_base_offsets64_float, __scatter_factored_base_offsets64_i64,
              __scatter_factored_base_offsets64_double}},

            // Masked loads and stores of consecutive lanes. There is no
            // pseudo masked load; __masked_load_blend_* is defined for Xe only.
            {{MemoryOpKind::MaskedLoad, MemoryOpForm::Pointer, MemoryOpFlavor::Lowered, 0},
             {__masked_load_i8, __masked_load_i16, __masked_load_half, __masked_load_i32, __masked_load_float,
              __masked_load_i64, __masked_load_double}},
            {{MemoryOpKind::MaskedLoad, MemoryOpForm::Pointer, MemoryOpFlavor::Blend, 0},
             {__masked_load_blend_i8, __masked_load_blend_i16, __masked_load_blend_half, __masked_load_blend_i32,
              __masked_load_blend_float, __masked_load_blend_i64, __masked_load_blend_double}},
            {{MemoryOpKind::MaskedStore, MemoryOpForm::Pointer, MemoryOpFlavor::Pseudo, 0},
             {__pseudo_masked_store_i8, __pseudo_masked_store_i16, __pseudo_masked_store_half,
              __pseudo_masked_store_i32, __pseudo_masked_store_float, __pseudo_masked_store_i64,
              __pseudo_masked_store_double}},
            {{MemoryOpKind::MaskedStore, MemoryOpForm::Pointer, MemoryOpFlavor::Lowered, 0},
             {__masked_store_i8, __masked_store_i16, __masked_store_half, __masked_store_i32, __masked_store_float,
              __masked_store_i64, __masked_store_double}},
            {{MemoryOpKind::MaskedStore, MemoryOpForm::Pointer, MemoryOpFlavor::Blend, 0},
             {__masked_store_blend_i8, __masked_store_blend_i16, __masked_store_blend_half, __masked_store_blend_i32,
              __masked_store_blend_float, __masked_store_blend_i64, __masked_store_blend_double}},
        };

        // Prefetches take i64 addresses on every target (stdlib.ispc casts
        // the pointer to int64) and have no element type.
        const PrefetchInfo prefetches[] = {
            {MemoryOpFlavor::Pseudo, PrefetchHint::Read1, __pseudo_prefetch_read_varying_1},
            {MemoryOpFlavor::Pseudo, PrefetchHint::Read2, __pseudo_prefetch_read_varying_2},
            {MemoryOpFlavor::Pseudo, PrefetchHint::Read3, __pseudo_prefetch_read_varying_3},
            {MemoryOpFlavor::Pseudo, PrefetchHint::ReadNT, __pseudo_prefetch_read_varying_nt},
            {MemoryOpFlavor::Pseudo, PrefetchHint::Write1, __pseudo_prefetch_write_varying_1},
            {MemoryOpFlavor::Pseudo, PrefetchHint::Write2, __pseudo_prefetch_write_varying_2},
            {MemoryOpFlavor::Pseudo, PrefetchHint::Write3, __pseudo_prefetch_write_varying_3},
            {MemoryOpFlavor::Lowered, PrefetchHint::Read1, __prefetch_read_varying_1},
            {MemoryOpFlavor::Lowered, PrefetchHint::Read2, __prefetch_read_varying_2},
            {MemoryOpFlavor::Lowered, PrefetchHint::Read3, __prefetch_read_varying_3},
            {MemoryOpFlavor::Lowered, PrefetchHint::ReadNT, __prefetch_read_varying_nt},
            {MemoryOpFlavor::Lowered, PrefetchHint::Write1, __prefetch_write_varying_1},
            {MemoryOpFlavor::Lowered, PrefetchHint::Write2, __prefetch_write_varying_2},
            {MemoryOpFlavor::Lowered, PrefetchHint::Write3, __prefetch_write_varying_3},
        };

        ops.reserve(std::size(families) * kElements.size() + std::size(prefetches));
        for (const Family &f : families) {
            for (size_t i = 0; i < kElements.size(); ++i) {
                // A name in the wrong column would give the builtin another
                // element's key.
                Assert(llvm::StringRef(f.names[i]).ends_with(kElements[i].suffix));
                MemoryOpKey key;
                key.kind = f.key.kind;
                key.form = f.key.form;
                key.element = kElements[i].element;
                key.flavor = f.key.flavor;
                key.offsetBits = f.key.offsetBits;
                ops.push_back({f.names[i], key, lLayout(key.kind, key.form)});
            }
        }
        for (const PrefetchInfo &p : prefetches) {
            MemoryOpKey key;
            key.kind = MemoryOpKind::Prefetch;
            key.form = MemoryOpForm::General;
            key.flavor = p.flavor;
            key.offsetBits = 64;
            key.hint = p.hint;
            ops.push_back({p.name, key, lLayout(key.kind, key.form)});
        }

        for (const MemoryOp &op : ops) {
            bool newName = byName.try_emplace(op.name, &op).second;
            bool newKey = byKey.try_emplace(lPack(op.key), &op).second;
            Assert(newName && newKey);
        }
    }
};

const Registry &lRegistry() {
    static const Registry registry;
    return registry;
}

} // namespace

const MemoryOp *LookupMemoryOp(llvm::StringRef name) {
    const Registry &r = lRegistry();
    auto it = r.byName.find(name);
    return it == r.byName.end() ? nullptr : it->second;
}

const MemoryOp *FindMemoryOp(const MemoryOpKey &key) {
    const Registry &r = lRegistry();
    auto it = r.byKey.find(lPack(key));
    return it == r.byKey.end() ? nullptr : it->second;
}

llvm::Function *GetMemoryOpFunction(llvm::Module *M, const MemoryOpKey &key) {
    const MemoryOp *op = FindMemoryOp(key);
    return op == nullptr ? nullptr : M->getFunction(op->name);
}

llvm::Type *GetMemoryElementType(MemoryElement element) {
    for (const ElementInfo &e : kElements) {
        if (e.element == element) {
            return *e.type;
        }
    }
    return nullptr;
}

unsigned GetMemoryElementBytes(MemoryElement element) {
    llvm::Type *type = GetMemoryElementType(element);
    return type == nullptr ? 0 : type->getPrimitiveSizeInBits() / 8;
}

bool HasMemoryOpSignature(const MemoryOp &op, llvm::FunctionType *type) {
    const MemoryOpLayout &layout = op.layout;
    if (type->isVarArg() || type->getNumParams() != layout.numArgs) {
        return false;
    }
    auto param = [type](int8_t index) { return type->getParamType(index); };

    // The mask gives the lane count; every per-lane operand must agree with it.
    llvm::FixedVectorType *maskType = llvm::dyn_cast<llvm::FixedVectorType>(param(layout.mask));
    if (maskType == nullptr || !maskType->getElementType()->isIntegerTy()) {
        return false;
    }
    const unsigned lanes = maskType->getNumElements();
    if (lanes > ISPC_MAX_NVEC) {
        return false;
    }
    if (layout.addresses >= 0 && !lIsIntVector(param(layout.addresses), lanes, op.offsetBits())) {
        return false;
    }
    if (layout.base >= 0 && !param(layout.base)->isPointerTy()) {
        return false;
    }
    if (layout.scale >= 0 && !param(layout.scale)->isIntegerTy(32)) {
        return false;
    }
    if (layout.offsets >= 0 && !lIsIntVector(param(layout.offsets), lanes, op.offsetBits())) {
        return false;
    }
    if (layout.delta >= 0 && param(layout.delta) != param(layout.offsets)) {
        return false;
    }
    llvm::Type *element = GetMemoryElementType(op.element());
    if (layout.value >= 0 && !lIsElementVector(param(layout.value), lanes, element)) {
        return false;
    }
    if (op.kind() == MemoryOpKind::Gather || op.kind() == MemoryOpKind::MaskedLoad) {
        return lIsElementVector(type->getReturnType(), lanes, element);
    }
    return type->getReturnType()->isVoidTy();
}

void VerifyMemoryOpDeclarations(llvm::Module *M) {
    for (const MemoryOp &op : lRegistry().ops) {
        llvm::Function *f = M->getFunction(op.name);
        if (f != nullptr && !HasMemoryOpSignature(op, f->getFunctionType())) {
            std::string message = std::string("Builtin ") + op.name +
                                  " does not have the signature MemoryOpRegistry expects for its form";
            FATAL(message.c_str());
        }
    }
}

} // namespace ispc
