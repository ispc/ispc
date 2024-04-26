/*
  Copyright (c) 2022-2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#pragma once

#include "ISPCPass.h"

#ifdef ISPC_XE_ENABLED

namespace ispc {
/*
 * MemoryCoalescing: basis for store/load coalescing optimizations.
 *
 * Memory coalescing tries to merge several memory operations into
 * wider ones. There are two different flows for store and load coalescings.
 *
 * This is a function pass. Each BB is handled separately.
 *
 * The general idea of optimization is to iterate over BB and analyse pointers
 * for all optimization targets (target is an instruction that can be optimized).
 * Instructions that uses the same pointer can be coalesced.
 *
 * Each memory instruction has a pointer operand. To perform coalescing,
 * we need to determine its base pointer and offset for it. This is done
 * via recurrent algorithm. Only constant offset can be handled.
 * TODO: found in experiments: sometimes pointer is passed as null
 * and actual pointer came via calculated offset. Such situation is not
 * handled now (Target: Xe-TPM).
 *
 * Sometimies coalescing can fail in case when it is not sure if the
 * transformation is actually safe. This can happen if some "bad"
 * instructions met between optimization targets. To handle this, all
 * optimization targets are collected in several blocks. They also can be
 * treated as non-interfering fragments of BB. Each block can be
 * safely optimized. Bad instructions trigger creation of such blocks.
 * For now, stopper detection is implemented in a general way for all
 * types of coalescing. It can be changed in future.
 *
 * Basicly, implementation can treat offset in different ways. Afterall,
 * implementation builds accesses for optimized instructions. All
 * offsets are presented in bytes now.
 *
 * To create new coalescing type, one should implement several handlers.
 * More info about them below.
 *
 * There are also some helpers that are not used by optimization
 * directly, but can be used by several coalescing implementations.
 * Such helpers contains some general stuff.
 */
struct MemoryCoalescing : public llvm::PassInfoMixin<MemoryCoalescing> {

    // Optimization runner
    llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);

  protected:
    typedef int64_t OffsetT;
    typedef llvm::SmallVector<OffsetT, 16> OffsetsVecT;
    typedef llvm::SmallVector<llvm::Instruction *, 4> BlockInstsVecT;

    // Available coalescing types
    enum class MemType { OPT_LOAD, OPT_STORE };

    // Inst optimization data
    class InstData {
      public:
        // Instruction itself
        llvm::Instruction *Inst;
        // Offsets for memory accesses
        OffsetsVecT Offsets;
        // Unused for loads, stored value for stores
        llvm::Value *Val;
        InstData() = delete;
        InstData(llvm::Instruction *Inst, OffsetsVecT &Offsets, llvm::Value *Val)
            : Inst(Inst), Offsets(Offsets), Val(Val) {}
    };

    // Ptr optimization data
    class PtrData {
      public:
        // The insertion point for the ptr users.
        // All instructions will be placed before it.
        // InsertPoint should be the first instruction in block for load coalescing
        // and the last one for store coalescing. This is achieved by different
        // traversal in both types of optimization.
        llvm::Instruction *InsertPoint;
        // Data for insts that use this ptr
        std::vector<InstData> InstsData;
        PtrData() : InsertPoint(nullptr) {}
        void addInstruction(llvm::Instruction *Inst, OffsetsVecT &Offsets, llvm::Value *Val) {
            InstsData.push_back(InstData(Inst, Offsets, Val));
            if (!InsertPoint)
                InsertPoint = Inst;
        }
    };

    // Data for optimization block
    class OptBlock {
      public:
        // Data for all ptrs in the block
        std::unordered_map<llvm::Value *, PtrData> PtrsData;
        void addInstruction(llvm::Instruction *Inst, llvm::Value *Ptr, OffsetsVecT &Offsets,
                            llvm::Value *Val = nullptr) {
            PtrsData[Ptr].addInstruction(Inst, Offsets, Val);
        }
    };

    // Instructions that are possibly dead. It includes some newly created instructions that
    // are created during optimization and memory instructions that are optimized out. All
    // instructions from this list are treated as dead if they don't have any users.
    std::set<llvm::Instruction *> PossiblyDead;
    // Modification flag
    bool modifiedAny = false;
    // Address space for optimization
    // TODO: not obvious how it would work with TPM on Xe now.
    AddressSpace AddrSpace;

  private:
    // Info for complex GEPs
    struct GEPVarOffsetInfo {
        llvm::Instruction *GEP;
        llvm::User::value_op_iterator FirstConstUse;
        GEPVarOffsetInfo(llvm::Instruction *GEP) : GEP(GEP) {}
    };

    // DenseMap helper for complex GEPs
    struct DenseMapInfo {
        static inline GEPVarOffsetInfo getEmptyKey() {
            return GEPVarOffsetInfo(llvm::DenseMapInfo<llvm::Instruction *>::getEmptyKey());
        }
        static inline GEPVarOffsetInfo getTombstoneKey() {
            return GEPVarOffsetInfo(llvm::DenseMapInfo<llvm::Instruction *>::getTombstoneKey());
        }
        static inline bool isSentinel(const GEPVarOffsetInfo &Val) {
            return Val.GEP == getEmptyKey().GEP || Val.GEP == getTombstoneKey().GEP;
        }
        static unsigned getHashValue(const GEPVarOffsetInfo &Val) {
            return hash_combine_range(Val.GEP->value_op_begin(), Val.FirstConstUse);
        }
        static bool isEqual(const GEPVarOffsetInfo &LHS, const GEPVarOffsetInfo &RHS) {
            if (isSentinel(LHS) || isSentinel(RHS))
                return LHS.GEP == RHS.GEP;

            for (auto lhs_it = LHS.GEP->value_op_begin(), rhs_it = RHS.GEP->value_op_begin(), lhs_e = LHS.FirstConstUse,
                      rhs_e = RHS.FirstConstUse;
                 lhs_it != lhs_e || rhs_it != rhs_e; ++lhs_it, ++rhs_it) {
                if (lhs_it == lhs_e || rhs_it == rhs_e)
                    return false;
                if (*lhs_it != *rhs_it)
                    return false;
            }

            return true;
        }
    };

    // Helper for ptr offset analysis. It holds
    // Ptr+Offset data. IsConstantOffset is used to
    // check if optimization is appliable for such ptr.
    // Ptr can be null, in this case only Offset
    // matters. This is used for arithmetic analysis.
    // The data is consistent only if the IsConstantOffset
    // flag is true.
    struct BasePtrInfo {
        llvm::Value *Ptr;
        OffsetT Offset;
        bool IsConstantOffset;

        BasePtrInfo() : Ptr(nullptr), Offset(0), IsConstantOffset(false) {}
    };

    // Coalescing type
    const MemType OptType;
    // Cached data for visited ptr instructions
    std::unordered_map<llvm::Value *, BasePtrInfo> BasePtrInfoCache;
    // Cached data for complex GEPs partial replacements
    llvm::DenseMap<GEPVarOffsetInfo, llvm::Value *, DenseMapInfo> ComplexGEPsInfoCache;
    // Blocks to be optimized
    std::vector<OptBlock> Blocks;

    // Find base ptr and its offset
    BasePtrInfo findBasePtr(llvm::Value *PtrOperand);
    // Analyse GEP that has variable offset. This is used by findBasePtr.
    // Such GEPs can have same variable part while the final (different) part is constant.
    BasePtrInfo analyseVarOffsetGEP(llvm::GetElementPtrInst *GEP);
    // Analyse arithmetics. That allows to handle more cases when offset is calculated not via GEP.
    // TODO: not implemented now, returns result with IsConstantOffset=false.
    BasePtrInfo analyseArithmetics(llvm::BinaryOperator *Arithm);
    // Return true if Inst blocks further optimization of currently collected
    // optimization targets. This is a stopper for collecting
    // instructions.
    bool stopsOptimization(llvm::Instruction *Inst) const;
    // Add Block to blocks list and return new one.
    OptBlock finishBlock(OptBlock &Block);
    // Collect data for optimization
    void analyseInsts(llvm::BasicBlock &BB);
    // Apply coalescing
    void applyOptimization();
    // Delete dead instructions
    void deletePossiblyDeadInsts();
    // Reset all internal structures
    void clear();

  protected:
    // Initialization
    MemoryCoalescing(MemType OptType, AddressSpace AddrSpace) : AddrSpace(AddrSpace), OptType(OptType) {}

    /* ------ Handlers ------ */
    // Methods in this block are interface for different coalescing types.

    // Return true if coalescing can handle Inst.
    virtual bool isOptimizationTarget(llvm::Instruction *Inst) const = 0;
    // Return pointer value or null if there is no one. This should handle
    // all optimization targets.
    virtual llvm::Value *getPointer(llvm::Instruction *Inst) const = 0;
    // Return offset for implied values. Scatter and gathers, for example,
    // can have vectorized offset, so the result is a list. If the offset
    // is not constant, return empty list. This should handle all optimization
    // targets.
    virtual OffsetsVecT getOffset(llvm::Instruction *Inst) const = 0;
    // Return value being stored. For load coalescing, simply return null.
    // This function is not called under load coalescing. For store coalescing,
    // this should handle all optimizations target.
    virtual llvm::Value *getStoredValue(llvm::Instruction *Inst) const = 0;
    // Perform optimization on ptr data.
    virtual void optimizePtr(llvm::Value *Ptr, PtrData &PD, llvm::Instruction *InsertPoint) = 0;

    // TODO: this handler must call runOnBasicBlockImpl to run optimization.
    // This function is needed due to DEBUG_START/END_PASS logic. Maybe there is
    // a better way to solve it.
    virtual bool runOnBasicBlock(llvm::BasicBlock &BB) = 0;
    bool runOnBasicBlockImpl(llvm::BasicBlock &BB);

    /* ------- Helpers ------ */
    // Methods in this block are not used in optimization directly
    // and can be invoked by handlers implementations

    // Collect constant offsets from vector. If offset is not a constant, empty list
    // is returned.
    OffsetsVecT getConstOffsetFromVector(llvm::Value *VecVal) const;
    // Multiplies all elements by scale.
    void applyScale(OffsetsVecT &Offsets, OffsetT scale) const;
    // Build InsertElementInst. This is offset based insertion and it can deal with mixed types.
    llvm::Value *buildIEI(llvm::Value *InsertTo, llvm::Value *Val, OffsetT OffsetBytes,
                          llvm::Instruction *InsertBefore) const;
    // Build ExtractElementInst. This is offset based extraction and it can deal with mixed types.
    llvm::Value *buildEEI(llvm::Value *ExtractFrom, OffsetT OffsetBytes, llvm::Type *DstTy,
                          llvm::Instruction *InsertBefore) const;
    // Build Cast (BitCast or IntToPtr)
    llvm::Value *buildCast(llvm::Value *Val, llvm::Type *DstTy, llvm::Instruction *InsertBefore) const;
    // Extract element from block values. Can aggregate value from several block instructions.
    llvm::Value *extractValueFromBlock(const BlockInstsVecT &BlockInstsVec, OffsetT OffsetBytes, llvm::Type *DstTy,
                                       llvm::Instruction *InsertBefore) const;
    // Get scalar type size in bytes
    unsigned getScalarTypeSize(llvm::Type *Ty) const;
};

struct XeGatherCoalescing : public MemoryCoalescing {

    XeGatherCoalescing() : MemoryCoalescing(MemoryCoalescing::MemType::OPT_LOAD, AddressSpace::ispc_global) {}

  private:
    bool isOptimizationTarget(llvm::Instruction *Inst) const;
    llvm::Value *getPointer(llvm::Instruction *Inst) const;
    OffsetsVecT getOffset(llvm::Instruction *Inst) const;
    llvm::Value *getStoredValue(llvm::Instruction *Inst) const { return nullptr; }
    void optimizePtr(llvm::Value *Ptr, PtrData &PD, llvm::Instruction *InsertPoint);
    bool runOnBasicBlock(llvm::BasicBlock &BB);

    llvm::CallInst *getPseudoGatherConstOffset(llvm::Instruction *Inst) const;
    bool isConstOffsetPseudoGather(llvm::CallInst *CI) const;
};

} // namespace ispc

#endif
