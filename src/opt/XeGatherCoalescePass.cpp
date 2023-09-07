/*
  Copyright (c) 2022-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include "XeGatherCoalescePass.h"

#ifdef ISPC_XE_ENABLED

namespace ispc {

// Optimization runner
llvm::PreservedAnalyses MemoryCoalescing::run(llvm::Function &Fn, llvm::FunctionAnalysisManager &FAM) {
    llvm::TimeTraceScope FuncScope("MemoryCoalescing::run", Fn.getName());
    bool modifiedAny = false;
    for (llvm::BasicBlock &BB : Fn) {
        modifiedAny |= runOnBasicBlock(BB);
    }

    if (!modifiedAny) {
        // No changes, all analyses are preserved.
        return llvm::PreservedAnalyses::all();
    }

    llvm::PreservedAnalyses PA;
    PA.preserveSet<llvm::CFGAnalyses>();
    return PA;
}

// Find base pointer info for ptr operand.
MemoryCoalescing::BasePtrInfo MemoryCoalescing::findBasePtr(llvm::Value *PtrOperand) {
    // Look for previously handled value
    auto it = BasePtrInfoCache.find(PtrOperand);
    if (it != BasePtrInfoCache.end())
        return it->second;

    BasePtrInfo res;
    if (auto BCI = llvm::dyn_cast<llvm::BitCastInst>(PtrOperand)) {
        res = findBasePtr(BCI->getOperand(0));
    } else if (auto GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(PtrOperand)) {
        if (GEP->hasAllConstantIndices()) {
            // Easy case. Collect offset.
            llvm::APInt acc(g->target->is32Bit() ? 32 : 64, 0, true);
            bool checker = GEP->accumulateConstantOffset(GEP->getModule()->getDataLayout(), acc);
            Assert(checker);

            // Run this analysis for GEP's ptr operand to handle possible bypasses
            res = findBasePtr(GEP->getPointerOperand());
            res.Offset += acc.getSExtValue();
        } else {
            // Bad case. We need partialy use the data in this GEP - there might be
            // intersections with others. The handler for this case is implemented
            // in a separate method.
            res = analyseVarOffsetGEP(GEP);
        }
    } else if (auto Arithm = llvm::dyn_cast<llvm::BinaryOperator>(PtrOperand)) {
        // Handle arithmetics
        res = analyseArithmetics(Arithm);
    } else if (llvm::isa<llvm::PointerType>(PtrOperand->getType())) {
        // This block is reached when no possible bypasses are left.
        // Use this ptr as a base
        res.Ptr = PtrOperand;
        res.Offset = 0;
        res.IsConstantOffset = true;
    } else if (auto Const = llvm::dyn_cast<llvm::ConstantInt>(PtrOperand)) {
        // Constant operand - set offset.
        res.Ptr = nullptr;
        res.Offset = Const->getSExtValue();
        res.IsConstantOffset = true;
    } else {
        // That is a non-constant offset
        res.IsConstantOffset = false;
    }

    // Finally, cache result and return
    BasePtrInfoCache[PtrOperand] = res;
    return res;
}

// Analyse GEP that has some indices non-constant.
// TODO: Consider adding full chain analysis. Decided to leave this idea for now
// because it may introduce many redundant address calculations: current approach
// would copy full chain even if some common part could be calculated somewhere eariler.
MemoryCoalescing::BasePtrInfo MemoryCoalescing::analyseVarOffsetGEP(llvm::GetElementPtrInst *GEP) {
    BasePtrInfo res;
    // Find the last constant idxs: we may be able to use them as constant offset for
    // several GEPs with common pre-constant part
    auto FirstConstIdx = GEP->getNumOperands();
    Assert(FirstConstIdx);
    unsigned i = FirstConstIdx - 1;
    do {
        if (!llvm::isa<llvm::ConstantInt>(GEP->getOperand(i)))
            break;
        FirstConstIdx = i;
    } while (i--);
    // Early return in case when no constant offset was found:
    // further actions are useless.
    if (GEP->getNumOperands() == FirstConstIdx) {
        res.Ptr = GEP;
        res.Offset = 0;
        res.IsConstantOffset = true;
        return res;
    }
    // Initialize var offset ops info.
    auto FirstConstUse = GEP->value_op_begin();
    for (unsigned i = 0; i < FirstConstIdx; ++i)
        FirstConstUse++;
    GEPVarOffsetInfo GEPVarOffsetData(GEP);
    GEPVarOffsetData.FirstConstUse = FirstConstUse;
    // Try to find existing dangling PartialGEP for such Ops combination
    // or create new one.
    auto DanglingGEP_it = ComplexGEPsInfoCache.find(GEPVarOffsetData);
    if (DanglingGEP_it == ComplexGEPsInfoCache.end()) {
        std::vector<llvm::Value *> PartialIdxs;
        for (unsigned i = 1; i < FirstConstIdx; ++i)
            PartialIdxs.push_back(GEP->getOperand(i));
        llvm::Value *tPtr = GEP->getPointerOperand();
        llvm::Type *tType = GEP->getSourceElementType();
        auto ret = ComplexGEPsInfoCache.insert(
            {GEPVarOffsetData, llvm::GetElementPtrInst::Create(tType, tPtr, PartialIdxs, "partial_gep")});
        DanglingGEP_it = ret.first;
    }
    llvm::Value *DanglingGEP = DanglingGEP_it->second;
    // Collect idxs. We will use them to find offset. Push zero constant first:
    // this will be needed for correct offset accumulation further.
    std::vector<llvm::Value *> Idxs = {llvm::ConstantInt::get(LLVMTypes::Int32Type, 0)};
    for (unsigned i = FirstConstIdx; i < GEP->getNumOperands(); ++i) {
        Idxs.push_back(GEP->getOperand(i));
    }
    // Get partial GEP type
    llvm::PointerType *PartialType = llvm::cast<llvm::PointerType>(DanglingGEP->getType());

    // Create temporary GEP that will help us to get some useful info
    llvm::Value *tPtr = llvm::ConstantPointerNull::get(PartialType);
    Assert(llvm::cast<llvm::GetElementPtrInst>(DanglingGEP));
    llvm::Type *tType = llvm::cast<llvm::GetElementPtrInst>(DanglingGEP)->getSourceElementType();

    llvm::GetElementPtrInst *GEPHelper = llvm::GetElementPtrInst::Create(tType, tPtr, Idxs);
    // Accumulate offset from helper
    llvm::APInt acc(g->target->is32Bit() ? 32 : 64, 0, true);
    bool checker = GEPHelper->accumulateConstantOffset(GEP->getModule()->getDataLayout(), acc);
    Assert(checker);
    // Finally, store data.
    res.Ptr = DanglingGEP;
    res.Offset = acc.getSExtValue();
    res.IsConstantOffset = true;

    return res;
}

// Analyse arithmetic calculations on pointer.
// TODO: not implemented, returns stopper
MemoryCoalescing::BasePtrInfo MemoryCoalescing::analyseArithmetics(llvm::BinaryOperator *Arithm) {
    BasePtrInfo res;
    res.IsConstantOffset = false;
    return res;
}

// Basic block optimization runner.
// TODO: runOnBasicBlock must call it to run optimization. See comment above.
bool MemoryCoalescing::runOnBasicBlockImpl(llvm::BasicBlock &BB) {
    analyseInsts(BB);
    applyOptimization();
    deletePossiblyDeadInsts();
    clear();
    return modifiedAny;
}

void MemoryCoalescing::deletePossiblyDeadInsts() {
    for (auto *Inst : PossiblyDead) {
        if (Inst->use_empty())
            Inst->eraseFromParent();
    }
}

void MemoryCoalescing::clear() {
    BasePtrInfoCache.clear();
    ComplexGEPsInfoCache.clear();
    PossiblyDead.clear();
    Blocks.clear();
}

// Analyse instructions in BB. Gather them into optimizable blocks.
void MemoryCoalescing::analyseInsts(llvm::BasicBlock &BB) {
    auto bi = BB.begin(), be = BB.end();
    auto rbi = BB.rbegin(), rbe = BB.rend();
    Assert(OptType == MemType::OPT_LOAD || OptType == MemType::OPT_STORE);
    OptBlock CurrentBlock;
    for (; (OptType == MemType::OPT_LOAD) ? (bi != be) : (rbi != rbe);) {
        llvm::Instruction *Inst = nullptr;
        if (OptType == MemType::OPT_LOAD) {
            Inst = &*bi;
            ++bi;
        } else {
            Inst = &*rbi;
            ++rbi;
        }

        if (isOptimizationTarget(Inst)) {
            // Find ptr and offsets
            BasePtrInfo BasePtr = findBasePtr(getPointer(Inst));
            OffsetsVecT Offsets = getOffset(Inst);
            if (BasePtr.IsConstantOffset && BasePtr.Ptr && !Offsets.empty()) {
                if (OptType == MemType::OPT_STORE && !CurrentBlock.PtrsData.empty()) {
                    Assert(CurrentBlock.PtrsData.size() == 1 && "Store coalescing can handle only one pointer at once");
                    if (CurrentBlock.PtrsData.find(BasePtr.Ptr) == CurrentBlock.PtrsData.end()) {
                        // Finish current block so the instruction is added to the new one
                        CurrentBlock = finishBlock(CurrentBlock);
                    }
                }

                // Recalculate offsets for BasePtr
                for (auto &Offset : Offsets)
                    Offset += BasePtr.Offset;
                // Add inst to block
                CurrentBlock.addInstruction(Inst, BasePtr.Ptr, Offsets,
                                            OptType == MemType::OPT_STORE ? getStoredValue(Inst) : nullptr);
                // Instruction that was added to the block won't block optimization
                continue;
            }
        }

        if (stopsOptimization(Inst)) {
            // Add current block and create new one
            CurrentBlock = finishBlock(CurrentBlock);
        }
    }

    // Add current block
    Blocks.push_back(CurrentBlock);
}

// Apply optimization for all optimizable blocks.
void MemoryCoalescing::applyOptimization() {
    for (auto &Block : Blocks) {
        for (auto &PD : Block.PtrsData) {
            // Apply dangling GEP
            if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(PD.first))
                if (!GEP->getParent())
                    GEP->insertBefore(PD.second.InsertPoint);
            // Run optimization
            optimizePtr(PD.first, PD.second, PD.second.InsertPoint);
        }
    }
}

// Check whether instruction blocks optimization.
// Optimization target checker should be called before because any store
// blocks optimization in general case.
bool MemoryCoalescing::stopsOptimization(llvm::Instruction *Inst) const {
    if (OptType == MemType::OPT_LOAD) {
        // For load coalescing, only stores can introduce problems.
        if (auto CI = llvm::dyn_cast<llvm::CallInst>(Inst)) {
            // Such call may introduce load-store-load sequence
            if (!CI->onlyReadsMemory())
                for (unsigned i = 0; i < Inst->getNumOperands(); ++i)
                    if (GetAddressSpace(Inst->getOperand(i)) == AddrSpace)
                        return true;
        } else if (auto SI = llvm::dyn_cast<llvm::StoreInst>(Inst)) {
            // May introduce load-store-load sequence
            if (GetAddressSpace(SI->getPointerOperand()) == AddrSpace)
                return true;
        }
    } else if (OptType == MemType::OPT_STORE) {
        // For store coalescing, both loads and stores can introduce problems.
        if (auto CI = llvm::dyn_cast<llvm::CallInst>(Inst)) {
            // Such call may introduce store-load/store-store sequence
            for (unsigned i = 0; i < Inst->getNumOperands(); ++i) {
                if (GetAddressSpace(Inst->getOperand(i)) == AddrSpace)
                    return true;
            }
        } else if (auto LI = llvm::dyn_cast<llvm::LoadInst>(Inst)) {
            // May introduce store-load-store sequence
            if (GetAddressSpace(LI->getPointerOperand()) == AddrSpace)
                return true;
        } else if (auto SI = llvm::dyn_cast<llvm::StoreInst>(Inst)) {
            // May introduce store-store-store sequence
            if (GetAddressSpace(SI->getPointerOperand()) == AddrSpace)
                return true;
        }
    } else
        Assert(0 && "Bad optimization type");

    return false;
}

// Store current block and init new one.
MemoryCoalescing::OptBlock MemoryCoalescing::finishBlock(MemoryCoalescing::OptBlock &Block) {
    Blocks.push_back(Block);
    return OptBlock();
}

// Get constant offset from vector.
MemoryCoalescing::OffsetsVecT MemoryCoalescing::getConstOffsetFromVector(llvm::Value *VecVal) const {
    Assert(VecVal && llvm::isa<llvm::VectorType>(VecVal->getType()) && "Expected vector type");
    OffsetsVecT res;
    auto ConstVec = llvm::dyn_cast<llvm::ConstantDataVector>(VecVal);

    if (!ConstVec)
        return res;
    for (unsigned i = 0, size = llvm::cast<llvm::FixedVectorType>(ConstVec->getType())->getNumElements(); i < size;
         ++i) {
        auto ConstantInt = llvm::dyn_cast<llvm::ConstantInt>(ConstVec->getElementAsConstant(i));
        if (!ConstantInt) {
            // Actually, we don't expect this to happen
            res.clear();
            break;
        }
        res.push_back(ConstantInt->getSExtValue());
    }

    return res;
}

// Apply scale on offsets.
void MemoryCoalescing::applyScale(MemoryCoalescing::OffsetsVecT &Offsets, MemoryCoalescing::OffsetT Scale) const {
    for (auto &Offset : Offsets)
        Offset *= Scale;
}

unsigned MemoryCoalescing::getScalarTypeSize(llvm::Type *Ty) const {
    Ty = Ty->getScalarType();
    if (Ty->isPointerTy())
        return g->target->is32Bit() ? 4 : 8;
    return Ty->getPrimitiveSizeInBits() >> 3;
}

llvm::Value *MemoryCoalescing::buildIEI(llvm::Value *InsertTo, llvm::Value *Val, MemoryCoalescing::OffsetT OffsetBytes,
                                        llvm::Instruction *InsertBefore) const {
    llvm::Type *ScalarType = InsertTo->getType()->getScalarType();
    llvm::Type *ValTy = Val->getType();

    Assert(!ValTy->isVectorTy() && !ValTy->isAggregateType() && "Expected scalar type");
    Assert(InsertTo->getType()->isVectorTy() && "Expected vector type");

    unsigned ScalarTypeBytes = getScalarTypeSize(ScalarType);
    unsigned ValTyBytes = getScalarTypeSize(ValTy);
    unsigned Idx = OffsetBytes / ScalarTypeBytes;
    unsigned Rem = OffsetBytes % ScalarTypeBytes;
    Idx = Rem ? Idx + 1 : Idx;
    llvm::Value *FinalInsertElement = Val;
    if (ValTy != ScalarType) {
        if (ValTyBytes == ScalarTypeBytes) {
            // Apply cast
            FinalInsertElement = buildCast(Val, ScalarType, InsertBefore);
        } else {
            // Need to create eei-cast-iei-cast chain.
            // Extract scalar type value
            auto *EEI = llvm::ExtractElementInst::Create(InsertTo, llvm::ConstantInt::get(LLVMTypes::Int64Type, Idx),
                                                         "mem_coal_diff_ty_eei", InsertBefore);
            // Cast it to vector of smaller types
            auto *Cast = buildCast(EEI, llvm::FixedVectorType::get(ValTy, ScalarTypeBytes / ValTyBytes), InsertBefore);
            // Insert value into casted type. Do it via this builder so we don't duplicate logic of offset calculations.
            auto *IEI = buildIEI(Cast, Val, Rem, InsertBefore);
            // Cast to original type
            FinalInsertElement = buildCast(IEI, ScalarType, InsertBefore);
        }
    }

    return llvm::InsertElementInst::Create(
        InsertTo, FinalInsertElement, llvm::ConstantInt::get(LLVMTypes::Int64Type, Idx), "mem_coal_iei", InsertBefore);
}

llvm::Value *MemoryCoalescing::buildCast(llvm::Value *Val, llvm::Type *DstTy, llvm::Instruction *InsertBefore) const {
    // No cast needed: early return
    if (Val->getType() == DstTy)
        return Val;

    if (DstTy->isPointerTy() && !Val->getType()->isPointerTy()) {
        return new llvm::IntToPtrInst(Val, DstTy, "coal_diff_ty_ptr_cast", InsertBefore);
    } else if (!DstTy->isPointerTy() && Val->getType()->isPointerTy()) {
        auto *PtrToInt = new llvm::PtrToIntInst(Val, g->target->is32Bit() ? LLVMTypes::Int32Type : LLVMTypes::Int64Type,
                                                "coal_diff_ty_ptr_cast", InsertBefore);
        return buildCast(PtrToInt, DstTy, InsertBefore);
    } else {
        return new llvm::BitCastInst(Val, DstTy, "coal_diff_ty_cast", InsertBefore);
    }
}

llvm::Value *MemoryCoalescing::buildEEI(llvm::Value *ExtractFrom, MemoryCoalescing::OffsetT OffsetBytes,
                                        llvm::Type *DstTy, llvm::Instruction *InsertBefore) const {
    Assert(!DstTy->isVectorTy() && !DstTy->isAggregateType() && "Expected scalar type");
    Assert(ExtractFrom->getType()->isVectorTy() && "Expected vector type");

    llvm::Type *ScalarType = ExtractFrom->getType()->getScalarType();
    unsigned ScalarTypeBytes = getScalarTypeSize(ScalarType);
    unsigned DstTyBytes = getScalarTypeSize(DstTy);
    unsigned Idx = OffsetBytes / ScalarTypeBytes;
    unsigned Rem = OffsetBytes % ScalarTypeBytes;
    llvm::Value *Res = nullptr;

    if (Rem != 0) {
        // Unaligned case: the resulting value starts inside Idx element.
        // TODO: this is handled via shuffle vector. Actually, it can be done
        // for all cases, but it is possible that such shuffle vector would
        // introduce redundant instructions. This should be investigated
        // at least on Xe target.

        // Cast source to byte vector
        Res = buildCast(
            ExtractFrom,
            llvm::FixedVectorType::get(LLVMTypes::Int8Type,
                                       ScalarTypeBytes *
                                           llvm::cast<llvm::FixedVectorType>(ExtractFrom->getType())->getNumElements()),
            InsertBefore);
        // Prepare Idxs vector for shuffle vector
        std::vector<unsigned int> ByteIdxs(DstTyBytes);
        OffsetT CurrIdx = OffsetBytes;
        for (auto &Val : ByteIdxs)
            Val = CurrIdx++;
        llvm::ArrayRef<unsigned int> ByteIdxsArg(ByteIdxs);
        // Extract bytes via shuffle vector
        Res = new llvm::ShuffleVectorInst(Res, llvm::UndefValue::get(Res->getType()),
                                          llvm::ConstantDataVector::get(*g->ctx, ByteIdxsArg), "coal_unaligned_loader",
                                          InsertBefore);
        // Cast byte vector to scalar value
        Res = buildCast(Res, llvm::IntegerType::get(*g->ctx, /* NumBits */ DstTyBytes << 3), InsertBefore);
        // Cast to actual DstTy
        return buildCast(Res, DstTy, InsertBefore);
    }

    Res = llvm::ExtractElementInst::Create(ExtractFrom, llvm::ConstantInt::get(LLVMTypes::Int64Type, Idx),
                                           "mem_coal_eei", InsertBefore);
    if (DstTy == ScalarType) {
        // Done here
        return Res;
    } else if (DstTyBytes == ScalarTypeBytes) {
        // Just create bitcast
        return buildCast(Res, DstTy, InsertBefore);
    }

    // Smaller type case. Need to insert cast-eei chain.
    auto Cast = buildCast(Res, llvm::FixedVectorType::get(DstTy, ScalarTypeBytes / DstTyBytes), InsertBefore);
    // Now call the builder with adjusted types
    return buildEEI(Cast, Rem, DstTy, InsertBefore);
}

llvm::Value *MemoryCoalescing::extractValueFromBlock(const MemoryCoalescing::BlockInstsVecT &BlockInstsVec,
                                                     MemoryCoalescing::OffsetT OffsetBytes, llvm::Type *DstTy,
                                                     llvm::Instruction *InsertBefore) const {
    Assert(BlockInstsVec.size() > 0);
    OffsetT ScalarTypeSize = getScalarTypeSize(BlockInstsVec[0]->getType());
    OffsetT BlockSizeInBytes =
        ScalarTypeSize * llvm::cast<llvm::FixedVectorType>(BlockInstsVec[0]->getType())->getNumElements();
    unsigned StartIdx = OffsetBytes / BlockSizeInBytes;
    unsigned EndIdx = (OffsetBytes + getScalarTypeSize(DstTy) - 1) / BlockSizeInBytes;
    unsigned BlocksAffected = EndIdx - StartIdx + 1;
    Assert(EndIdx < BlockInstsVec.size());
    if (BlocksAffected == 1) {
        // Simple case: just get value needed
        return buildEEI(BlockInstsVec[StartIdx], OffsetBytes % BlockSizeInBytes, DstTy, InsertBefore);
    } else {
        // Need to get value from several blocks
        llvm::Value *ByteVec =
            llvm::UndefValue::get(llvm::FixedVectorType::get(LLVMTypes::Int8Type, getScalarTypeSize(DstTy)));
        for (OffsetT CurrOffset = OffsetBytes, TargetOffset = 0; CurrOffset < OffsetBytes + getScalarTypeSize(DstTy);
             ++CurrOffset, ++TargetOffset) {
            unsigned Idx = CurrOffset / BlockSizeInBytes;
            unsigned LocalOffset = CurrOffset % BlockSizeInBytes;
            llvm::Value *Elem = buildEEI(BlockInstsVec[Idx], LocalOffset, LLVMTypes::Int8Type, InsertBefore);
            ByteVec = buildIEI(ByteVec, Elem, TargetOffset, InsertBefore);
        }
        llvm::Value *ScalarizedByteVec = buildCast(
            ByteVec, llvm::IntegerType::get(*g->ctx, /* NumBits */ getScalarTypeSize(DstTy) << 3), InsertBefore);
        return buildCast(ScalarizedByteVec, DstTy, InsertBefore);
    }
}

bool XeGatherCoalescing::runOnBasicBlock(llvm::BasicBlock &bb) {
    DEBUG_START_BB("XeGatherCoalescing");
    bool modifiedAny = false;
    modifiedAny = runOnBasicBlockImpl(bb);
    DEBUG_END_BB("XeGatherCoalescing");
    return modifiedAny;
}

void XeGatherCoalescing::optimizePtr(llvm::Value *Ptr, PtrData &PD, llvm::Instruction *InsertPoint) {
    // Analyse memory accesses
    OffsetT MinIdx = INT64_MAX;
    OffsetT MaxIdx = INT64_MIN;
    unsigned TotalMemOpsCounter = PD.InstsData.size();
    llvm::Type *LargestType = nullptr;
    unsigned LargestTypeSize = 0;
    for (auto &ID : PD.InstsData) {
        // Adjust borders
        for (auto Idx : ID.Offsets) {
            MaxIdx = std::max(Idx, MaxIdx);
            MinIdx = std::min(Idx, MinIdx);
        }
        // Largest type is needed to handle the case with different type sizes
        unsigned TypeSize = getScalarTypeSize(ID.Inst->getType());
        if (TypeSize > LargestTypeSize) {
            LargestTypeSize = TypeSize;
            LargestType = ID.Inst->getType()->getScalarType();
            if (LargestType->isPointerTy())
                LargestType = g->target->is32Bit() ? LLVMTypes::Int32Type : LLVMTypes::Int64Type;
        }
    }

    // Calculate data length
    Assert(LargestTypeSize > 0);
    uint64_t DataSize = MaxIdx - MinIdx + LargestTypeSize;
    // Calculate size of block loads in powers of two:
    // block loads are aligned to OWORDs
    unsigned ReqSize = 1;
    while (ReqSize < DataSize)
        ReqSize <<= 1;

    // Adjust req size and calculate num if insts needed
    // TODO: experiment showed performance improvement with
    // max ReqSize of 4 * OWORD instead of 8 * OWORD.
    // Further investigation is needed.
    unsigned MemInstsNeeded = 1;
    if (ReqSize > 4 * OWORD) {
        // Dealing with powers of two
        MemInstsNeeded = ReqSize / (4 * OWORD);
        ReqSize = 4 * OWORD;
    }

    // TODO: not clear if we should skip it
    if (ReqSize < OWORD) {
        // Skip for now.
        return;
    }

    // Check for threshold
    int MemOpsDiff = TotalMemOpsCounter - MemInstsNeeded;
    if (MemOpsDiff < g->opt.thresholdForXeGatherCoalescing) {
        return;
    }

    // Build block loads
    BlockInstsVecT BlockLDs;
    for (unsigned i = 0; i < MemInstsNeeded; ++i) {
        llvm::Constant *Offset = llvm::ConstantInt::get(LLVMTypes::Int64Type, MinIdx + i * ReqSize);
        llvm::PtrToIntInst *PtrToInt =
            new llvm::PtrToIntInst(Ptr, LLVMTypes::Int64Type, "vectorized_ptrtoint", InsertPoint);

        llvm::Instruction *Addr = llvm::BinaryOperator::CreateAdd(PtrToInt, Offset, "vectorized_address", InsertPoint);
        llvm::Type *RetType = llvm::FixedVectorType::get(LargestType, ReqSize / LargestTypeSize);
        llvm::IntToPtrInst *PtrForLd =
            new llvm::IntToPtrInst(Addr, llvm::PointerType::get(RetType, 0), "vectorized_address_ptr", InsertPoint);
        llvm::LoadInst *LD = new llvm::LoadInst(RetType, PtrForLd, "vectorized_ld_exp", InsertPoint);

        //  If the Offset is zero, we generate a LD with default alignment for the target.
        //  If the Offset is non-zero, we should re-align the LD based on its value.
        if (Offset != nullptr && !Offset->isZeroValue()) {
            const uint64_t offset{llvm::dyn_cast<llvm::ConstantInt>(Offset)->getZExtValue()};
#if ISPC_LLVM_VERSION >= ISPC_LLVM_16_0
            auto aligned_offset = llvm::bit_floor(offset);
#else
            auto aligned_offset = llvm::PowerOf2Floor(offset);
#endif
            const uint64_t alignment{llvm::MinAlign(offset, std::min(aligned_offset, static_cast<uint64_t>(OWORD)))};
            LD->setAlignment(llvm::Align{alignment});
        }

        BlockLDs.push_back(LD);
    }

    // Replace users
    for (auto &ID : PD.InstsData) {
        if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(ID.Inst)) {
            // Adjust offset
            OffsetT Offset = ID.Offsets[0] - MinIdx;
            LI->replaceAllUsesWith(extractValueFromBlock(BlockLDs, Offset, LI->getType(), InsertPoint));
        } else if (auto *Gather = getPseudoGatherConstOffset(ID.Inst)) {
            llvm::Value *NewVal = llvm::UndefValue::get(Gather->getType());
            unsigned CurrElem = 0;
            for (auto Offset : ID.Offsets) {
                // Adjust offset
                OffsetT AdjOffset = Offset - MinIdx;
                auto *ExtractedValue =
                    extractValueFromBlock(BlockLDs, AdjOffset, Gather->getType()->getScalarType(), InsertPoint);
                NewVal = llvm::InsertElementInst::Create(NewVal, ExtractedValue,
                                                         llvm::ConstantInt::get(LLVMTypes::Int64Type, CurrElem),
                                                         "gather_coal_iei", InsertPoint);
                ++CurrElem;
            }
            Gather->replaceAllUsesWith(NewVal);
        }

        // Mark to delete
        PossiblyDead.insert(ID.Inst);
    }

    // Done
    modifiedAny = true;
}

llvm::CallInst *XeGatherCoalescing::getPseudoGatherConstOffset(llvm::Instruction *Inst) const {
    if (auto CI = llvm::dyn_cast<llvm::CallInst>(Inst)) {
        llvm::Function *Function = CI->getCalledFunction();
        if (Function && Function->getName().startswith("__pseudo_gather_base_offsets"))
            if (isConstOffsetPseudoGather(CI))
                return CI;
    }
    return nullptr;
}

bool XeGatherCoalescing::isConstOffsetPseudoGather(llvm::CallInst *CI) const {
    Assert(CI != nullptr && CI->getCalledFunction() &&
           CI->getCalledFunction()->getName().startswith("__pseudo_gather_base_offsets"));
    llvm::Value *opOffset = CI->getOperand(2);

    return (llvm::isa<llvm::ConstantDataVector>(opOffset) || llvm::isa<llvm::ConstantAggregateZero>(opOffset) ||
            llvm::isa<llvm::ConstantVector>(opOffset));
}

bool XeGatherCoalescing::isOptimizationTarget(llvm::Instruction *Inst) const {
    if (auto LI = llvm::dyn_cast<llvm::LoadInst>(Inst)) {
        if (!LI->getType()->isVectorTy() && !LI->getType()->isAggregateType())
            return GetAddressSpace(LI->getPointerOperand()) == AddrSpace;
    } else if (auto Gather = getPseudoGatherConstOffset(Inst)) {
        return GetAddressSpace(getPointer(Gather)) == AddrSpace;
    }

    return false;
}

MemoryCoalescing::OffsetsVecT XeGatherCoalescing::getOffset(llvm::Instruction *Inst) const {
    OffsetsVecT Res;

    if (llvm::isa<llvm::LoadInst>(Inst))
        Res.push_back(0);
    else if (auto Gather = getPseudoGatherConstOffset(Inst)) {
        if (llvm::isa<llvm::ConstantAggregateZero>(Gather->getOperand(2))) {
            Res.push_back(0);
        } else {
            Res = getConstOffsetFromVector(Gather->getOperand(2));
            applyScale(Res, llvm::cast<llvm::ConstantInt>(Gather->getOperand(1))->getSExtValue());
        }
    }

    return Res;
}

llvm::Value *XeGatherCoalescing::getPointer(llvm::Instruction *Inst) const {
    if (auto LI = llvm::dyn_cast<llvm::LoadInst>(Inst)) {
        return LI->getPointerOperand();
    } else if (auto Gather = getPseudoGatherConstOffset(Inst)) {
        return Gather->getOperand(0);
    }

    return nullptr;
}

} // namespace ispc

#endif
