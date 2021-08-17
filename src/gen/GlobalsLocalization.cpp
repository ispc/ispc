/*
  Copyright (c) 2014, 2016-2021, Intel Corporation
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

/** @file GlobalsLocalization.cpp
    @brief Localize global variables used in kernels
*/

#ifdef ISPC_XE_ENABLED
#define DEBUG_TYPE "localize_globals"

#include "GlobalsLocalization.h"

#include <iterator>

#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/ADT/SCCIterator.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/Statistic.h>
#include <llvm/Analysis/CallGraph.h>
#include <llvm/Analysis/CallGraphSCCPass.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/GenXIntrinsics/GenXIntrOpts.h>
#include <llvm/GenXIntrinsics/GenXIntrinsics.h>
#include <llvm/GenXIntrinsics/GenXMetadata.h>
#if ISPC_LLVM_VERSION < ISPC_LLVM_11_0
#include <llvm/IR/CallSite.h>
#endif
#include <llvm/IR/CFG.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>
#include <llvm/InitializePasses.h>
#include <llvm/Pass.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Scalar.h>

using namespace llvm;

using LocalizationLimitT = int32_t;
static constexpr auto LocalizeAll = std::numeric_limits<LocalizationLimitT>::max();
static cl::opt<LocalizationLimitT>
    LocalizationLimit("localization-limit", cl::desc("maximum size (in bytes) used to localize global variables"),
                      cl::init(LocalizeAll));

namespace llvm {
void initializeGlobalsLocalizationPass(PassRegistry &);
} // namespace llvm

/// Localizing global variables
/// ^^^^^^^^^^^^^^^^^^^^^^^^^^^
///
/// General idea of localizing global variables into locals. Globals used in
/// different kernels get a seperate copy and they are always invisiable to
/// other kernels and we can safely localize all globals used (including
/// indirectly) in a kernel. For example,
///
/// .. code-block:: text
///
///   @gv1 = global <8 x float> zeroinitializer, align 32
///   @gv2 = global <8 x float> zeroinitializer, align 32
///   @gv3 = global <8 x float> zeroinitializer, align 32
///
///   define dllexport void @f0() {
///     call @f1()
///     call @f2()
///     call @f3()
///   }
///
///   define internal void @f1() {
///     ; ...
///     store <8 x float> %splat1, <8 x float>* @gv1, align 32
///   }
///
///   define internal void @f2() {
///     ; ...
///     store <8 x float> %splat2, <8 x float>* @gv2, align 32
///   }
///
///   define internal void @f3() {
///     %1 = <8 x float>* @gv1, align 32
///     %2 = <8 x float>* @gv2, align 32
///     %3 = fadd <8 x float> %1, <8 x float> %2
///     store <8 x float> %3, <8 x float>* @gv3, align 32
///   }
///
/// will be transformed into
///
/// .. code-block:: text
///
///   define dllexport void @f0() {
///     %v1 = alloca <8 x float>, align 32
///     %v2 = alloca <8 x float>, align 32
///     %v3 = alloca <8 x float>, align 32
///
///     %0 = load <8 x float> * %v1, align 32
///     %1 = { <8 x float> } call @f1_transformed(<8 x float> %0)
///     %2 = extractvalue { <8 x float> } %1, 0
///     store <8  x float> %2, <8 x float>* %v1, align 32
///
///     %3 = load <8 x float> * %v2, align 32
///     %4 = { <8 x float> } call @f2_transformed(<8 x float> %3)
///     %5 = extractvalue { <8 x float> } %4, 0
///     store <8  x float> %5, <8 x float>* %v1, align 32
///
///     %6 = load <8 x float> * %v1, align 32
///     %7 = load <8 x float> * %v2, align 32
///     %8 = load <8 x float> * %v3, align 32
///
///     %9 = { <8 x float>, <8 x float>, <8 x float> }
///          call @f3_transformed(<8 x float> %6, <8 x float> %7, <8 x float>
///          %8)
///
///     %10 = extractvalue { <8 x float>, <8 x float>, <8 x float> } %9, 0
///     store <8  x float> %10, <8 x float>* %v1, align 32
///     %11 = extractvalue { <8 x float>, <8 x float>, <8 x float> } %9, 1
///     store <8  x float> %11, <8 x float>* %v2, align 32
///     %12 = extractvalue { <8 x float>, <8 x float>, <8 x float> } %9, 2
///     store <8  x float> %12, <8 x float>* %v3, align 32
///   }
///
/// All callees will be updated accordingly, E.g. f1_transformed becomes
///
/// .. code-block:: text
///
///   define internal { <8 x float> } @f1_transformed(<8 x float> %v1) {
///     %0 = alloca <8 x float>, align 32
///     store <8 x float> %v1, <8 x float>* %0, align 32
///     ; ...
///     store <8 x float> %splat1, <8 x float>* @0, align 32
///     ; ...
///     %1 = load <8 x float>* %0, align 32
///     %2 = insertvalue { <8 x float> } undef, <8 x float> %1, 0
///     ret { <8 x float> } %2
///   }
///
namespace {

// \brief Collect necessary information for global variable localization.
class LocalizationInfo {
  public:
    typedef SetVector<GlobalVariable *> GlobalSetTy;

    explicit LocalizationInfo(Function *F) : Fn(F) {}
    LocalizationInfo() : Fn(0) {}

    Function *getFunction() const { return Fn; }
    bool empty() const { return Globals.empty(); }
    GlobalSetTy &getGlobals() { return Globals; }

    // \brief Add a global.
    void addGlobal(GlobalVariable *GV) { Globals.insert(GV); }

    // \brief Add all globals from callee.
    void addGlobals(LocalizationInfo &LI) { Globals.insert(LI.getGlobals().begin(), LI.getGlobals().end()); }

    void setArgIndex(GlobalVariable *GV, unsigned ArgIndex) {
        assert(!IndexMap.count(GV));
        IndexMap[GV] = ArgIndex;
    }
    unsigned getArgIndex(GlobalVariable *GV) const {
        assert(IndexMap.count(GV));
        return IndexMap.lookup(GV);
    }

  private:
    // \brief The function being analyzed.
    Function *Fn;

    // \brief Global variables that are used directly or indirectly.
    GlobalSetTy Globals;

    // This map keeps track of argument index for a global variable.
    SmallDenseMap<GlobalVariable *, unsigned> IndexMap;
};

struct GlobalsLocalization : public CallGraphSCCPass {
    static char ID;

    GlobalsLocalization() : CallGraphSCCPass(ID) {
        initializeGlobalsLocalizationPass(*PassRegistry::getPassRegistry());
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const { CallGraphSCCPass::getAnalysisUsage(AU); }

    virtual bool runOnSCC(CallGraphSCC &SCC);

    virtual bool doInitialization(CallGraph &CG);
    virtual bool doFinalization(CallGraph &CG);

  private:
    CallGraphNode *ProcessNode(CallGraphNode *CGN);

    // Fix argument passing for kernels.
    CallGraphNode *TransformKernel(Function *F);

    // \brief Create allocas for globals and replace their uses.
    void LocalizeGlobals(LocalizationInfo &LI);

    // \brief Compute the localized global variables for each function.
    void AnalyzeGlobals(CallGraph &CG);

    // \brief Returns the localization info associated to a function.
    LocalizationInfo &getLocalizationInfo(Function *F) {
        if (!GlobalInfo.count(F)) {
            LocalizationInfo *LI = new LocalizationInfo(F);
            LocalizationInfoObjs.push_back(LI);
            GlobalInfo[F] = LI;
            return *LI;
        }
        return *GlobalInfo[F];
    }

    void addDirectGlobal(Function *F, GlobalVariable *GV) { getLocalizationInfo(F).addGlobal(GV); }

    // \brief Add all globals from callee to caller.
    void addIndirectGlobal(Function *F, Function *Callee) {
        getLocalizationInfo(F).addGlobals(getLocalizationInfo(Callee));
    }

    // This map captures all global variables to be localized.
    SmallDenseMap<Function *, LocalizationInfo *> GlobalInfo;

    // Kernels in the module being processed.
    SmallPtrSet<Function *, 8> Kernels;

    // Already visited functions.
    SmallPtrSet<Function *, 8> AlreadyVisited;

    // LocalizationInfo objects created.
    SmallVector<LocalizationInfo *, 8> LocalizationInfoObjs;
};

} // namespace

template <typename Range> using iterator_t = decltype(std::begin(std::declval<Range &>()));

template <typename Range> using range_pointer_t = typename std::iterator_traits<iterator_t<Range>>::pointer;

template <typename Range> using range_reference_t = typename std::iterator_traits<iterator_t<Range>>::reference;

/* Returns the first iterator (let's name it RetIt) such that
 * std::accumulate(First, RetIt, 0) > Bound (not full truth, read below).
 *
 * Arguments:
 * \p First, \p Last - considered range
 * \p Bound - considered Bound
 * \p Op - functor that returns T, takes T and decltype(*First)
 *    respectively as arguments. It is meant to increment current partial sum.
 *    First argument is previous partial sum, second argument is upcoming value
 *    from the range, new partial sum is returned.
 *
 * Arguments of \p PlusEqualOp may not be equal, so the range may possibly point
 * not to T type. In this case partial sum is calculated for transformed range
 * (transformation is hidden in \p Op).
 */
template <typename ForwardIt, typename PlusEqualOp, typename T>
ForwardIt upperPartialSumBound(ForwardIt First, ForwardIt Last, T Bound, PlusEqualOp Op) {
    T CurSum = 0;
    for (; First != Last; ++First) {
        CurSum = Op(CurSum, *First);
        if (CurSum > Bound)
            return First;
    }
    return Last;
}

// Currently weight of the global defines by its size
static int calcGVWeight(const GlobalVariable &GV, const DataLayout &DL) {
    return DL.getTypeAllocSize(GV.getValueType());
}

/* selectGlobalsToLocalize - chooses which globals to localize
 * Returns vector of pointers to such globals.
 *
 * Algorithm: exclude globals that definitely should not be localized
 * sort globals by weight, choose first smallest ones, sum of which is under \p
 * Bound
 *
 * \p Globals - range of globals to choose from
 * \p Bound - bound not to overcome
 * \p ExcludePred - functor : GVRef -> bool, true if global should not be
 * localized \p WeightCalculator - functor : GVRef -> decltype(Bound), returns
 * weight of global
 */
template <typename ForwardRange, typename ExcludePredT, typename T, typename WeightCalculatorT>
auto selectGlobalsToLocalize(ForwardRange Globals, T Bound, ExcludePredT ExcludePred,
                             WeightCalculatorT WeightCalculator) -> std::vector<range_pointer_t<ForwardRange>> {
    assert(Bound >= 0 && "bound must be nonnegative");
    using GVPtr = range_pointer_t<ForwardRange>;
    using GVRef = range_reference_t<ForwardRange>;
    if (Bound == 0)
        return std::vector<GVPtr>();

    // filter out those, that we must exclude
    auto Unexcluded = make_filter_range(Globals, [ExcludePred](GVRef GV) { return !ExcludePred(GV); });
    using GVWithWeightT = std::pair<GVPtr, int>;

    if (Bound == LocalizeAll) {
        std::vector<GVPtr> ToLocalize;
        transform(Unexcluded, std::back_inserter(ToLocalize), [](GVRef GV) { return &GV; });
        return ToLocalize;
    }

    std::vector<GVWithWeightT> ToLocalizeWithWeight;
    transform(Unexcluded, std::back_inserter(ToLocalizeWithWeight),
              [WeightCalculator](GVRef GV) { return std::make_pair(&GV, WeightCalculator(GV)); });

    // sort globals by weight
    std::sort(ToLocalizeWithWeight.begin(), ToLocalizeWithWeight.end(),
              [](GVWithWeightT LHS, GVWithWeightT RHS) { return LHS.second < RHS.second; });

    // filter max number of lightest ones, which weight sum is under the bound
    auto FirstNotToLocalize =
        upperPartialSumBound(ToLocalizeWithWeight.begin(), ToLocalizeWithWeight.end(), Bound,
                             [](decltype(Bound) Base, GVWithWeightT Inc) { return Base + Inc.second; });

    // collect them back to ToLocalize
    std::vector<GVPtr> ToLocalize;
    ToLocalize.reserve(FirstNotToLocalize - ToLocalizeWithWeight.begin());
    std::transform(ToLocalizeWithWeight.begin(), FirstNotToLocalize, std::back_inserter(ToLocalize),
                   [](GVWithWeightT GV) { return GV.first; });

    return ToLocalize;
}

bool GlobalsLocalization::doInitialization(CallGraph &CG) {
    // Analyze global variable usages and for each function attaches global
    // variables to be copy-in and copy-out.
    AnalyzeGlobals(CG);

    auto getValue = [](Metadata *M) -> Value * {
        if (auto VM = dyn_cast<ValueAsMetadata>(M))
            return VM->getValue();
        return nullptr;
    };

    // Collect all kernels from named metadata.
    if (NamedMDNode *Named = CG.getModule().getNamedMetadata(genx::FunctionMD::GenXKernels)) {
        assert(Named);
        for (unsigned I = 0, E = Named->getNumOperands(); I != E; ++I) {
            MDNode *Node = Named->getOperand(I);
            if (Function *F = dyn_cast_or_null<Function>(getValue(Node->getOperand(0))))
                Kernels.insert(F);
        }
    }

    // no change.
    return false;
}

bool GlobalsLocalization::doFinalization(CallGraph &CG) {
    bool Changed = false;
    for (Module::global_iterator I = CG.getModule().global_begin(); I != CG.getModule().global_end();
         /*empty*/) {
        GlobalVariable *GV = &*I++;
        if (GV->use_empty()) {
            GV->eraseFromParent();
            Changed = true;
        }
    }

    for (LocalizationInfo *Obj : LocalizationInfoObjs)
        delete Obj;

    return Changed;
}

bool GlobalsLocalization::runOnSCC(CallGraphSCC &SCC) {
    bool Changed = false, LocalChange;

    // Iterate until we stop transforming from this SCC.
    do {
        LocalChange = false;
        for (CallGraphSCC::iterator I = SCC.begin(), E = SCC.end(); I != E; ++I) {
            if (CallGraphNode *CGN = ProcessNode(*I)) {
                LocalChange = true;
                SCC.ReplaceNode(*I, CGN);
            }
        }
        Changed |= LocalChange;
    } while (LocalChange);

    return Changed;
}

// Sometimes we can get phi with GEP (or maybe some other inst) as an argument.
// While GEP's arguments are constants, its OK as GEP is a constant to.
// But when we replace constants with lokals, GEP becomes a normal instruction,
// a normal instruction, that is placed before phi - wrong IR, we need to fix
// it. Here it is fixed.
static void fixPhiUseIssue(Instruction *Inst) {
    auto PhiUse = cast<PHINode>(Inst->use_begin()->getUser());
    auto InstOpNoInPhi = Inst->use_begin()->getOperandNo();
    assert(Inst->getParent() == PhiUse->getParent());
    Inst->removeFromParent();
    Inst->insertBefore(PhiUse->getIncomingBlock(InstOpNoInPhi)->getTerminator());
}

// Replace uses of global variables with the corresponding allocas with a
// specified function.
//
// Returns vector of instructions with phi use, that should be later fixed.
static std::vector<Instruction *> replaceUsesWithinFunction(SmallDenseMap<Value *, Value *> &GlobalsToReplace,
                                                            Function *F) {
    std::vector<Instruction *> PhiUseIssueInsts;
    for (auto I = inst_begin(F), E = inst_end(F); I != E; ++I) {
        Instruction *Inst = &*I;
        for (unsigned i = 0, e = Inst->getNumOperands(); i < e; ++i) {
            auto Iter = GlobalsToReplace.find(Inst->getOperand(i));
            if (Iter != GlobalsToReplace.end())
                Inst->setOperand(i, Iter->second);
        }
        if (Inst->getNumUses() == 1) {
            auto PhiUse = dyn_cast<PHINode>(Inst->use_begin()->getUser());
            if (PhiUse && Inst->getParent() == PhiUse->getParent()) {
                PhiUseIssueInsts.push_back(Inst);
            }
        }
    }
    return PhiUseIssueInsts;
}

// \brief Create allocas for globals directly used in this kernel and
// replace all uses.
void GlobalsLocalization::LocalizeGlobals(LocalizationInfo &LI) {
    const LocalizationInfo::GlobalSetTy &Globals = LI.getGlobals();
    typedef LocalizationInfo::GlobalSetTy::const_iterator IteratorTy;

    SmallDenseMap<Value *, Value *> GlobalsToReplace;
    Function *Fn = LI.getFunction();
    for (IteratorTy I = Globals.begin(), E = Globals.end(); I != E; ++I) {
        GlobalVariable *GV = (*I);
        LLVM_DEBUG(dbgs() << "Localizing global: " << *GV);

        Instruction &FirstI = *Fn->getEntryBlock().begin();
        Type *ElemTy = GV->getType()->getElementType();
        AllocaInst *Alloca = new AllocaInst(ElemTy, 0, GV->getName() + ".local", &FirstI);
        Alloca->setAlignment(llvm::MaybeAlign(GV->getAlignment()).valueOrOne());

        if (GV->hasInitializer() && !isa<UndefValue>(GV->getInitializer()))
            new StoreInst(GV->getInitializer(), Alloca, &FirstI);

        GlobalsToReplace.insert(std::make_pair(GV, Alloca));
    }

    // Replaces all globals uses within this function.
    auto PhiUseIssueInsts = replaceUsesWithinFunction(GlobalsToReplace, Fn);

    for (auto InstWithPhiUse : PhiUseIssueInsts) {
        fixPhiUseIssue(InstWithPhiUse);
    }

    for (IteratorTy I = Globals.begin(), E = Globals.end(); I != E; ++I) {
        GlobalVariable *GV = (*I);
        if (GV->user_empty())
            GV->eraseFromParent();
    }
}

CallGraphNode *GlobalsLocalization::ProcessNode(CallGraphNode *CGN) {
    Function *F = CGN->getFunction();

    // nothing to do for declarations or already visited functions.
    if (!F || F->isDeclaration() || AlreadyVisited.count(F))
        return 0;

    // Variables to be localized.
    LocalizationInfo &LI = getLocalizationInfo(F);

    // This is a kernel.
    if (Kernels.count(F)) {
        // Localize globals for kernels.
        if (!LI.getGlobals().empty())
            LocalizeGlobals(LI);

        // No changes to this kernel's prototype.
        return 0;
    }
    return 0;
}

static void breakConstantVector(unsigned i, Instruction *CurInst, Instruction *InsertPt) {
    ConstantVector *CV = cast<ConstantVector>(CurInst->getOperand(i));

    // Splat case.
    if (auto S = dyn_cast_or_null<ConstantExpr>(CV->getSplatValue())) {
        // Turn element into an instruction
        auto Inst = S->getAsInstruction();
        Inst->setDebugLoc(CurInst->getDebugLoc());
        Inst->insertBefore(InsertPt);

        // Splat this value.
        IRBuilder<> Builder(InsertPt);
        Value *NewVal = Builder.CreateVectorSplat(CV->getNumOperands(), Inst);

        // Update i-th operand with newly created splat.
        CurInst->setOperand(i, NewVal);
    }

    SmallVector<Value *, 8> Vals;
    bool HasConstExpr = false;
    for (unsigned j = 0, N = CV->getNumOperands(); j < N; ++j) {
        Value *Elt = CV->getOperand(j);
        if (auto CE = dyn_cast<ConstantExpr>(Elt)) {
            auto Inst = CE->getAsInstruction();
            Inst->setDebugLoc(CurInst->getDebugLoc());
            Inst->insertBefore(InsertPt);
            Vals.push_back(Inst);
            HasConstExpr = true;
        } else
            Vals.push_back(Elt);
    }

    if (HasConstExpr) {
        Value *Val = UndefValue::get(CV->getType());
        IRBuilder<> Builder(InsertPt);
        for (unsigned j = 0, N = CV->getNumOperands(); j < N; ++j)
            Val = Builder.CreateInsertElement(Val, Vals[j], j);
        CurInst->setOperand(i, Val);
    }
}

static void breakConstantExprs(Function *F) {
    for (po_iterator<BasicBlock *> i = po_begin(&F->getEntryBlock()), e = po_end(&F->getEntryBlock()); i != e; ++i) {
        BasicBlock *BB = *i;
        // The effect of this loop is that we process the instructions in reverse
        // order, and we re-process anything inserted before the instruction
        // being processed.
        for (Instruction *CurInst = BB->getTerminator(); CurInst;) {
            PHINode *PN = dyn_cast<PHINode>(CurInst);
            for (unsigned i = 0, e = CurInst->getNumOperands(); i < e; ++i) {
                auto InsertPt = PN ? PN->getIncomingBlock(i)->getTerminator() : CurInst;
                Value *Op = CurInst->getOperand(i);
                if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Op)) {
                    Instruction *NewInst = CE->getAsInstruction();
                    NewInst->setDebugLoc(CurInst->getDebugLoc());
                    NewInst->insertBefore(CurInst);
                    CurInst->setOperand(i, NewInst);
                } else if (isa<ConstantVector>(Op))
                    breakConstantVector(i, CurInst, InsertPt);
            }
            CurInst = CurInst == &BB->front() ? nullptr : CurInst->getPrevNode();
        }
    }
}

// For each function, compute the list of globals that need to be passed as
// copy-in and copy-out arguments.
void GlobalsLocalization::AnalyzeGlobals(CallGraph &CG) {
    Module &M = CG.getModule();

    // No global variables.
    if (M.global_empty())
        return;

    // Store functions in a SetVector to keep order and make searching efficient.
    SetVector<Function *> Funcs;
    for (auto I = scc_begin(&CG), IE = scc_end(&CG); I != IE; ++I) {
        const std::vector<CallGraphNode *> &SCCNodes = *I;
        for (const CallGraphNode *Node : SCCNodes) {
            Function *F = Node->getFunction();
            if (F != nullptr && !F->isDeclaration()) {
                Funcs.insert(F);
                breakConstantExprs(F);
            }
        }
    }
    auto PrintIndexChecker = [](Use &IUI) {
        CallInst *CI = dyn_cast<CallInst>(IUI.getUser());
        if (!CI)
            return false;
        Function *Callee = CI->getCalledFunction();
        if (!Callee)
            return false;
        unsigned IntrinID = GenXIntrinsic::getAnyIntrinsicID(Callee);
        return (IntrinID == GenXIntrinsic::genx_print_format_index);
    };
    auto UsesPrintChecker = [PrintIndexChecker](const Use &UI) {
        auto *User = UI.getUser();
        return std::any_of(User->use_begin(), User->use_end(), PrintIndexChecker);
    };
    const auto &DL = M.getDataLayout();
    auto ToLocalize = selectGlobalsToLocalize(
        M.globals(), LocalizationLimit.getValue(),
        [UsesPrintChecker](const GlobalVariable &GV) {
            // don't localize global constant format string if it's used by
            // print_index intrinsic
            bool UsesPrintIndex = std::any_of(GV.use_begin(), GV.use_end(), UsesPrintChecker);
            // Avoid localizing constant addrspace as a workaround for print implementation.
            // Constant strings should stay constant strings till spirv.
            return GV.hasAttribute(genx::FunctionMD::GenXVolatile) || UsesPrintIndex || GV.getAddressSpace() != 0;
        },
        [&DL](const GlobalVariable &GV) { return calcGVWeight(GV, DL); });
    for (auto I = Funcs.begin(), E = Funcs.end(); I != E; ++I) {
        Function *Fn = *I;
        LLVM_DEBUG(dbgs() << "Visiting " << Fn->getName());

        // Collect globals used directly.
        for (GlobalVariable *GV : ToLocalize) {
            for (Value::use_iterator UI = GV->use_begin(), UE = GV->use_end(); UI != UE; ++UI) {
                Instruction *Inst = dyn_cast<Instruction>(UI->getUser());
                // not used in this function.
                if (!Inst || Inst->getParent()->getParent() != Fn)
                    continue;

                // Find the global being used and populate this info.
                for (unsigned i = 0, e = Inst->getNumOperands(); i < e; ++i) {
                    Value *Op = Inst->getOperand(i);
                    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Op))
                        addDirectGlobal(Fn, GV);
                }
            }
        }

        // Collect globals used indirectly.
        for (inst_iterator II = inst_begin(Fn), IE = inst_end(Fn); II != IE; ++II) {
            Instruction *Inst = &*II;
            // Ignore InvokeInst.
            if (CallInst *CI = dyn_cast<CallInst>(Inst)) {
                // Ignore indirect calls
                if (Function *Callee = CI->getCalledFunction()) {
                    // Collect all globals from its callee.
                    if (!Callee->isDeclaration())
                        addIndirectGlobal(Fn, Callee);
                }
            }
        }
    }
}

char GlobalsLocalization::ID = 0;
INITIALIZE_PASS_BEGIN(GlobalsLocalization, "GlobalsLocalization", "Localize globals before SPIR-V", false, false)
INITIALIZE_PASS_DEPENDENCY(CallGraphWrapperPass)
INITIALIZE_PASS_END(GlobalsLocalization, "GlobalsLocalization", "Localize globals before SPIR-V", false, false)

Pass *llvm::createGlobalsLocalizationPass() { return new GlobalsLocalization(); }
#endif
