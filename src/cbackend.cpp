//===-- CBackend.cpp - Library for converting LLVM code to C --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This library converts LLVM code to C code, compilable by GCC and other C
// compilers.
//
//===----------------------------------------------------------------------===//

#include "ispc.h"
#include "module.h"

#include <math.h>
#include <sstream>
#include <stdio.h>
#include <string.h>

#ifndef _MSC_VER
#include <inttypes.h>
#define HAVE_PRINTF_A 1
#define ENABLE_CBE_PRINTF_A 1
#endif

#ifndef PRIx64
#define PRIx64 "llx"
#endif

#include "llvmutil.h"

#if ISPC_LLVM_VERSION == ISPC_LLVM_3_2
#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/InlineAsm.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Intrinsics.h"
#include "llvm/Module.h"
#else // LLVM 3.3+
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#endif
#include "llvm/Pass.h"
#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_6 // <= 3.6
#include "llvm/PassManager.h"
#else // LLVM 3.7+
#include "llvm/IR/LegacyPassManager.h"
#endif
#if ISPC_LLVM_VERSION == ISPC_LLVM_3_2
#include "llvm/TypeFinder.h"
#else // LLVM_3_3+
#include "llvm/IR/TypeFinder.h"
#endif
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_4 // 3.2, 3.3, 3.4
#include "llvm/Support/InstIterator.h"
#else // 3.5+
#include "llvm/IR/InstIterator.h"
#endif
#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_5
#include "llvm/Analysis/FindUsedTypes.h"
#endif
#include "llvm/Analysis/LoopInfo.h"
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_5
#include "llvm/IR/CFG.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/FileSystem.h"
#include <llvm/IR/IRPrintingPasses.h>
#else
#include "llvm/Analysis/Verifier.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include <llvm/Assembly/PrintModulePass.h>
#endif
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/CodeGen/Passes.h"
//#include "llvm/Target/Mangler.h"
#include "llvm/Transforms/Scalar.h"
#if ISPC_LLVM_VERSION >= ISPC_LLVM_7_0
#include "llvm/Transforms/Utils.h"
#endif
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#if ISPC_LLVM_VERSION == ISPC_LLVM_3_2 // 3.2
#include "llvm/DataLayout.h"
#else // LLVM 3.3+
#include "llvm/IR/DataLayout.h"
#endif
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#if ISPC_LLVM_VERSION == ISPC_LLVM_3_2 // 3.2
#include "llvm/Support/InstVisitor.h"
#elif ISPC_LLVM_VERSION <= ISPC_LLVM_3_4 // 3.3, 3.4
#include "llvm/InstVisitor.h"
#else // LLVM 3.5+
#include "llvm/IR/InstVisitor.h"
#endif
#include "llvm/Support/Host.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"

#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_4 // 3.2, 3.3, 3.4
#include "llvm/Config/config.h"
#endif

#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#if ISPC_LLVM_VERSION > ISPC_LLVM_7_0
#include "llvm/IR/PatternMatch.h"
#endif
#include <algorithm>
// Some ms header decided to define setjmp as _setjmp, undo this for this file.
#ifdef _MSC_VER
#undef setjmp
#define snprintf _snprintf
#endif
///////////////////////////////////////////////////////////////////////////////
// This part of code was in LLVM's ConstantsScanner.h,
// but it was removed in revision #232397
namespace constant_scanner {
class constant_iterator : public std::iterator<std::forward_iterator_tag, const llvm::Constant, ptrdiff_t> {
    llvm::const_inst_iterator InstI; // Method instruction iterator
    unsigned OpIdx;                  // Operand index

    bool isAtConstant() const {
        assert(!InstI.atEnd() && OpIdx < InstI->getNumOperands() && "isAtConstant called with invalid arguments!");
        return llvm::isa<llvm::Constant>(InstI->getOperand(OpIdx));
    }

  public:
    constant_iterator(const llvm::Function *F) : InstI(llvm::inst_begin(F)), OpIdx(0) {
        // Advance to first constant... if we are not already at constant or end
        if (InstI != llvm::inst_end(F) &&                      // InstI is valid?
            (InstI->getNumOperands() == 0 || !isAtConstant())) // Not at constant?
            operator++();
    }

    constant_iterator(const llvm::Function *F, bool) // end ctor
        : InstI(llvm::inst_end(F)), OpIdx(0) {}

    bool operator==(const constant_iterator &x) const { return OpIdx == x.OpIdx && InstI == x.InstI; }
    bool operator!=(const constant_iterator &x) const { return !(*this == x); }

    pointer operator*() const {
        assert(isAtConstant() && "Dereferenced an iterator at the end!");
        return llvm::cast<llvm::Constant>(InstI->getOperand(OpIdx));
    }

    constant_iterator &operator++() { // Preincrement implementation
        ++OpIdx;
        do {
            unsigned NumOperands = InstI->getNumOperands();
            while (OpIdx < NumOperands && !isAtConstant()) {
                ++OpIdx;
            }

            if (OpIdx < NumOperands)
                return *this; // Found a constant!
            ++InstI;
            OpIdx = 0;
        } while (!InstI.atEnd());

        return *this; // At the end of the method
    }
};

inline constant_iterator constant_begin(const llvm::Function *F) { return constant_iterator(F); }

inline constant_iterator constant_end(const llvm::Function *F) { return constant_iterator(F, true); }

} // namespace constant_scanner

///////////////////////////////////////////////////////////////////////////////
// FIXME:
namespace {
/// TypeFinder - Walk over a module, identifying all of the types that are
/// used by the module.
class TypeFinder {
    // To avoid walking constant expressions multiple times and other IR
    // objects, we keep several helper maps.
    llvm::DenseSet<const llvm::Value *> VisitedConstants;
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_6 // LLVM 3.6+
    llvm::DenseSet<const llvm::Metadata *> VisitedMDNodes;
#endif
    llvm::DenseSet<llvm::Type *> VisitedTypes;
    std::vector<llvm::ArrayType *> &ArrayTypes;
    std::vector<llvm::IntegerType *> &IntegerTypes;
    std::vector<bool> &IsVolatile;
    std::vector<int> &Alignment;

  public:
    TypeFinder(std::vector<llvm::ArrayType *> &t, std::vector<llvm::IntegerType *> &i, std::vector<bool> &v,
               std::vector<int> &a)
        : ArrayTypes(t), IntegerTypes(i), IsVolatile(v), Alignment(a) {}

    void run(const llvm::Module &M) {
        // Get types from global variables.
        for (llvm::Module::const_global_iterator I = M.global_begin(), E = M.global_end(); I != E; ++I) {
            incorporateType(I->getType());
            if (I->hasInitializer())
                incorporateValue(I->getInitializer());
        }

        // Get types from aliases.
        for (llvm::Module::const_alias_iterator I = M.alias_begin(), E = M.alias_end(); I != E; ++I) {
            incorporateType(I->getType());
            if (const llvm::Value *Aliasee = I->getAliasee())
                incorporateValue(Aliasee);
        }

        llvm::SmallVector<std::pair<unsigned, llvm::MDNode *>, 4> MDForInst;

        // Get types from functions.
        for (llvm::Module::const_iterator FI = M.begin(), E = M.end(); FI != E; ++FI) {
            incorporateType(FI->getType());

            for (llvm::Function::const_iterator BB = FI->begin(), E = FI->end(); BB != E; ++BB)
                for (llvm::BasicBlock::const_iterator II = BB->begin(), E = BB->end(); II != E; ++II) {
                    const llvm::Instruction &I = *II;

                    // Operands of SwitchInsts changed format after 3.1
                    // Seems like there ought to be better way to do what we
                    // want here.  For now, punt on SwitchInsts.
                    if (llvm::isa<llvm::SwitchInst>(&I))
                        continue;

                    // Incorporate the type of the instruction and all its operands.
                    incorporateType(I.getType());
                    if (llvm::isa<llvm::StoreInst>(&I))
                        if (llvm::IntegerType *ITy = llvm::dyn_cast<llvm::IntegerType>(I.getType())) {
                            IntegerTypes.push_back(ITy);
                            const llvm::StoreInst *St = llvm::dyn_cast<llvm::StoreInst>(&I);
                            IsVolatile.push_back(St->isVolatile());
                            Alignment.push_back(St->getAlignment());
                        }

                    if (llvm::isa<llvm::LoadInst>(&I))
                        if (llvm::IntegerType *ITy = llvm::dyn_cast<llvm::IntegerType>(I.getType())) {
                            IntegerTypes.push_back(ITy);
                            const llvm::LoadInst *St = llvm::dyn_cast<llvm::LoadInst>(&I);
                            IsVolatile.push_back(St->isVolatile());
                            Alignment.push_back(St->getAlignment());
                        }

                    for (llvm::User::const_op_iterator OI = I.op_begin(), OE = I.op_end(); OI != OE; ++OI)
                        incorporateValue(*OI);

                    // Incorporate types hiding in metadata.
                    I.getAllMetadataOtherThanDebugLoc(MDForInst);
                    for (unsigned i = 0, e = MDForInst.size(); i != e; ++i)
                        incorporateMDNode(MDForInst[i].second);

                    MDForInst.clear();
                }
        }

        for (llvm::Module::const_named_metadata_iterator I = M.named_metadata_begin(), E = M.named_metadata_end();
             I != E; ++I) {
#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_7 /* 3.2, 3.3, 3.4, 3.5, 3.6, 3.7 */
            const llvm::NamedMDNode *NMD = I;
#else /* LLVM 3.8+ */
            const llvm::NamedMDNode *NMD = &*I;
#endif
            for (unsigned i = 0, e = NMD->getNumOperands(); i != e; ++i)
                incorporateMDNode(NMD->getOperand(i));
        }
    }

  private:
    void incorporateType(llvm::Type *Ty) {
        // Check to see if we're already visited this type.
        if (!VisitedTypes.insert(Ty).second)
            return;

        if (llvm::ArrayType *ATy = llvm::dyn_cast<llvm::ArrayType>(Ty))
            ArrayTypes.push_back(ATy);

        // Recursively walk all contained types.
        for (llvm::Type::subtype_iterator I = Ty->subtype_begin(), E = Ty->subtype_end(); I != E; ++I)
            incorporateType(*I);
    }

    /// incorporateValue - This method is used to walk operand lists finding
    /// types hiding in constant expressions and other operands that won't be
    /// walked in other ways.  GlobalValues, basic blocks, instructions, and
    /// inst operands are all explicitly enumerated.
    void incorporateValue(const llvm::Value *V) {
#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_5 // 3.2, 3.3, 3.4, 3.5
        if (const llvm::MDNode *M = llvm::dyn_cast<llvm::MDNode>(V)) {
            incorporateMDNode(M);
            return;
        }
#else /* LLVN 3.6+ */
        if (const llvm::MetadataAsValue *MV = llvm::dyn_cast<llvm::MetadataAsValue>(V)) {
            incorporateMDNode(MV->getMetadata());
            return;
        }
#endif
        if (!llvm::isa<llvm::Constant>(V) || llvm::isa<llvm::GlobalValue>(V))
            return;

        // Already visited?
        if (!VisitedConstants.insert(V).second)
            return;

        // Check this type.
        incorporateType(V->getType());

        // Look in operands for types.
        const llvm::User *U = llvm::cast<llvm::User>(V);
        for (llvm::Constant::const_op_iterator I = U->op_begin(), E = U->op_end(); I != E; ++I)
            incorporateValue(*I);
    }

#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_5 // 3.2, 3.3, 3.4, 3.5
    void incorporateMDNode(const llvm::MDNode *V) {

        // Already visited?
        if (!VisitedConstants.insert(V).second)
            return;

        // Look in operands for types.
        for (unsigned i = 0, e = V->getNumOperands(); i != e; ++i)
            if (llvm::Value *Op = V->getOperand(i))
                incorporateValue(Op);
    }
#else // LLVM 3.6+
    void incorporateMDNode(const llvm::Metadata *M) {

        // Already visited?
        if (!VisitedMDNodes.insert(M).second)
            return;

        if (const llvm::MDNode *N = llvm::dyn_cast<llvm::MDNode>(M)) {
            // Look in operands for types.
            for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
                if (const llvm::Metadata *O = N->getOperand(i))
                    incorporateMDNode(O);
        } else if (llvm::isa<llvm::MDString>(M)) {
            // Nothing to do with MDString.
        } else if (const llvm::ValueAsMetadata *V = llvm::dyn_cast<llvm::ValueAsMetadata>(M)) {
            incorporateValue(V->getValue());
        } else {
            // Some unknown Metadata subclass - has LLVM introduced something new?
            llvm_unreachable("Unknown Metadata subclass");
        }
    }
#endif
};
} // end anonymous namespace

static void findUsedArrayAndLongIntTypes(const llvm::Module *m, std::vector<llvm::ArrayType *> &t,
                                         std::vector<llvm::IntegerType *> &i, std::vector<bool> &IsVolatile,
                                         std::vector<int> &Alignment) {
    TypeFinder(t, i, IsVolatile, Alignment).run(*m);
}

static bool is_vec16_i64_ty(llvm::Type *Ty) {
    llvm::VectorType *VTy = llvm::dyn_cast<llvm::VectorType>(Ty);
    if ((VTy != NULL) && (VTy->getElementType()->isIntegerTy()) &&
        VTy->getElementType()->getPrimitiveSizeInBits() == 64)
        return true;
    return false;
}

namespace {
class CBEMCAsmInfo : public llvm::MCAsmInfo {
  public:
    CBEMCAsmInfo() {

#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_4 // 3.2, 3.3, 3.4
        GlobalPrefix = "";
#endif
        PrivateGlobalPrefix = "";
    }
};

/// CWriter - This class is the main chunk of code that converts an LLVM
/// module to a C translation unit.
class CWriter : public llvm::FunctionPass, public llvm::InstVisitor<CWriter> {
    llvm::formatted_raw_ostream &Out;
    llvm::IntrinsicLowering *IL;
    // llvm::Mangler *Mang;
    llvm::LoopInfo *LI;
    const llvm::Module *TheModule;
    const llvm::MCAsmInfo *TAsm;
    const llvm::MCRegisterInfo *MRI;
    const llvm::MCObjectFileInfo *MOFI;
    llvm::MCContext *TCtx;

    // FIXME: it's ugly to have the name be "TD" here, but it saves us
    // lots of ifdefs in the below since the new DataLayout and the old
    // TargetData have generally similar interfaces...
    const llvm::DataLayout *TD;

    std::map<const llvm::ConstantFP *, unsigned> FPConstantMap;
    std::map<const llvm::ConstantDataVector *, unsigned> VectorConstantMap;
    unsigned VectorConstantIndex;
    std::set<llvm::Function *> intrinsicPrototypesAlreadyGenerated;
    std::set<const llvm::Argument *> ByValParams;
    unsigned FPCounter;
    unsigned OpaqueCounter;
    llvm::DenseMap<const llvm::Value *, unsigned> AnonValueNumbers;
    unsigned NextAnonValueNumber;

    std::string includeName;
    int vectorWidth;

    /// UnnamedStructIDs - This contains a unique ID for each struct that is
    /// either anonymous or has no name.
    llvm::DenseMap<llvm::StructType *, unsigned> UnnamedStructIDs;
    llvm::DenseMap<llvm::ArrayType *, unsigned> ArrayIDs;

  public:
    static char ID;
    explicit CWriter(llvm::formatted_raw_ostream &o, const char *incname, int vecwidth)
        : FunctionPass(ID), Out(o), IL(0), /* Mang(0), */ LI(0), TheModule(0), TAsm(0), MRI(0), MOFI(0), TCtx(0), TD(0),
          OpaqueCounter(0), NextAnonValueNumber(0), includeName(incname ? incname : "generic_defs.h"),
          vectorWidth(vecwidth) {
#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_6 // <= 3.6
        initializeLoopInfoPass(*llvm::PassRegistry::getPassRegistry());
#else // LLVM 3.7+
        initializeLoopInfoWrapperPassPass(*llvm::PassRegistry::getPassRegistry());
#endif
        FPCounter = 0;
        VectorConstantIndex = 0;
    }

#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_9 // <= 3.9
    virtual const char *getPassName() const { return "C backend"; }
#else // LLVM 4.0+
    virtual llvm::StringRef getPassName() const { return "C backend"; }
#endif

    void getAnalysisUsage(llvm::AnalysisUsage &AU) const {
#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_6 // <= 3.6
        AU.addRequired<llvm::LoopInfo>();
#else // LLVM 3.7+
        AU.addRequired<llvm::LoopInfoWrapperPass>();
#endif
        AU.setPreservesAll();
    }

    virtual bool doInitialization(llvm::Module &M);

    bool runOnFunction(llvm::Function &F) {
        // Do not codegen any 'available_externally' functions at all, they have
        // definitions outside the translation unit.
        if (F.hasAvailableExternallyLinkage())
            return false;

#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_6 // <= 3.6
        LI = &getAnalysis<llvm::LoopInfo>();
#else // LLVM 3.7+
        LI = &getAnalysis<llvm::LoopInfoWrapperPass>().getLoopInfo();
#endif

        // Get rid of intrinsics we can't handle.
        lowerIntrinsics(F);

        // Output all floating point constants that cannot be printed accurately.
        printFloatingPointConstants(F);

        // Output all vector constants so they can be accessed with single
        // vector loads
        printVectorConstants(F);

        printFunction(F);
        return false;
    }

    virtual bool doFinalization(llvm::Module &M) {
        // Free memory...
        delete IL;
        delete TD;
        // delete Mang;
        delete TCtx;
        delete TAsm;
        delete MRI;
        delete MOFI;
        FPConstantMap.clear();
        VectorConstantMap.clear();
        ByValParams.clear();
        intrinsicPrototypesAlreadyGenerated.clear();
        UnnamedStructIDs.clear();
        ArrayIDs.clear();
        return false;
    }

    llvm::raw_ostream &printType(llvm::raw_ostream &Out, llvm::Type *Ty, bool isSigned = false,
                                 const std::string &VariableName = "", bool IgnoreName = false,
#if ISPC_LLVM_VERSION == ISPC_LLVM_3_2
                                 const llvm::AttrListPtr &PAL = llvm::AttrListPtr()
#elif ISPC_LLVM_VERSION <= ISPC_LLVM_4_0
                                 const llvm::AttributeSet &PAL = llvm::AttributeSet()
#else // LLVM 5.0+
                                 const llvm::AttributeList &PAL = llvm::AttributeList()
#endif
    );
    llvm::raw_ostream &printSimpleType(llvm::raw_ostream &Out, llvm::Type *Ty, bool isSigned,
                                       const std::string &NameSoFar = "");

    void printStructReturnPointerFunctionType(llvm::raw_ostream &Out,
#if ISPC_LLVM_VERSION == ISPC_LLVM_3_2
                                              const llvm::AttrListPtr &PAL,
#elif ISPC_LLVM_VERSION <= ISPC_LLVM_4_0
                                              const llvm::AttributeSet &PAL,
#else // LLVM 5.0+
                                              const llvm::AttributeList &PAL,
#endif
                                              llvm::PointerType *Ty);

    std::string getStructName(llvm::StructType *ST);
    std::string getArrayName(llvm::ArrayType *AT);

    /// writeOperandDeref - Print the result of dereferencing the specified
    /// operand with '*'.  This is equivalent to printing '*' then using
    /// writeOperand, but avoids excess syntax in some cases.
    void writeOperandDeref(llvm::Value *Operand) {
        if (isAddressExposed(Operand)) {
            // Already something with an address exposed.
            writeOperandInternal(Operand);
        } else {
            Out << "*(";
            writeOperand(Operand);
            Out << ")";
        }
    }

    void writeOperand(llvm::Value *Operand, bool Static = false);
    void writeInstComputationInline(llvm::Instruction &I);
    void writeOperandInternal(llvm::Value *Operand, bool Static = false);
    void writeOperandWithCast(llvm::Value *Operand, unsigned Opcode);
    void writeOperandWithCast(llvm::Value *Operand, const llvm::ICmpInst &I);
    bool writeInstructionCast(const llvm::Instruction &I);

    void writeMemoryAccess(llvm::Value *Operand, llvm::Type *OperandType, bool IsVolatile, unsigned Alignment);

  private:
    void lowerIntrinsics(llvm::Function &F);
    /// Prints the definition of the intrinsic function F. Supports the
    /// intrinsics which need to be explicitly defined in the CBackend.
    void printIntrinsicDefinition(const llvm::Function &F, llvm::raw_ostream &Out);

    void printModuleTypes();
    void printContainedStructs(llvm::Type *Ty, llvm::SmallPtrSet<llvm::Type *, 16> &);
    void printContainedArrays(llvm::ArrayType *ATy, llvm::SmallPtrSet<llvm::Type *, 16> &);
    void printFloatingPointConstants(llvm::Function &F);
    void printFloatingPointConstants(const llvm::Constant *C);
    void printVectorConstants(llvm::Function &F);
    void printFunctionSignature(const llvm::Function *F, bool Prototype);

    void printFunction(llvm::Function &);
    void printBasicBlock(llvm::BasicBlock *BB);
    void printLoop(llvm::Loop *L);

    bool printCast(unsigned opcode, llvm::Type *SrcTy, llvm::Type *DstTy);
    void printConstant(llvm::Constant *CPV, bool Static);
    void printConstantWithCast(llvm::Constant *CPV, unsigned Opcode);
    bool printConstExprCast(const llvm::ConstantExpr *CE, bool Static);
    void printConstantArray(llvm::ConstantArray *CPA, bool Static);
    void printConstantVector(llvm::ConstantVector *CV, bool Static);
    void printConstantDataSequential(llvm::ConstantDataSequential *CDS, bool Static);

    /// isAddressExposed - Return true if the specified value's name needs to
    /// have its address taken in order to get a C value of the correct type.
    /// This happens for global variables, byval parameters, and direct allocas.
    bool isAddressExposed(const llvm::Value *V) const {
        if (const llvm::Argument *A = llvm::dyn_cast<llvm::Argument>(V))
            return ByValParams.count(A);
        return llvm::isa<llvm::GlobalVariable>(V) || isDirectAlloca(V);
    }

    // isInlinableInst - Attempt to inline instructions into their uses to build
    // trees as much as possible.  To do this, we have to consistently decide
    // what is acceptable to inline, so that variable declarations don't get
    // printed and an extra copy of the expr is not emitted.
    //
    static bool isInlinableInst(const llvm::Instruction &I) {
        // Always inline cmp instructions, even if they are shared by multiple
        // expressions.  GCC generates horrible code if we don't.
        if (llvm::isa<llvm::CmpInst>(I) && llvm::isa<llvm::VectorType>(I.getType()) == false)
            return true;

#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_5 // 3.5+
        // This instruction returns a struct on LLVM older than 3.4, and can not be inlined
        if (llvm::isa<llvm::AtomicCmpXchgInst>(I))
            return false;
#endif

        // Must be an expression, must be used exactly once.  If it is dead, we
        // emit it inline where it would go.
        if (I.getType() == llvm::Type::getVoidTy(I.getContext()) || !I.hasOneUse() ||
#if ISPC_LLVM_VERSION > ISPC_LLVM_7_0 // 8.0+
            I.isTerminator()
#else
            llvm::isa<llvm::TerminatorInst>(I)
#endif
            || llvm::isa<llvm::CallInst>(I) || llvm::isa<llvm::PHINode>(I) || llvm::isa<llvm::LoadInst>(I) ||
            llvm::isa<llvm::VAArgInst>(I) || llvm::isa<llvm::InsertElementInst>(I) ||
            llvm::isa<llvm::InsertValueInst>(I) || llvm::isa<llvm::ExtractValueInst>(I) ||
            llvm::isa<llvm::SelectInst>(I))
            // Don't inline a load across a store or other bad things!
            return false;

        // Must not be used in inline asm, extractelement, or shufflevector.
        if (I.hasOneUse()) {

#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_5 // 3.5+
            const llvm::Instruction &User = llvm::cast<llvm::Instruction>(*I.user_back());
#else
            const llvm::Instruction &User = llvm::cast<llvm::Instruction>(*I.use_back());
#endif
            if (isInlineAsm(User) || llvm::isa<llvm::ExtractElementInst>(User) ||
                llvm::isa<llvm::ShuffleVectorInst>(User) || llvm::isa<llvm::AtomicRMWInst>(User) ||
                llvm::isa<llvm::AtomicCmpXchgInst>(User))
                return false;
        }

        // Only inline instruction it if it's use is in the same BB as the inst.
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_5 // 3.5+
        return I.getParent() == llvm::cast<llvm::Instruction>(I.user_back())->getParent();
#else
        return I.getParent() == llvm::cast<llvm::Instruction>(I.use_back())->getParent();
#endif
    }

    // isDirectAlloca - Define fixed sized allocas in the entry block as direct
    // variables which are accessed with the & operator.  This causes GCC to
    // generate significantly better code than to emit alloca calls directly.
    //
    static const llvm::AllocaInst *isDirectAlloca(const llvm::Value *V) {
        const llvm::AllocaInst *AI = llvm::dyn_cast<llvm::AllocaInst>(V);
        if (!AI)
            return 0;
        if (AI->isArrayAllocation())
            return 0; // FIXME: we can also inline fixed size array allocas!
        if (AI->getParent() != &AI->getParent()->getParent()->getEntryBlock())
            return 0;
        return AI;
    }

    // isInlineAsm - Check if the instruction is a call to an inline asm chunk.
    static bool isInlineAsm(const llvm::Instruction &I) {
        if (const llvm::CallInst *CI = llvm::dyn_cast<llvm::CallInst>(&I))
            return llvm::isa<llvm::InlineAsm>(CI->getCalledValue());
        return false;
    }

    // Instruction visitation functions
    friend class llvm::InstVisitor<CWriter>;

    void visitReturnInst(llvm::ReturnInst &I);
    void visitBranchInst(llvm::BranchInst &I);
    void visitSwitchInst(llvm::SwitchInst &I);
    void visitIndirectBrInst(llvm::IndirectBrInst &I);
    void visitInvokeInst(llvm::InvokeInst &I) { llvm_unreachable("Lowerinvoke pass didn't work!"); }
    void visitResumeInst(llvm::ResumeInst &I) { llvm_unreachable("DwarfEHPrepare pass didn't work!"); }
    void visitUnreachableInst(llvm::UnreachableInst &I);

    void visitPHINode(llvm::PHINode &I);
    void visitBinaryOperator(llvm::Instruction &I);
    void visitICmpInst(llvm::ICmpInst &I);
    void visitFCmpInst(llvm::FCmpInst &I);

    void visitCastInst(llvm::CastInst &I);
    void visitSelectInst(llvm::SelectInst &I);
    void visitCallInst(llvm::CallInst &I);
    void visitInlineAsm(llvm::CallInst &I);
    bool visitBuiltinCall(llvm::CallInst &I, llvm::Intrinsic::ID ID, bool &WroteCallee);

    void visitAllocaInst(llvm::AllocaInst &I);
    void visitLoadInst(llvm::LoadInst &I);
    void visitStoreInst(llvm::StoreInst &I);
    void visitGetElementPtrInst(llvm::GetElementPtrInst &I);
    void visitVAArgInst(llvm::VAArgInst &I);

    void visitInsertElementInst(llvm::InsertElementInst &I);
    void visitExtractElementInst(llvm::ExtractElementInst &I);
    void visitShuffleVectorInst(llvm::ShuffleVectorInst &SVI);

    void visitInsertValueInst(llvm::InsertValueInst &I);
    void visitExtractValueInst(llvm::ExtractValueInst &I);

    void visitAtomicRMWInst(llvm::AtomicRMWInst &I);
    void visitAtomicCmpXchgInst(llvm::AtomicCmpXchgInst &I);

    void visitInstruction(llvm::Instruction &I) {
#ifndef NDEBUG
        llvm::errs() << "C Writer does not know about " << I;
#endif
        llvm_unreachable(0);
    }

    void outputLValue(llvm::Instruction *I) { Out << "  " << GetValueName(I) << " = "; }

    bool isGotoCodeNecessary(llvm::BasicBlock *From, llvm::BasicBlock *To);
    void printPHICopiesForSuccessor(llvm::BasicBlock *CurBlock, llvm::BasicBlock *Successor, unsigned Indent);
    void printBranchToBlock(llvm::BasicBlock *CurBlock, llvm::BasicBlock *SuccBlock, unsigned Indent);
    void printGEPExpression(llvm::Value *Ptr, llvm::gep_type_iterator I, llvm::gep_type_iterator E, bool Static);

    std::string GetValueName(const llvm::Value *Operand);
};
} // namespace

char CWriter::ID = 0;

static std::string CBEMangle(const std::string &S) {
    std::string Result;

    for (unsigned i = 0, e = S.size(); i != e; ++i) {
        if (i + 1 != e && ((S[i] == '>' && S[i + 1] == '>') || (S[i] == '<' && S[i + 1] == '<'))) {
            Result += '_';
            Result += 'A' + (S[i] & 15);
            Result += 'A' + ((S[i] >> 4) & 15);
            Result += '_';
            i++;
        } else if (isalnum(S[i]) || S[i] == '_' || S[i] == '<' || S[i] == '>') {
            Result += S[i];
        } else {
            Result += '_';
            Result += 'A' + (S[i] & 15);
            Result += 'A' + ((S[i] >> 4) & 15);
            Result += '_';
        }
    }
    return Result;
}

std::string CWriter::getStructName(llvm::StructType *ST) {
    if (!ST->isLiteral() && !ST->getName().empty())
        return CBEMangle("l_" + ST->getName().str());

    return "l_unnamed_" + llvm::utostr(UnnamedStructIDs[ST]);
}

std::string CWriter::getArrayName(llvm::ArrayType *AT) { return "l_array_" + llvm::utostr(ArrayIDs[AT]); }

/// printStructReturnPointerFunctionType - This is like printType for a struct
/// return type, except, instead of printing the type as void (*)(Struct*, ...)
/// print it as "Struct (*)(...)", for struct return functions.
void CWriter::printStructReturnPointerFunctionType(llvm::raw_ostream &Out,
#if ISPC_LLVM_VERSION == ISPC_LLVM_3_2
                                                   const llvm::AttrListPtr &PAL,
#elif ISPC_LLVM_VERSION <= ISPC_LLVM_4_0
                                                   const llvm::AttributeSet &PAL,
#else // LLVM 5.0+
                                                   const llvm::AttributeList &PAL,
#endif
                                                   llvm::PointerType *TheTy) {
    llvm::FunctionType *FTy = llvm::cast<llvm::FunctionType>(TheTy->getElementType());
    std::string tstr;
    llvm::raw_string_ostream FunctionInnards(tstr);
    FunctionInnards << " (*) (";
    bool PrintedType = false;

    llvm::FunctionType::param_iterator I = FTy->param_begin(), E = FTy->param_end();
    llvm::Type *RetTy = llvm::cast<llvm::PointerType>(*I)->getElementType();
    unsigned Idx = 1;
    for (++I, ++Idx; I != E; ++I, ++Idx) {
        if (PrintedType)
            FunctionInnards << ", ";
        llvm::Type *ArgTy = *I;
#if ISPC_LLVM_VERSION == ISPC_LLVM_3_2
        if (PAL.getParamAttributes(Idx).hasAttribute(llvm::Attributes::ByVal)) {
#elif ISPC_LLVM_VERSION <= ISPC_LLVM_4_0
        if (PAL.getParamAttributes(Idx).hasAttribute(llvm::AttributeSet::FunctionIndex, llvm::Attribute::ByVal)) {
#else // LLVM 5.0+
        if (PAL.getParamAttributes(Idx).hasAttribute(llvm::Attribute::ByVal)) {
#endif
            assert(ArgTy->isPointerTy());
            ArgTy = llvm::cast<llvm::PointerType>(ArgTy)->getElementType();
        }
        printType(FunctionInnards, ArgTy,
#if ISPC_LLVM_VERSION == ISPC_LLVM_3_2
                  PAL.getParamAttributes(Idx).hasAttribute(llvm::Attributes::SExt),
#elif ISPC_LLVM_VERSION <= ISPC_LLVM_4_0
                  PAL.getParamAttributes(Idx).hasAttribute(llvm::AttributeSet::FunctionIndex, llvm::Attribute::SExt),
#else // LLVM 5.0+
                  PAL.getParamAttributes(Idx).hasAttribute(llvm::Attribute::SExt),
#endif
                  "");
        PrintedType = true;
    }
    if (FTy->isVarArg()) {
        if (!PrintedType)
            FunctionInnards << " int"; // dummy argument for empty vararg functs
        FunctionInnards << ", ...";
    } else if (!PrintedType) {
        FunctionInnards << "void";
    }
    FunctionInnards << ')';
    printType(Out, RetTy,
#if ISPC_LLVM_VERSION == ISPC_LLVM_3_2
              PAL.getParamAttributes(0).hasAttribute(llvm::Attributes::SExt),
#elif ISPC_LLVM_VERSION <= ISPC_LLVM_4_0
              PAL.getParamAttributes(0).hasAttribute(llvm::AttributeSet::ReturnIndex, llvm::Attribute::SExt),
#else // LLVM 5.0+
              PAL.getParamAttributes(0).hasAttribute(llvm::Attribute::SExt),
#endif
              FunctionInnards.str());
}

llvm::raw_ostream &CWriter::printSimpleType(llvm::raw_ostream &Out, llvm::Type *Ty, bool isSigned,
                                            const std::string &NameSoFar) {
    assert((Ty->isFloatingPointTy() || Ty->isX86_MMXTy() || Ty->isIntegerTy() || Ty->isVectorTy() || Ty->isVoidTy()) &&
           "Invalid type for printSimpleType");
    switch (Ty->getTypeID()) {
    case llvm::Type::VoidTyID:
        return Out << "void " << NameSoFar;
    case llvm::Type::IntegerTyID: {
        unsigned NumBits = llvm::cast<llvm::IntegerType>(Ty)->getBitWidth();
        if (NumBits == 1)
            return Out << "bool " << NameSoFar;
        else if (NumBits <= 8)
            return Out << (isSigned ? "" : "u") << "int8_t " << NameSoFar;
        else if (NumBits <= 16)
            return Out << (isSigned ? "" : "u") << "int16_t " << NameSoFar;
        else if (NumBits <= 32)
            return Out << (isSigned ? "" : "u") << "int32_t " << NameSoFar;
        else if (NumBits <= 64)
            return Out << (isSigned ? "" : "u") << "int64_t " << NameSoFar;
        else
            return Out << "iN<" << NumBits << "> " << NameSoFar;
    }
    case llvm::Type::FloatTyID:
        return Out << "float " << NameSoFar;
    case llvm::Type::DoubleTyID:
        return Out << "double " << NameSoFar;
    // Lacking emulation of FP80 on PPC, etc., we assume whichever of these is
    // present matches host 'long double'.
    case llvm::Type::X86_FP80TyID:
    case llvm::Type::PPC_FP128TyID:
    case llvm::Type::FP128TyID:
        return Out << "long double " << NameSoFar;

    case llvm::Type::X86_MMXTyID:
        return printSimpleType(Out, llvm::Type::getInt32Ty(Ty->getContext()), isSigned,
                               " __attribute__((vector_size(64))) " + NameSoFar);

    case llvm::Type::VectorTyID: {
        llvm::VectorType *VTy = llvm::cast<llvm::VectorType>(Ty);
#if 1
        const char *suffix = NULL;
        const llvm::Type *eltTy = VTy->getElementType();
        if (eltTy->isFloatTy())
            suffix = "f";
        else if (eltTy->isDoubleTy())
            suffix = "d";
        else {
            assert(eltTy->isIntegerTy());
            switch (eltTy->getPrimitiveSizeInBits()) {
            case 1:
                suffix = "i1";
                break;
            case 8:
                suffix = "i8";
                break;
            case 16:
                suffix = "i16";
                break;
            case 32:
                suffix = "i32";
                break;
            case 64:
                suffix = "i64";
                break;
            default:
                suffix = "iN";
                break;
            }
        }

        return Out << "__vec" << VTy->getNumElements() << "_" << suffix << " " << NameSoFar;
#else
        return printSimpleType(Out, VTy->getElementType(), isSigned,
                               " __attribute__((vector_size(" + utostr(TD->getTypeAllocSize(VTy)) + " ))) " +
                                   NameSoFar);
#endif
    }

    default:
#ifndef NDEBUG
        llvm::errs() << "Unknown primitive type: " << *Ty << "\n";
#endif
        llvm_unreachable(0);
    }
    return Out << "";
}

// Pass the Type* and the variable name and this prints out the variable
// declaration.
//
llvm::raw_ostream &CWriter::printType(llvm::raw_ostream &Out, llvm::Type *Ty, bool isSigned,
                                      const std::string &NameSoFar, bool IgnoreName,
#if ISPC_LLVM_VERSION == ISPC_LLVM_3_2
                                      const llvm::AttrListPtr &PAL
#elif ISPC_LLVM_VERSION <= ISPC_LLVM_4_0
                                      const llvm::AttributeSet &PAL
#else // LLVM 5.0+
                                      const llvm::AttributeList &PAL
#endif
) {

    if (Ty->isFloatingPointTy() || Ty->isX86_MMXTy() || Ty->isIntegerTy() || Ty->isVectorTy() || Ty->isVoidTy()) {
        printSimpleType(Out, Ty, isSigned, NameSoFar);
        return Out;
    }

    switch (Ty->getTypeID()) {
    case llvm::Type::FunctionTyID: {
        llvm::FunctionType *FTy = llvm::cast<llvm::FunctionType>(Ty);
        std::string tstr;
        llvm::raw_string_ostream FunctionInnards(tstr);
        FunctionInnards << " (" << NameSoFar << ") (";
        unsigned Idx = 1;
        for (llvm::FunctionType::param_iterator I = FTy->param_begin(), E = FTy->param_end(); I != E; ++I) {
            llvm::Type *ArgTy = *I;
#if ISPC_LLVM_VERSION == ISPC_LLVM_3_2
            if (PAL.getParamAttributes(Idx).hasAttribute(llvm::Attributes::ByVal)) {
#elif ISPC_LLVM_VERSION <= ISPC_LLVM_4_0
            if (PAL.getParamAttributes(Idx).hasAttribute(llvm::AttributeSet::FunctionIndex, llvm::Attribute::ByVal)) {
#else // LLVM 5.0+
            if (PAL.getParamAttributes(Idx).hasAttribute(llvm::Attribute::ByVal)) {
#endif
                assert(ArgTy->isPointerTy());
                ArgTy = llvm::cast<llvm::PointerType>(ArgTy)->getElementType();
            }
            if (I != FTy->param_begin())
                FunctionInnards << ", ";
            printType(FunctionInnards, ArgTy,
#if ISPC_LLVM_VERSION == ISPC_LLVM_3_2
                      PAL.getParamAttributes(Idx).hasAttribute(llvm::Attributes::SExt),
#elif ISPC_LLVM_VERSION <= ISPC_LLVM_4_0
                      PAL.getParamAttributes(Idx).hasAttribute(llvm::AttributeSet::FunctionIndex,
                                                               llvm::Attribute::SExt),
#else // LLVM 5.0+
                      PAL.getParamAttributes(Idx).hasAttribute(llvm::Attribute::SExt),
#endif
                      "");
            ++Idx;
        }
        if (FTy->isVarArg()) {
            if (!FTy->getNumParams())
                FunctionInnards << " int"; // dummy argument for empty vaarg functs
            FunctionInnards << ", ...";
        } else if (!FTy->getNumParams()) {
            FunctionInnards << "void";
        }
        FunctionInnards << ')';
        printType(Out, FTy->getReturnType(),
#if ISPC_LLVM_VERSION == ISPC_LLVM_3_2
                  PAL.getParamAttributes(0).hasAttribute(llvm::Attributes::SExt),
#elif ISPC_LLVM_VERSION <= ISPC_LLVM_4_0
                  PAL.getParamAttributes(0).hasAttribute(llvm::AttributeSet::ReturnIndex, llvm::Attribute::SExt),
#else // LLVM 5.0+
                  PAL.getParamAttributes(0).hasAttribute(llvm::Attribute::SExt),
#endif
                  FunctionInnards.str());
        return Out;
    }
    case llvm::Type::StructTyID: {
        llvm::StructType *STy = llvm::cast<llvm::StructType>(Ty);

        // Check to see if the type is named.
        if (!IgnoreName)
            return Out << getStructName(STy) << ' ' << NameSoFar;

        Out << "struct " << NameSoFar << " {\n";

        // print initialization func
        if (STy->getNumElements() > 0) {
            Out << "  static " << NameSoFar << " init(";
            unsigned Idx = 0;
            for (llvm::StructType::element_iterator I = STy->element_begin(), E = STy->element_end(); I != E;
                 ++I, ++Idx) {
                char buf[64];
                snprintf(buf, sizeof(buf), "v%d", Idx);
                printType(Out, *I, false, buf);
                if (Idx + 1 < STy->getNumElements())
                    Out << ", ";
            }
            Out << ") {\n";
            Out << "    " << NameSoFar << " ret;\n";
            for (Idx = 0; Idx < STy->getNumElements(); ++Idx)
                Out << "    ret.field" << Idx << " = v" << Idx << ";\n";
            Out << "    return ret;\n";
            Out << "  }\n";
        }

        unsigned Idx = 0;
        for (llvm::StructType::element_iterator I = STy->element_begin(), E = STy->element_end(); I != E; ++I) {
            Out << "  ";
            printType(Out, *I, false, "field" + llvm::utostr(Idx++));
            Out << ";\n";
        }
        Out << '}';
        if (STy->isPacked())
            Out << " __attribute__ ((packed))";
        return Out;
    }

    case llvm::Type::PointerTyID: {
        llvm::PointerType *PTy = llvm::cast<llvm::PointerType>(Ty);
        std::string ptrName = "*" + NameSoFar;

        if (PTy->getElementType()->isArrayTy() || PTy->getElementType()->isVectorTy())
            ptrName = "(" + ptrName + ")";

        if (!PAL.isEmpty())
            // Must be a function ptr cast!
            return printType(Out, PTy->getElementType(), false, ptrName, true, PAL);
        return printType(Out, PTy->getElementType(), false, ptrName);
    }

    case llvm::Type::ArrayTyID: {
        llvm::ArrayType *ATy = llvm::cast<llvm::ArrayType>(Ty);

        // Check to see if the type is named.
        if (!IgnoreName)
            return Out << getArrayName(ATy) << ' ' << NameSoFar;

        unsigned NumElements = (unsigned)ATy->getNumElements();
        if (NumElements == 0)
            NumElements = 1;
        // Arrays are wrapped in structs to allow them to have normal
        // value semantics (avoiding the array "decay").
        Out << "struct " << NameSoFar << " {\n";
        // init func
        Out << "  static " << NameSoFar << " init(";
        for (unsigned Idx = 0; Idx < NumElements; ++Idx) {
            char buf[64];
            snprintf(buf, sizeof(buf), "v%d", Idx);
            printType(Out, ATy->getElementType(), false, buf);
            if (Idx + 1 < NumElements)
                Out << ", ";
        }
        Out << ") {\n";
        Out << "    " << NameSoFar << " ret;\n";
        for (unsigned Idx = 0; Idx < NumElements; ++Idx)
            Out << "    ret.array[" << Idx << "] = v" << Idx << ";\n";
        Out << "    return ret;\n";
        Out << "  }\n  ";

        // if it's an array of i8s, also provide a version that takes a const
        // char *
        if (ATy->getElementType() == LLVMTypes::Int8Type) {
            Out << "  static " << NameSoFar << " init(const char *p) {\n";
            Out << "    " << NameSoFar << " ret;\n";
            Out << "    memcpy((uint8_t *)ret.array, (uint8_t *)p, " << NumElements << ");\n";
            Out << "    return ret;\n";
            Out << "  }\n";
        }

        printType(Out, ATy->getElementType(), false, "array[" + llvm::utostr(NumElements) + "]");
        return Out << ";\n} ";
    }

    default:
        llvm_unreachable("Unhandled case in getTypeProps!");
    }
    return Out << "";
}

void CWriter::printConstantArray(llvm::ConstantArray *CPA, bool Static) {
    // vec16_i64 should be handled separately

    if (is_vec16_i64_ty(CPA->getOperand(0)->getType())) {
        Out << "/* vec16_i64 should be loaded carefully on knc */";
        Out << "\n#if defined(KNC)\n";
        Out << "hilo2zmm";
        Out << "\n#endif\n";
    }
    Out << "(";
    printConstant(llvm::cast<llvm::Constant>(CPA->getOperand(0)), Static);
    Out << ")";

    for (unsigned i = 1, e = CPA->getNumOperands(); i != e; ++i) {
        Out << ", ";

        if (is_vec16_i64_ty(CPA->getOperand(i)->getType())) {
            Out << "/* vec16_i64 should be loaded carefully on knc */";
            Out << "\n#if defined(KNC) \n";
            Out << "hilo2zmm";
            Out << "\n#endif \n";
        }
        Out << "(";
        printConstant(llvm::cast<llvm::Constant>(CPA->getOperand(i)), Static);
        Out << ")";
    }
}

void CWriter::printConstantVector(llvm::ConstantVector *CP, bool Static) {
    printConstant(llvm::cast<llvm::Constant>(CP->getOperand(0)), Static);
    for (unsigned i = 1, e = CP->getNumOperands(); i != e; ++i) {
        Out << ", ";
        printConstant(llvm::cast<llvm::Constant>(CP->getOperand(i)), Static);
    }
}

void CWriter::printConstantDataSequential(llvm::ConstantDataSequential *CDS, bool Static) {
    // As a special case, print the array as a string if it is an array of
    // ubytes or an array of sbytes with positive values.
    //
    if (CDS->isCString()) {
        Out << '\"';
        // Keep track of whether the last number was a hexadecimal escape.
        bool LastWasHex = false;

        llvm::StringRef Bytes = CDS->getAsCString();

        // Do not include the last character, which we know is null
        for (unsigned i = 0, e = Bytes.size(); i != e; ++i) {
            unsigned char C = Bytes[i];

            // Print it out literally if it is a printable character.  The only thing
            // to be careful about is when the last letter output was a hex escape
            // code, in which case we have to be careful not to print out hex digits
            // explicitly (the C compiler thinks it is a continuation of the previous
            // character, sheesh...)
            //
            if (isprint(C) && (!LastWasHex || !isxdigit(C))) {
                LastWasHex = false;
                if (C == '"' || C == '\\')
                    Out << "\\" << (char)C;
                else
                    Out << (char)C;
            } else {
                LastWasHex = false;
                switch (C) {
                case '\n':
                    Out << "\\n";
                    break;
                case '\t':
                    Out << "\\t";
                    break;
                case '\r':
                    Out << "\\r";
                    break;
                case '\v':
                    Out << "\\v";
                    break;
                case '\a':
                    Out << "\\a";
                    break;
                case '\"':
                    Out << "\\\"";
                    break;
                case '\'':
                    Out << "\\\'";
                    break;
                default:
                    Out << "\\x";
                    Out << (char)((C / 16 < 10) ? (C / 16 + '0') : (C / 16 - 10 + 'A'));
                    Out << (char)(((C & 15) < 10) ? ((C & 15) + '0') : ((C & 15) - 10 + 'A'));
                    LastWasHex = true;
                    break;
                }
            }
        }
        Out << '\"';
    } else {
        printConstant(CDS->getElementAsConstant(0), Static);
        for (unsigned i = 1, e = CDS->getNumElements(); i != e; ++i) {
            Out << ", ";
            printConstant(CDS->getElementAsConstant(i), Static);
        }
    }
}

static inline std::string ftostr(const llvm::APFloat &V) {
    std::string Buf;
#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_9
    if (&V.getSemantics() == &llvm::APFloat::IEEEdouble) {
        llvm::raw_string_ostream(Buf) << V.convertToDouble();
        return Buf;
    } else if (&V.getSemantics() == &llvm::APFloat::IEEEsingle) {
        llvm::raw_string_ostream(Buf) << (double)V.convertToFloat();
        return Buf;
    }
#else // LLVM 4.0+
    if (&V.getSemantics() == &llvm::APFloat::IEEEdouble()) {
        llvm::raw_string_ostream(Buf) << V.convertToDouble();
        return Buf;
    } else if (&V.getSemantics() == &llvm::APFloat::IEEEsingle()) {
        llvm::raw_string_ostream(Buf) << (double)V.convertToFloat();
        return Buf;
    }
#endif
    return "<unknown format in ftostr>"; // error
}

// isFPCSafeToPrint - Returns true if we may assume that CFP may be written out
// textually as a double (rather than as a reference to a stack-allocated
// variable). We decide this by converting CFP to a string and back into a
// double, and then checking whether the conversion results in a bit-equal
// double to the original value of CFP. This depends on us and the target C
// compiler agreeing on the conversion process (which is pretty likely since we
// only deal in IEEE FP).
//
static bool isFPCSafeToPrint(const llvm::ConstantFP *CFP) {
    bool ignored;
    // Do long doubles in hex for now.
    if (CFP->getType() != llvm::Type::getFloatTy(CFP->getContext()) &&
        CFP->getType() != llvm::Type::getDoubleTy(CFP->getContext()))
        return false;
    llvm::APFloat APF = llvm::APFloat(CFP->getValueAPF()); // copy
    if (CFP->getType() == llvm::Type::getFloatTy(CFP->getContext()))
#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_9 // <= 3.9
        APF.convert(llvm::APFloat::IEEEdouble, llvm::APFloat::rmNearestTiesToEven, &ignored);
#else // LLVM 4.0+
        APF.convert(llvm::APFloat::IEEEdouble(), llvm::APFloat::rmNearestTiesToEven, &ignored);
#endif
#if HAVE_PRINTF_A && ENABLE_CBE_PRINTF_A
    char Buffer[100];
    snprintf(Buffer, sizeof(Buffer), "%a", APF.convertToDouble());
    if (!strncmp(Buffer, "0x", 2) || !strncmp(Buffer, "-0x", 3) || !strncmp(Buffer, "+0x", 3))
        return APF.bitwiseIsEqual(llvm::APFloat(atof(Buffer)));
    return false;
#else
    std::string StrVal = ftostr(APF);

    while (StrVal[0] == ' ')
        StrVal.erase(StrVal.begin());

    // Check to make sure that the stringized number is not some string like "Inf"
    // or NaN.  Check that the string matches the "[-+]?[0-9]" regex.
    if ((StrVal[0] >= '0' && StrVal[0] <= '9') ||
        ((StrVal[0] == '-' || StrVal[0] == '+') && (StrVal[1] >= '0' && StrVal[1] <= '9')))
        // Reparse stringized version!
        return APF.bitwiseIsEqual(llvm::APFloat(atof(StrVal.c_str())));
    return false;
#endif
}

/// Print out the casting for a cast operation. This does the double casting
/// necessary for conversion to the destination type, if necessary.
/// Return value indicates whether a closing paren is needed.
/// @brief Print a cast
bool CWriter::printCast(unsigned opc, llvm::Type *SrcTy, llvm::Type *DstTy) {
    if (llvm::isa<const llvm::VectorType>(DstTy)) {
        assert(llvm::isa<const llvm::VectorType>(SrcTy));
        switch (opc) {
        case llvm::Instruction::UIToFP:
            Out << "__cast_uitofp(";
            break;
        case llvm::Instruction::SIToFP:
            Out << "__cast_sitofp(";
            break;
        case llvm::Instruction::IntToPtr:
            llvm_unreachable("Invalid vector cast");
        case llvm::Instruction::Trunc:
            Out << "__cast_trunc(";
            break;
        case llvm::Instruction::BitCast:
            Out << "__cast_bits(";
            break;
        case llvm::Instruction::FPExt:
            Out << "__cast_fpext(";
            break;
        case llvm::Instruction::FPTrunc:
            Out << "__cast_fptrunc(";
            break;
        case llvm::Instruction::ZExt:
            Out << "__cast_zext(";
            break;
        case llvm::Instruction::PtrToInt:
            llvm_unreachable("Invalid vector cast");
        case llvm::Instruction::FPToUI:
            Out << "__cast_fptoui(";
            break;
        case llvm::Instruction::SExt:
            Out << "__cast_sext(";
            break;
        case llvm::Instruction::FPToSI:
            Out << "__cast_fptosi(";
            break;
        default:
            llvm_unreachable("Invalid cast opcode");
        }

        // print a call to the constructor for the destination type for the
        // first arg; this bogus first parameter is only used to convey the
        // desired return type to the callee.
        printType(Out, DstTy);
        Out << "(), ";

        return true;
    }

    // Print the destination type cast
    switch (opc) {
    case llvm::Instruction::BitCast: {
        if (DstTy->isPointerTy()) {
            Out << '(';
            printType(Out, DstTy);
            Out << ')';
            break;
        } else {
            Out << "__cast_bits((";
            printType(Out, DstTy);
            Out << ")0, ";
            return true;
        }
    }
    case llvm::Instruction::UIToFP:
    case llvm::Instruction::SIToFP:
    case llvm::Instruction::IntToPtr:
    case llvm::Instruction::Trunc:
    case llvm::Instruction::FPExt:
    case llvm::Instruction::FPTrunc: // For these the DstTy sign doesn't matter
        Out << '(';
        printType(Out, DstTy);
        Out << ')';
        break;
    case llvm::Instruction::ZExt:
    case llvm::Instruction::PtrToInt:
    case llvm::Instruction::FPToUI: // For these, make sure we get an unsigned dest
        Out << '(';
        printSimpleType(Out, DstTy, false);
        Out << ')';
        break;
    case llvm::Instruction::SExt:
    case llvm::Instruction::FPToSI: // For these, make sure we get a signed dest
        Out << '(';
        printSimpleType(Out, DstTy, true);
        Out << ')';
        break;
    default:
        llvm_unreachable("Invalid cast opcode");
    }

    // Print the source type cast
    switch (opc) {
    case llvm::Instruction::UIToFP:
    case llvm::Instruction::ZExt:
        Out << '(';
        printSimpleType(Out, SrcTy, false);
        Out << ')';
        break;
    case llvm::Instruction::SIToFP:
    case llvm::Instruction::SExt:
        Out << '(';
        printSimpleType(Out, SrcTy, true);
        Out << ')';
        break;
    case llvm::Instruction::IntToPtr:
    case llvm::Instruction::PtrToInt:
        // Avoid "cast to pointer from integer of different size" warnings
        Out << "(unsigned long)";
        break;
    case llvm::Instruction::Trunc:
    case llvm::Instruction::BitCast:
    case llvm::Instruction::FPExt:
    case llvm::Instruction::FPTrunc:
    case llvm::Instruction::FPToSI:
    case llvm::Instruction::FPToUI:
        break; // These don't need a source cast.
    default:
        llvm_unreachable("Invalid cast opcode");
        break;
    }
    return false;
}

/** Construct the name of a function with the given base and returning a
    vector of a given type, of the specified idth.  For example, if base
    is "foo" and matchType is i32 and width is 16, this will return the
    string "__foo_i32<__vec16_i32>".
 */
static const char *lGetTypedFunc(const char *base, llvm::Type *matchType, int width) {
    static const char *ty_desc_str[] = {"f", "d", "i1", "i8", "i16", "i32", "i64"};
    static const char *fn_desc_str[] = {"float", "double", "i1", "i8", "i16", "i32", "i64"};
    enum { DESC_FLOAT, DESC_DOUBLE, DESC_I1, DESC_I8, DESC_I16, DESC_I32, DESC_I64 } desc;

    switch (matchType->getTypeID()) {
    case llvm::Type::FloatTyID:
        desc = DESC_FLOAT;
        break;
    case llvm::Type::DoubleTyID:
        desc = DESC_DOUBLE;
        break;
    case llvm::Type::IntegerTyID: {
        switch (llvm::cast<llvm::IntegerType>(matchType)->getBitWidth()) {
        case 1:
            desc = DESC_I1;
            break;
        case 8:
            desc = DESC_I8;
            break;
        case 16:
            desc = DESC_I16;
            break;
        case 32:
            desc = DESC_I32;
            break;
        case 64:
            desc = DESC_I64;
            break;
        default:
            return NULL;
        }
        break;
    }
    default:
        return NULL;
    }

    char buf[64];
    snprintf(buf, 64, "__%s_%s<__vec%d_%s>", base, fn_desc_str[desc], width, ty_desc_str[desc]);
    return strdup(buf);
}

// printConstant - The LLVM Constant to C Constant converter.
void CWriter::printConstant(llvm::Constant *CPV, bool Static) {
    if (const llvm::ConstantExpr *CE = llvm::dyn_cast<llvm::ConstantExpr>(CPV)) {
        if (llvm::isa<llvm::VectorType>(CPV->getType())) {
            assert(CE->getOpcode() == llvm::Instruction::BitCast);
            llvm::ConstantExpr *Op = llvm::dyn_cast<llvm::ConstantExpr>(CE->getOperand(0));
            assert(Op && Op->getOpcode() == llvm::Instruction::BitCast);
            assert(llvm::isa<llvm::VectorType>(Op->getOperand(0)->getType()));

            Out << "(__cast_bits(";
            printType(Out, CE->getType());
            Out << "(), ";
            printConstant(Op->getOperand(0), Static);
            Out << "))";
            return;
        }
        switch (CE->getOpcode()) {
        case llvm::Instruction::Trunc:
        case llvm::Instruction::ZExt:
        case llvm::Instruction::SExt:
        case llvm::Instruction::FPTrunc:
        case llvm::Instruction::FPExt:
        case llvm::Instruction::UIToFP:
        case llvm::Instruction::SIToFP:
        case llvm::Instruction::FPToUI:
        case llvm::Instruction::FPToSI:
        case llvm::Instruction::PtrToInt:
        case llvm::Instruction::IntToPtr:
        case llvm::Instruction::BitCast: {
            if (CE->getOpcode() == llvm::Instruction::BitCast && CE->getType()->isPointerTy() == false) {
                Out << "__cast_bits((";
                printType(Out, CE->getType());
                Out << ")0, ";
                printConstant(CE->getOperand(0), Static);
                Out << ")";
                return;
            }

            Out << "(";
            bool closeParen = printCast(CE->getOpcode(), CE->getOperand(0)->getType(), CE->getType());
            if (CE->getOpcode() == llvm::Instruction::SExt &&
                CE->getOperand(0)->getType() == llvm::Type::getInt1Ty(CPV->getContext())) {
                // Make sure we really sext from bool here by subtracting from 0
                Out << "0-";
            }
            printConstant(CE->getOperand(0), Static);
            if (CE->getType() == llvm::Type::getInt1Ty(CPV->getContext()) &&
                (CE->getOpcode() == llvm::Instruction::Trunc || CE->getOpcode() == llvm::Instruction::FPToUI ||
                 CE->getOpcode() == llvm::Instruction::FPToSI || CE->getOpcode() == llvm::Instruction::PtrToInt)) {
                // Make sure we really truncate to bool here by anding with 1
                Out << "&1u";
            }
            Out << ')';
            if (closeParen)
                Out << ')';
            return;
        }
        case llvm::Instruction::GetElementPtr:
            assert(!llvm::isa<llvm::VectorType>(CPV->getType()));
            Out << "(";
            printGEPExpression(CE->getOperand(0), gep_type_begin(CPV), gep_type_end(CPV), Static);
            Out << ")";
            return;
        case llvm::Instruction::Select:
            assert(!llvm::isa<llvm::VectorType>(CPV->getType()));
            Out << '(';
            printConstant(CE->getOperand(0), Static);
            Out << '?';
            printConstant(CE->getOperand(1), Static);
            Out << ':';
            printConstant(CE->getOperand(2), Static);
            Out << ')';
            return;
        case llvm::Instruction::Add:
        case llvm::Instruction::FAdd:
        case llvm::Instruction::Sub:
        case llvm::Instruction::FSub:
        case llvm::Instruction::Mul:
        case llvm::Instruction::FMul:
        case llvm::Instruction::SDiv:
        case llvm::Instruction::UDiv:
        case llvm::Instruction::FDiv:
        case llvm::Instruction::URem:
        case llvm::Instruction::SRem:
        case llvm::Instruction::FRem:
        case llvm::Instruction::And:
        case llvm::Instruction::Or:
        case llvm::Instruction::Xor:
        case llvm::Instruction::ICmp:
        case llvm::Instruction::Shl:
        case llvm::Instruction::LShr:
        case llvm::Instruction::AShr: {
            assert(!llvm::isa<llvm::VectorType>(CPV->getType()));
            Out << '(';
            bool NeedsClosingParens = printConstExprCast(CE, Static);
            printConstantWithCast(CE->getOperand(0), CE->getOpcode());
            switch (CE->getOpcode()) {
            case llvm::Instruction::Add:
            case llvm::Instruction::FAdd:
                Out << " + ";
                break;
            case llvm::Instruction::Sub:
            case llvm::Instruction::FSub:
                Out << " - ";
                break;
            case llvm::Instruction::Mul:
            case llvm::Instruction::FMul:
                Out << " * ";
                break;
            case llvm::Instruction::URem:
            case llvm::Instruction::SRem:
            case llvm::Instruction::FRem:
                Out << " % ";
                break;
            case llvm::Instruction::UDiv:
            case llvm::Instruction::SDiv:
            case llvm::Instruction::FDiv:
                Out << " / ";
                break;
            case llvm::Instruction::And:
                Out << " & ";
                break;
            case llvm::Instruction::Or:
                Out << " | ";
                break;
            case llvm::Instruction::Xor:
                Out << " ^ ";
                break;
            case llvm::Instruction::Shl:
                Out << " << ";
                break;
            case llvm::Instruction::LShr:
            case llvm::Instruction::AShr:
                Out << " >> ";
                break;
            case llvm::Instruction::ICmp:
                switch (CE->getPredicate()) {
                case llvm::ICmpInst::ICMP_EQ:
                    Out << " == ";
                    break;
                case llvm::ICmpInst::ICMP_NE:
                    Out << " != ";
                    break;
                case llvm::ICmpInst::ICMP_SLT:
                case llvm::ICmpInst::ICMP_ULT:
                    Out << " < ";
                    break;
                case llvm::ICmpInst::ICMP_SLE:
                case llvm::ICmpInst::ICMP_ULE:
                    Out << " <= ";
                    break;
                case llvm::ICmpInst::ICMP_SGT:
                case llvm::ICmpInst::ICMP_UGT:
                    Out << " > ";
                    break;
                case llvm::ICmpInst::ICMP_SGE:
                case llvm::ICmpInst::ICMP_UGE:
                    Out << " >= ";
                    break;
                default:
                    llvm_unreachable("Illegal ICmp predicate");
                }
                break;
            default:
                llvm_unreachable("Illegal opcode here!");
            }
            printConstantWithCast(CE->getOperand(1), CE->getOpcode());
            if (NeedsClosingParens)
                Out << "))";
            Out << ')';
            return;
        }
        case llvm::Instruction::FCmp: {
            assert(!llvm::isa<llvm::VectorType>(CPV->getType()));
            Out << '(';
            bool NeedsClosingParens = printConstExprCast(CE, Static);
            if (CE->getPredicate() == llvm::FCmpInst::FCMP_FALSE)
                Out << "0";
            else if (CE->getPredicate() == llvm::FCmpInst::FCMP_TRUE)
                Out << "1";
            else {
                const char *op = 0;
                switch (CE->getPredicate()) {
                default:
                    llvm_unreachable("Illegal FCmp predicate");
                case llvm::FCmpInst::FCMP_ORD:
                    op = "ord";
                    break;
                case llvm::FCmpInst::FCMP_UNO:
                    op = "uno";
                    break;
                case llvm::FCmpInst::FCMP_UEQ:
                    op = "ueq";
                    break;
                case llvm::FCmpInst::FCMP_UNE:
                    op = "une";
                    break;
                case llvm::FCmpInst::FCMP_ULT:
                    op = "ult";
                    break;
                case llvm::FCmpInst::FCMP_ULE:
                    op = "ule";
                    break;
                case llvm::FCmpInst::FCMP_UGT:
                    op = "ugt";
                    break;
                case llvm::FCmpInst::FCMP_UGE:
                    op = "uge";
                    break;
                case llvm::FCmpInst::FCMP_OEQ:
                    op = "oeq";
                    break;
                case llvm::FCmpInst::FCMP_ONE:
                    op = "one";
                    break;
                case llvm::FCmpInst::FCMP_OLT:
                    op = "olt";
                    break;
                case llvm::FCmpInst::FCMP_OLE:
                    op = "ole";
                    break;
                case llvm::FCmpInst::FCMP_OGT:
                    op = "ogt";
                    break;
                case llvm::FCmpInst::FCMP_OGE:
                    op = "oge";
                    break;
                }
                Out << "llvm_fcmp_" << op << "(";
                printConstantWithCast(CE->getOperand(0), CE->getOpcode());
                Out << ", ";
                printConstantWithCast(CE->getOperand(1), CE->getOpcode());
                Out << ")";
            }
            if (NeedsClosingParens)
                Out << "))";
            Out << ')';
            return;
        }
        default:
#ifndef NDEBUG
            llvm::errs() << "CWriter Error: Unhandled constant expression: " << *CE << "\n";
#endif
            llvm_unreachable(0);
        }
    } else if (llvm::isa<llvm::UndefValue>(CPV) && CPV->getType()->isSingleValueType()) {
        if (CPV->getType()->isVectorTy()) {
            printType(Out, CPV->getType());
            Out << "( /* UNDEF */)";
            return;
        }

        Out << "((";
        printType(Out, CPV->getType()); // sign doesn't matter
        Out << ")/*UNDEF*/";
        Out << "0)";
        return;
    }

    if (llvm::ConstantInt *CI = llvm::dyn_cast<llvm::ConstantInt>(CPV)) {
        llvm::Type *Ty = CI->getType();
        if (Ty == llvm::Type::getInt1Ty(CPV->getContext()))
            Out << (CI->getZExtValue() ? '1' : '0');
        else if (Ty == llvm::Type::getInt32Ty(CPV->getContext()))
            Out << CI->getZExtValue() << 'u';
        else if (Ty == llvm::Type::getInt64Ty(CPV->getContext()))
            Out << CI->getZExtValue() << "ull";
        else if (Ty->getPrimitiveSizeInBits() > 64) {
            Out << "\"";
            // const uint64_t *Ptr64 = CPV->getUniqueInteger().getRawData();
            const uint64_t *Ptr64 = CI->getValue().getRawData();
            for (unsigned i = 0; i < Ty->getPrimitiveSizeInBits(); i++) {
                Out << ((Ptr64[i / (sizeof(uint64_t) * 8)] >> (i % (sizeof(uint64_t) * 8))) & 1);
            }
            Out << "\"";
        } else {
            Out << "((";
            printSimpleType(Out, Ty, false) << ')';
            if (CI->isMinValue(true))
                Out << CI->getZExtValue() << 'u';
            else
                Out << CI->getSExtValue();
            Out << ')';
        }
        return;
    }

    switch (CPV->getType()->getTypeID()) {
    case llvm::Type::FloatTyID:
    case llvm::Type::DoubleTyID:
    case llvm::Type::X86_FP80TyID:
    case llvm::Type::PPC_FP128TyID:
    case llvm::Type::FP128TyID: {
        llvm::ConstantFP *FPC = llvm::cast<llvm::ConstantFP>(CPV);
        std::map<const llvm::ConstantFP *, unsigned>::iterator I = FPConstantMap.find(FPC);
        if (I != FPConstantMap.end()) {
            // Because of FP precision problems we must load from a stack allocated
            // value that holds the value in hex.
            Out << "(*("
                << (FPC->getType() == llvm::Type::getFloatTy(CPV->getContext())
                        ? "float"
                        : FPC->getType() == llvm::Type::getDoubleTy(CPV->getContext()) ? "double" : "long double")
                << "*)&FPConstant" << I->second << ')';
        } else {
            double V;
            if (FPC->getType() == llvm::Type::getFloatTy(CPV->getContext()))
                V = FPC->getValueAPF().convertToFloat();
            else if (FPC->getType() == llvm::Type::getDoubleTy(CPV->getContext()))
                V = FPC->getValueAPF().convertToDouble();
            else {
                // Long double.  Convert the number to double, discarding precision.
                // This is not awesome, but it at least makes the CBE output somewhat
                // useful.
                llvm::APFloat Tmp = FPC->getValueAPF();
                bool LosesInfo;
#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_9 // <= 3.9
                Tmp.convert(llvm::APFloat::IEEEdouble, llvm::APFloat::rmTowardZero, &LosesInfo);
#else // LLVM 4.0+
                Tmp.convert(llvm::APFloat::IEEEdouble(), llvm::APFloat::rmTowardZero, &LosesInfo);
#endif
                V = Tmp.convertToDouble();
            }

            if (std::isnan(V)) {
                // The value is NaN

                // FIXME the actual NaN bits should be emitted.
                // The prefix for a quiet NaN is 0x7FF8. For a signalling NaN,
                // it's 0x7ff4.
                const unsigned long QuietNaN = 0x7ff8UL;
                // const unsigned long SignalNaN = 0x7ff4UL;

                // We need to grab the first part of the FP #
                char Buffer[100];

                uint64_t ll = llvm::DoubleToBits(V);
                snprintf(Buffer, sizeof(Buffer), "0x%" PRIx64, ll);

                std::string Num(&Buffer[0], &Buffer[6]);
                unsigned long Val = strtoul(Num.c_str(), 0, 16);

                if (FPC->getType() == llvm::Type::getFloatTy(FPC->getContext()))
                    Out << "LLVM_NAN" << (Val == QuietNaN ? "" : "S") << "F(\"" << Buffer << "\") /*nan*/ ";
                else
                    Out << "LLVM_NAN" << (Val == QuietNaN ? "" : "S") << "(\"" << Buffer << "\") /*nan*/ ";
            } else if (std::isinf(V)) {
                // The value is Inf
                if (V < 0)
                    Out << '-';
                Out << "LLVM_INF" << (FPC->getType() == llvm::Type::getFloatTy(FPC->getContext()) ? "F" : "")
                    << " /*inf*/ ";
            } else {
                std::string Num;
#if HAVE_PRINTF_A && ENABLE_CBE_PRINTF_A
                // Print out the constant as a floating point number.
                char Buffer[100];
                snprintf(Buffer, sizeof(Buffer), "%a", V);
                Num = Buffer;
#else
                Num = ftostr(FPC->getValueAPF());
#endif
                Out << Num;
            }
        }
        break;
    }

    case llvm::Type::ArrayTyID: {
        llvm::ArrayType *AT = llvm::cast<llvm::ArrayType>(CPV->getType());
        if (Static)
            // arrays are wrapped in structs...
            Out << "{ ";
        else {
            // call init func of the struct it's wrapped in...
            printType(Out, CPV->getType());
            Out << "::init (";
        }
        if (llvm::ConstantArray *CA = llvm::dyn_cast<llvm::ConstantArray>(CPV)) {
            printConstantArray(CA, Static);
        } else if (llvm::ConstantDataSequential *CDS = llvm::dyn_cast<llvm::ConstantDataSequential>(CPV)) {
            printConstantDataSequential(CDS, Static);
        } else {
            assert(llvm::isa<llvm::ConstantAggregateZero>(CPV) || llvm::isa<llvm::UndefValue>(CPV));
            if (AT->getNumElements()) {
                Out << ' ';
                llvm::Constant *CZ = llvm::Constant::getNullValue(AT->getElementType());
                printConstant(CZ, Static);
                for (unsigned i = 1, e = (unsigned)AT->getNumElements(); i != e; ++i) {
                    Out << ", ";
                    printConstant(CZ, Static);
                }
            }
        }
        if (Static)
            Out << " }";
        else
            Out << ")";
        break;
    }
    case llvm::Type::VectorTyID: {
        llvm::VectorType *VT = llvm::dyn_cast<llvm::VectorType>(CPV->getType());

        if (llvm::isa<llvm::ConstantAggregateZero>(CPV)) {
            // All zeros; call the __setzero_* function.
            const char *setZeroFunc = lGetTypedFunc("setzero", VT->getElementType(), vectorWidth);
            assert(setZeroFunc != NULL);
            Out << setZeroFunc << "()";
        } else if (llvm::isa<llvm::UndefValue>(CPV)) {
            // Undefined value; call __undef_* so that we can potentially pass
            // this information along..
            const char *undefFunc = lGetTypedFunc("undef", VT->getElementType(), vectorWidth);
            assert(undefFunc != NULL);
            Out << undefFunc << "()";
        } else {
            const char *smearFunc = lGetTypedFunc("smear", VT->getElementType(), vectorWidth);

            if (llvm::ConstantVector *CV = llvm::dyn_cast<llvm::ConstantVector>(CPV)) {
                llvm::Constant *splatValue = CV->getSplatValue();
                if (splatValue != NULL && smearFunc != NULL) {
                    // If it's a basic type and has a __smear_* function, then
                    // call that.
                    Out << smearFunc << "(";
                    printConstant(splatValue, Static);
                    Out << ")";
                } else {
                    // Otherwise call the constructor for the type
                    printType(Out, CPV->getType());
                    Out << "(";
                    printConstantVector(CV, Static);
                    Out << ")";
                }
            } else if (llvm::ConstantDataVector *CDV = llvm::dyn_cast<llvm::ConstantDataVector>(CPV)) {
                llvm::Constant *splatValue = CDV->getSplatValue();
                if (splatValue != NULL && smearFunc != NULL) {
                    Out << smearFunc << "(";
                    printConstant(splatValue, Static);
                    Out << ")";
                } else if (VectorConstantMap.find(CDV) != VectorConstantMap.end()) {
                    // If we have emitted an static const array with the
                    // vector's values, just load from it.
                    unsigned index = VectorConstantMap[CDV];
                    int alignment = 4 * std::min(vectorWidth, 16);

                    Out << "__load<" << alignment << ">(";

                    // Cast the pointer to the array of element values to a
                    // pointer to the vector type.
                    Out << "(const ";
                    printSimpleType(Out, CDV->getType(), true, "");
                    Out << " *)";

                    Out << "(VectorConstant" << index << "))";
                } else {
                    printType(Out, CPV->getType());
                    Out << "(";
                    printConstantDataSequential(CDV, Static);
                    Out << ")";
                }
            } else {
                llvm::report_fatal_error("Unexpected vector type");
            }
        }

        break;
    }
    case llvm::Type::StructTyID:
        if (!Static) {
            // call init func...
            printType(Out, CPV->getType());
            Out << "::init";
        }
        if (llvm::isa<llvm::ConstantAggregateZero>(CPV) || llvm::isa<llvm::UndefValue>(CPV)) {
            llvm::StructType *ST = llvm::cast<llvm::StructType>(CPV->getType());
            Out << '(';
            if (ST->getNumElements()) {
                Out << ' ';
                printConstant(llvm::Constant::getNullValue(ST->getElementType(0)), Static);
                for (unsigned i = 1, e = ST->getNumElements(); i != e; ++i) {
                    Out << ", ";
                    printConstant(llvm::Constant::getNullValue(ST->getElementType(i)), Static);
                }
            }
            Out << ')';
        } else {
            Out << '(';
            if (CPV->getNumOperands()) {
                // It is a kludge. It is needed because we cannot support short vectors
                // when generating code for knl-generic in multitarget mode.
                // Short vectors are mapped to "native" vectors and cause AVX-512 code
                // generation in static block initialization (__vec16_* in ::init function).
                bool isGenericKNL = g->target->getISA() == Target::GENERIC &&
                                    !g->target->getTreatGenericAsSmth().empty() && g->mangleFunctionsWithTarget;
                if (isGenericKNL && CPV->getOperand(0)->getType()->isVectorTy())
                    llvm::report_fatal_error("knl-generic-* target doesn's support short vectors");
                Out << ' ';
                printConstant(llvm::cast<llvm::Constant>(CPV->getOperand(0)), Static);
                for (unsigned i = 1, e = CPV->getNumOperands(); i != e; ++i) {
                    Out << ", ";
                    if (isGenericKNL && CPV->getOperand(i)->getType()->isVectorTy())
                        llvm::report_fatal_error("knl-generic-* target doesn's support short vectors");
                    printConstant(llvm::cast<llvm::Constant>(CPV->getOperand(i)), Static);
                }
            }
            Out << ')';
        }
        break;

    case llvm::Type::PointerTyID:
        if (llvm::isa<llvm::ConstantPointerNull>(CPV)) {
            Out << "((";
            printType(Out, CPV->getType()); // sign doesn't matter
            Out << ")/*NULL*/0)";
            break;
        } else if (llvm::GlobalValue *GV = llvm::dyn_cast<llvm::GlobalValue>(CPV)) {
            writeOperand(GV, Static);
            break;
        }
        // FALL THROUGH
    default:
#ifndef NDEBUG
        llvm::errs() << "Unknown constant type: " << *CPV << "\n";
#endif
        llvm_unreachable(0);
    }
}

// Some constant expressions need to be casted back to the original types
// because their operands were casted to the expected type. This function takes
// care of detecting that case and printing the cast for the ConstantExpr.
bool CWriter::printConstExprCast(const llvm::ConstantExpr *CE, bool Static) {
    bool NeedsExplicitCast = false;
    llvm::Type *Ty = CE->getOperand(0)->getType();
    bool TypeIsSigned = false;
    switch (CE->getOpcode()) {
    case llvm::Instruction::Add:
    case llvm::Instruction::Sub:
    case llvm::Instruction::Mul:
        // We need to cast integer arithmetic so that it is always performed
        // as unsigned, to avoid undefined behavior on overflow.
    case llvm::Instruction::LShr:
    case llvm::Instruction::URem:
    case llvm::Instruction::UDiv:
        NeedsExplicitCast = true;
        break;
    case llvm::Instruction::AShr:
    case llvm::Instruction::SRem:
    case llvm::Instruction::SDiv:
        NeedsExplicitCast = true;
        TypeIsSigned = true;
        break;
    case llvm::Instruction::SExt:
        Ty = CE->getType();
        NeedsExplicitCast = true;
        TypeIsSigned = true;
        break;
    case llvm::Instruction::ZExt:
    case llvm::Instruction::Trunc:
    case llvm::Instruction::FPTrunc:
    case llvm::Instruction::FPExt:
    case llvm::Instruction::UIToFP:
    case llvm::Instruction::SIToFP:
    case llvm::Instruction::FPToUI:
    case llvm::Instruction::FPToSI:
    case llvm::Instruction::PtrToInt:
    case llvm::Instruction::IntToPtr:
    case llvm::Instruction::BitCast:
        Ty = CE->getType();
        NeedsExplicitCast = true;
        break;
    default:
        break;
    }
    if (NeedsExplicitCast) {
        Out << "((";
        if (Ty->isIntegerTy() && Ty != llvm::Type::getInt1Ty(Ty->getContext()))
            printSimpleType(Out, Ty, TypeIsSigned);
        else
            printType(Out, Ty); // not integer, sign doesn't matter
        Out << ")(";
    }
    return NeedsExplicitCast;
}

//  Print a constant assuming that it is the operand for a given Opcode. The
//  opcodes that care about sign need to cast their operands to the expected
//  type before the operation proceeds. This function does the casting.
void CWriter::printConstantWithCast(llvm::Constant *CPV, unsigned Opcode) {

    // Extract the operand's type, we'll need it.
    llvm::Type *OpTy = CPV->getType();

    // Indicate whether to do the cast or not.
    bool shouldCast = false;
    bool typeIsSigned = false;

    // Based on the Opcode for which this Constant is being written, determine
    // the new type to which the operand should be casted by setting the value
    // of OpTy. If we change OpTy, also set shouldCast to true so it gets
    // casted below.
    switch (Opcode) {
    default:
        // for most instructions, it doesn't matter
        break;
    case llvm::Instruction::Add:
    case llvm::Instruction::Sub:
    case llvm::Instruction::Mul:
        // We need to cast integer arithmetic so that it is always performed
        // as unsigned, to avoid undefined behavior on overflow.
    case llvm::Instruction::LShr:
    case llvm::Instruction::UDiv:
    case llvm::Instruction::URem:
        shouldCast = true;
        break;
    case llvm::Instruction::AShr:
    case llvm::Instruction::SDiv:
    case llvm::Instruction::SRem:
        shouldCast = true;
        typeIsSigned = true;
        break;
    }

    // Write out the casted constant if we should, otherwise just write the
    // operand.
    if (shouldCast) {
        Out << "((";
        printSimpleType(Out, OpTy, typeIsSigned);
        Out << ")";
        printConstant(CPV, false);
        Out << ")";
    } else
        printConstant(CPV, false);
}

std::string CWriter::GetValueName(const llvm::Value *Operand) {

    // Resolve potential alias.
    if (const llvm::GlobalAlias *GA = llvm::dyn_cast<llvm::GlobalAlias>(Operand)) {
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_5 /* LLVM 3.5+ */
        if (const llvm::Value *V = GA->getAliasee())
#else /* <= LLVM 3.4 */
        if (const llvm::Value *V = GA->resolveAliasedGlobal(false))
#endif
            Operand = V;
    }

    // Mangle globals with the standard mangler interface for LLC compatibility.
    if (const llvm::GlobalValue *GV = llvm::dyn_cast<llvm::GlobalValue>(Operand)) {
        (void)GV;
        // llvm::SmallString<128> Str;
        // Mang->getNameWithPrefix(Str, GV, false);
        // return CBEMangle(Str.str().str());
        return CBEMangle(Operand->getName().str().c_str());
    }

    std::string Name = Operand->getName();

    if (Name.empty()) { // Assign unique names to local temporaries.
        unsigned &No = AnonValueNumbers[Operand];
        if (No == 0)
            No = ++NextAnonValueNumber;
        Name = "tmp__" + llvm::utostr(No);
    }

    std::string VarName;
    VarName.reserve(Name.capacity());

    for (std::string::iterator I = Name.begin(), E = Name.end(); I != E; ++I) {
        char ch = *I;

        if (!((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '_')) {
            char buffer[5];
            snprintf(buffer, sizeof(buffer), "_%x_", ch);
            VarName += buffer;
        } else
            VarName += ch;
    }

    if (llvm::isa<llvm::BasicBlock>(Operand))
        VarName += "_label";
    else
        VarName += "_";

    return VarName;
}

/// writeInstComputationInline - Emit the computation for the specified
/// instruction inline, with no destination provided.
void CWriter::writeInstComputationInline(llvm::Instruction &I) {
    // If this is a non-trivial bool computation, make sure to truncate down to
    // a 1 bit value.  This is important because we want "add i1 x, y" to return
    // "0" when x and y are true, not "2" for example.
    bool NeedBoolTrunc = false;
    if (I.getType() == llvm::Type::getInt1Ty(I.getContext()) && !llvm::isa<llvm::ICmpInst>(I) &&
        !llvm::isa<llvm::FCmpInst>(I))
        NeedBoolTrunc = true;

    if (NeedBoolTrunc)
        Out << "((";

    visit(I);

    if (NeedBoolTrunc)
        Out << ")&1)";
}

void CWriter::writeOperandInternal(llvm::Value *Operand, bool Static) {
    if (llvm::Instruction *I = llvm::dyn_cast<llvm::Instruction>(Operand))
        // Should we inline this instruction to build a tree?
        if (isInlinableInst(*I) && !isDirectAlloca(I)) {
            Out << '(';
            writeInstComputationInline(*I);
            Out << ')';
            return;
        }

    llvm::Constant *CPV = llvm::dyn_cast<llvm::Constant>(Operand);

    if (CPV && !llvm::isa<llvm::GlobalValue>(CPV))
        printConstant(CPV, Static);
    else
        Out << GetValueName(Operand);
}

void CWriter::writeOperand(llvm::Value *Operand, bool Static) {
    bool isAddressImplicit = isAddressExposed(Operand);
    if (isAddressImplicit)
        Out << "(&"; // Global variables are referenced as their addresses by llvm

    writeOperandInternal(Operand, Static);

    if (isAddressImplicit)
        Out << ')';
}

// Some instructions need to have their result value casted back to the
// original types because their operands were casted to the expected type.
// This function takes care of detecting that case and printing the cast
// for the Instruction.
bool CWriter::writeInstructionCast(const llvm::Instruction &I) {
    llvm::Type *Ty = I.getOperand(0)->getType();
    switch (I.getOpcode()) {
    case llvm::Instruction::Add:
    case llvm::Instruction::Sub:
    case llvm::Instruction::Mul:
        // We need to cast integer arithmetic so that it is always performed
        // as unsigned, to avoid undefined behavior on overflow.
    case llvm::Instruction::LShr:
    case llvm::Instruction::URem:
    case llvm::Instruction::UDiv:
        Out << "((";
        printSimpleType(Out, Ty, false);
        Out << ")(";
        return true;
    case llvm::Instruction::AShr:
    case llvm::Instruction::SRem:
    case llvm::Instruction::SDiv:
        Out << "((";
        printSimpleType(Out, Ty, true);
        Out << ")(";
        return true;
    default:
        break;
    }
    return false;
}

// Write the operand with a cast to another type based on the Opcode being used.
// This will be used in cases where an instruction has specific type
// requirements (usually signedness) for its operands.
void CWriter::writeOperandWithCast(llvm::Value *Operand, unsigned Opcode) {

    // Extract the operand's type, we'll need it.
    llvm::Type *OpTy = Operand->getType();

    // Indicate whether to do the cast or not.
    bool shouldCast = false;

    // Indicate whether the cast should be to a signed type or not.
    bool castIsSigned = false;

    // Based on the Opcode for which this Operand is being written, determine
    // the new type to which the operand should be casted by setting the value
    // of OpTy. If we change OpTy, also set shouldCast to true.
    switch (Opcode) {
    default:
        // for most instructions, it doesn't matter
        break;
    case llvm::Instruction::Add:
    case llvm::Instruction::Sub:
    case llvm::Instruction::Mul:
        // We need to cast integer arithmetic so that it is always performed
        // as unsigned, to avoid undefined behavior on overflow.
    case llvm::Instruction::LShr:
    case llvm::Instruction::UDiv:
    case llvm::Instruction::URem: // Cast to unsigned first
        shouldCast = true;
        castIsSigned = false;
        break;
    case llvm::Instruction::GetElementPtr:
    case llvm::Instruction::AShr:
    case llvm::Instruction::SDiv:
    case llvm::Instruction::SRem: // Cast to signed first
        shouldCast = true;
        castIsSigned = true;
        break;
    }

    // Write out the casted operand if we should, otherwise just write the
    // operand.
    if (shouldCast) {
        Out << "((";
        printSimpleType(Out, OpTy, castIsSigned);
        Out << ")";
        writeOperand(Operand);
        Out << ")";
    } else
        writeOperand(Operand);
}

// Write the operand with a cast to another type based on the icmp predicate
// being used.
void CWriter::writeOperandWithCast(llvm::Value *Operand, const llvm::ICmpInst &Cmp) {
    // This has to do a cast to ensure the operand has the right signedness.
    // Also, if the operand is a pointer, we make sure to cast to an integer when
    // doing the comparison both for signedness and so that the C compiler doesn't
    // optimize things like "p < NULL" to false (p may contain an integer value
    // f.e.).
    bool shouldCast = Cmp.isRelational();

    // Write out the casted operand if we should, otherwise just write the
    // operand.
    if (!shouldCast) {
        writeOperand(Operand);
        return;
    }

    // Should this be a signed comparison?  If so, convert to signed.
    bool castIsSigned = Cmp.isSigned();

    // If the operand was a pointer, convert to a large integer type.
    llvm::Type *OpTy = Operand->getType();
    if (OpTy->isPointerTy())
        OpTy = TD->getIntPtrType(Operand->getContext());

    Out << "((";
    printSimpleType(Out, OpTy, castIsSigned);
    Out << ")";
    writeOperand(Operand);
    Out << ")";
}

// generateCompilerSpecificCode - This is where we add conditional compilation
// directives to cater to specific compilers as need be.
//
static void generateCompilerSpecificCode(llvm::formatted_raw_ostream &Out, const llvm::DataLayout *TD) {
    // We output GCC specific attributes to preserve 'linkonce'ness on globals.
    // If we aren't being compiled with GCC, just drop these attributes.
    Out << "#ifndef __GNUC__  /* Can only support \"linkonce\" vars with GCC */\n"
        << "#define __attribute__(X)\n"
        << "#endif\n\n";

    // On Mac OS X, "external weak" is spelled "__attribute__((weak_import))".
    Out << "#if defined(__GNUC__) && defined(__APPLE_CC__)\n"
        << "#define __EXTERNAL_WEAK__ __attribute__((weak_import))\n"
        << "#elif defined(__GNUC__)\n"
        << "#define __EXTERNAL_WEAK__ __attribute__((weak))\n"
        << "#else\n"
        << "#define __EXTERNAL_WEAK__\n"
        << "#endif\n\n";

    // For now, turn off the weak linkage attribute on Mac OS X. (See above.)
    Out << "#if defined(__GNUC__) && defined(__APPLE_CC__)\n"
        << "#define __ATTRIBUTE_WEAK__\n"
        << "#elif defined(__GNUC__)\n"
        << "#define __ATTRIBUTE_WEAK__ __attribute__((weak))\n"
        << "#else\n"
        << "#define __ATTRIBUTE_WEAK__\n"
        << "#endif\n\n";

    // Add hidden visibility support. FIXME: APPLE_CC?
    Out << "#if defined(__GNUC__)\n"
        << "#define __HIDDEN__ __attribute__((visibility(\"hidden\")))\n"
        << "#endif\n\n";

    // Define NaN and Inf as GCC builtins if using GCC, as 0 otherwise
    // From the GCC documentation:
    //
    //   double __builtin_nan (const char *str)
    //
    // This is an implementation of the ISO C99 function nan.
    //
    // Since ISO C99 defines this function in terms of strtod, which we do
    // not implement, a description of the parsing is in order. The string is
    // parsed as by strtol; that is, the base is recognized by leading 0 or
    // 0x prefixes. The number parsed is placed in the significand such that
    // the least significant bit of the number is at the least significant
    // bit of the significand. The number is truncated to fit the significand
    // field provided. The significand is forced to be a quiet NaN.
    //
    // This function, if given a string literal, is evaluated early enough
    // that it is considered a compile-time constant.
    //
    //   float __builtin_nanf (const char *str)
    //
    // Similar to __builtin_nan, except the return type is float.
    //
    //   double __builtin_inf (void)
    //
    // Similar to __builtin_huge_val, except a warning is generated if the
    // target floating-point format does not support infinities. This
    // function is suitable for implementing the ISO C99 macro INFINITY.
    //
    //   float __builtin_inff (void)
    //
    // Similar to __builtin_inf, except the return type is float.
    Out << "#if (defined(__GNUC__) || defined(__clang__)) && !defined(__INTEL_COMPILER)\n"
        << "#define LLVM_NAN(NanStr)   __builtin_nan(NanStr)   /* Double */\n"
        << "#define LLVM_NANF(NanStr)  __builtin_nanf(NanStr)  /* Float */\n"
        << "#define LLVM_NANS(NanStr)  __builtin_nans(NanStr)  /* Double */\n"
        << "#define LLVM_NANSF(NanStr) __builtin_nansf(NanStr) /* Float */\n"
        << "#define LLVM_INF           __builtin_inf()         /* Double */\n"
        << "#define LLVM_INFF          __builtin_inff()        /* Float */\n"
        << "//#define LLVM_PREFETCH(addr,rw,locality) "
           "__builtin_prefetch(addr,rw,locality)\n"
        << "//#define __ATTRIBUTE_CTOR__ __attribute__((constructor))\n"
        << "//#define __ATTRIBUTE_DTOR__ __attribute__((destructor))\n"
        << "#elif defined(_MSC_VER) || defined(__INTEL_COMPILER)\n"
        << "#include <limits>\n"
        << "#define LLVM_NAN(NanStr)   std::numeric_limits<double>::quiet_NaN()\n"
        << "#define LLVM_NANF(NanStr)  std::numeric_limits<float>::quiet_NaN()\n"
        << "#define LLVM_NANS(NanStr)  std::numeric_limits<double>::signaling_NaN()\n"
        << "#define LLVM_NANSF(NanStr) std::numeric_limits<float>::signaling_NaN()\n"
        << "#define LLVM_INF           std::numeric_limits<double>::infinity()\n"
        << "#define LLVM_INFF          std::numeric_limits<float>::infinity()\n"
        << "//#define LLVM_PREFETCH(addr,rw,locality)            /* PREFETCH */\n"
        << "//#define __ATTRIBUTE_CTOR__\n"
        << "//#define __ATTRIBUTE_DTOR__\n"
        << "#else\n"
        << "#error \"Not MSVC, clang, or g++?\"\n"
        << "#endif\n\n";

    // LLVM_ASM() is used to define mapping of the symbol to a different name,
    // this is expected to be MacOS-only feature. So defining it only for
    // gcc and clang (Intel Compiler on Linux/MacOS is also ok).
    // For example, this feature is required to translate symbols described in
    // "Symbol Variants Release Notes" document (on Apple website).
    Out << "#if (defined(__GNUC__) || defined(__clang__))\n"
        << "#define LLVM_ASM(X) __asm(X)\n"
        << "#endif\n\n";

    Out << "#if defined(__clang__) || defined(__INTEL_COMPILER) || "
           "(__GNUC__ < 4) /* Old GCCs, or compilers not GCC */ \n"
        << "#define __builtin_stack_save() 0   /* not implemented */\n"
        << "#define __builtin_stack_restore(X) /* noop */\n"
        << "#endif\n\n";

#if 0
  // Output typedefs for 128-bit integers. If these are needed with a
  // 32-bit target or with a C compiler that doesn't support mode(TI),
  // more drastic measures will be needed.
  Out << "#if __GNUC__ && __LP64__ /* 128-bit integer types */\n"
      << "typedef int __attribute__((mode(TI))) llvmInt128;\n"
      << "typedef unsigned __attribute__((mode(TI))) llvmUInt128;\n"
      << "#endif\n\n";
#endif

    // Output target-specific code that should be inserted into main.
    Out << "#define CODE_FOR_MAIN() /* Any target-specific code for main()*/\n";
}

/// FindStaticTors - Given a static ctor/dtor list, unpack its contents into
/// the StaticTors set.
static void FindStaticTors(llvm::GlobalVariable *GV, std::set<llvm::Function *> &StaticTors) {
    llvm::ConstantArray *InitList = llvm::dyn_cast<llvm::ConstantArray>(GV->getInitializer());
    if (!InitList)
        return;

    for (unsigned i = 0, e = InitList->getNumOperands(); i != e; ++i)
        if (llvm::ConstantStruct *CS = llvm::dyn_cast<llvm::ConstantStruct>(InitList->getOperand(i))) {
            if (CS->getNumOperands() != 2)
                return; // Not array of 2-element structs.

            if (CS->getOperand(1)->isNullValue())
                return; // Found a null terminator, exit printing.
            llvm::Constant *FP = CS->getOperand(1);
            if (llvm::ConstantExpr *CE = llvm::dyn_cast<llvm::ConstantExpr>(FP))
                if (CE->isCast())
                    FP = CE->getOperand(0);
            if (llvm::Function *F = llvm::dyn_cast<llvm::Function>(FP))
                StaticTors.insert(F);
        }
}

enum SpecialGlobalClass { NotSpecial = 0, GlobalCtors, GlobalDtors, NotPrinted };

/// getGlobalVariableClass - If this is a global that is specially recognized
/// by LLVM, return a code that indicates how we should handle it.
static SpecialGlobalClass getGlobalVariableClass(const llvm::GlobalVariable *GV) {
    // If this is a global ctors/dtors list, handle it now.
    if (GV->hasAppendingLinkage() && GV->use_empty()) {
        if (GV->getName() == "llvm.global_ctors")
            return GlobalCtors;
        else if (GV->getName() == "llvm.global_dtors")
            return GlobalDtors;
    }

    // Otherwise, if it is other metadata, don't print it.  This catches things
    // like debug information.
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_5 && ISPC_LLVM_VERSION <= ISPC_LLVM_3_8 /* LLVM 3.5-3.8 */
    // Here we compare char *
    if (!strcmp(GV->getSection(), "llvm.metadata"))
#else
    // Here we compare strings
    if (GV->getSection() == "llvm.metadata")
#endif
        return NotPrinted;

    return NotSpecial;
}

// PrintEscapedString - Print each character of the specified string, escaping
// it if it is not printable or if it is an escape char.
static void PrintEscapedString(const char *Str, unsigned Length, llvm::raw_ostream &Out) {
    for (unsigned i = 0; i != Length; ++i) {
        unsigned char C = Str[i];
        if (isprint(C) && C != '\\' && C != '"')
            Out << C;
        else if (C == '\\')
            Out << "\\\\";
        else if (C == '\"')
            Out << "\\\"";
        else if (C == '\t')
            Out << "\\t";
        else
            Out << "\\x" << llvm::hexdigit(C >> 4) << llvm::hexdigit(C & 0x0F);
    }
}

// PrintEscapedString - Print each character of the specified string, escaping
// it if it is not printable or if it is an escape char.
static void PrintEscapedString(const std::string &Str, llvm::raw_ostream &Out) {
    PrintEscapedString(Str.c_str(), Str.size(), Out);
}

bool CWriter::doInitialization(llvm::Module &M) {
    llvm::FunctionPass::doInitialization(M);

    // Initialize
    TheModule = &M;

    TD = new llvm::DataLayout(&M);
    IL = new llvm::IntrinsicLowering(*TD);
    // AddPrototypes was removed from LLVM 9.0.
    // It looks like that usage of this method does not affect ISPC functionality
    // so it is safe to just remove it for LLVM 9.0+ versions.
#if ISPC_LLVM_VERSION <= ISPC_LLVM_8_0
    IL->AddPrototypes(M);
#endif

#if 0
  std::string Triple = TheModule->getTargetTriple();
  if (Triple.empty())
    Triple = llvm::sys::getDefaultTargetTriple();

  std::string E;
  if (const llvm::Target *Match = llvm::TargetRegistry::lookupTarget(Triple, E))
    TAsm = Match->createMCAsmInfo(Triple);
#endif
    TAsm = new CBEMCAsmInfo();
    MRI = new llvm::MCRegisterInfo();
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_4 // LLVM 3.4+
    TCtx = new llvm::MCContext(TAsm, MRI, NULL);
#else
    TCtx = new llvm::MCContext(*TAsm, *MRI, NULL);
#endif
    // Mang = new llvm::Mangler(*TCtx, *TD);

    // Keep track of which functions are static ctors/dtors so they can have
    // an attribute added to their prototypes.
    std::set<llvm::Function *> StaticCtors, StaticDtors;
    for (llvm::Module::global_iterator I = M.global_begin(), E = M.global_end(); I != E; ++I) {
#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_7 /* 3.2, 3.3, 3.4, 3.5, 3.6, 3.7 */
        switch (getGlobalVariableClass(I)) {
#else /* LLVM 3.8+ */
        switch (getGlobalVariableClass(&*I)) {
#endif
        default:
            break;
        case GlobalCtors:
            FindStaticTors(&*I, StaticCtors);
            break;
        case GlobalDtors:
            FindStaticTors(&*I, StaticDtors);
            break;
        }
    }

    Out << "/*******************************************************************\n";
    Out << "  This file has been automatically generated by ispc\n";
    Out << "  DO NOT EDIT THIS FILE DIRECTLY\n";
    Out << " *******************************************************************/\n\n";

    Out << "/* Provide Declarations */\n";
    Out << "#include <stdarg.h>\n"; // Varargs support
    Out << "#include <setjmp.h>\n"; // Unwind support
    Out << "#include <limits.h>\n"; // With overflow intrinsics support.
    Out << "#include <stdlib.h>\n";
    Out << "#ifdef _MSC_VER\n";
    Out << "  #define NOMINMAX\n";
    Out << "  #include <windows.h>\n";
    Out << "#endif // _MSC_VER\n";
    Out << "#include <stdlib.h>\n";
    Out << "#include <stdint.h>\n";
    Out << "/* get a declaration for alloca */\n";
    Out << "#ifdef _MSC_VER\n";
    Out << "  #include <malloc.h>\n";
    Out << "  #define alloca _alloca\n";
    Out << "#else\n";
    Out << "  #include <alloca.h>\n";
    Out << "#endif\n\n";

    if (g->opt.fastMath) {
        Out << "#define ISPC_FAST_MATH 1\n";
    } else {
        Out << "#undef ISPC_FAST_MATH\n";
    }

    if (g->opt.forceAlignedMemory) {
        Out << "#define ISPC_FORCE_ALIGNED_MEMORY\n";
    }

    Out << "#include \"" << includeName << "\"\n";

    Out << "\n/* Basic Library Function Declarations */\n";
    Out << "extern \"C\" {\n";
    Out << "int puts(unsigned char *);\n";
    Out << "unsigned int putchar(unsigned int);\n";
    Out << "int fflush(void *);\n";
    Out << "int printf(const unsigned char *, ...);\n";
    Out << "uint8_t *memcpy(uint8_t *, uint8_t *, uint64_t );\n";
    Out << "uint8_t *memset(uint8_t *, uint8_t, uint64_t );\n";
    Out << "void memset_pattern16(void *, const void *, uint64_t );\n";
    Out << "}\n\n";

    generateCompilerSpecificCode(Out, TD);

    // Provide a definition for `bool' if not compiling with a C++ compiler.
    Out << "\n"
        << "#ifndef __cplusplus\ntypedef unsigned char bool;\n#endif\n"

        << "\n\n/* Support for floating point constants */\n"
        << "typedef uint64_t ConstantDoubleTy;\n"
        << "typedef uint32_t ConstantFloatTy;\n"
        << "typedef struct { unsigned long long f1; unsigned short f2; "
           "unsigned short pad[3]; } ConstantFP80Ty;\n"
        // This is used for both kinds of 128-bit long double; meaning differs.
        << "typedef struct { uint64_t f1, f2; } ConstantFP128Ty;\n"
        << "\n\n/* Global Declarations */\n\n";

    // First output all the declarations for the program, because C requires
    // Functions & globals to be declared before they are used.
    //
    if (!M.getModuleInlineAsm().empty()) {
        Out << "/* Module asm statements */\n"
            << "asm(";

        // Split the string into lines, to make it easier to read the .ll file.
        std::string Asm = M.getModuleInlineAsm();
        size_t CurPos = 0;
        size_t NewLine = Asm.find_first_of('\n', CurPos);
        while (NewLine != std::string::npos) {
            // We found a newline, print the portion of the asm string from the
            // last newline up to this newline.
            Out << "\"";
            PrintEscapedString(std::string(Asm.begin() + CurPos, Asm.begin() + NewLine), Out);
            Out << "\\n\"\n";
            CurPos = NewLine + 1;
            NewLine = Asm.find_first_of('\n', CurPos);
        }
        Out << "\"";
        PrintEscapedString(std::string(Asm.begin() + CurPos, Asm.end()), Out);
        Out << "\");\n"
            << "/* End Module asm statements */\n";
    }

    // Loop over the symbol table, emitting all named constants.
    printModuleTypes();

    // Global variable declarations...
    if (!M.global_empty()) {
        Out << "\n/* External Global Variable Declarations */\n";
        for (llvm::Module::global_iterator I = M.global_begin(), E = M.global_end(); I != E; ++I) {

            if (I->hasExternalLinkage() || I->hasExternalWeakLinkage() || I->hasCommonLinkage())
                Out << "extern ";
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_5 // LLVM 3.5+
            else if (I->hasDLLImportStorageClass())
#else
            else if (I->hasDLLImportLinkage())
#endif
                Out << "__declspec(dllimport) ";
            else
                continue; // Internal Global

            // Thread Local Storage
            if (I->isThreadLocal())
                Out << "__thread ";

            printType(Out, I->getType()->getElementType(), false, GetValueName(&*I));

            if (I->hasExternalWeakLinkage())
                Out << " __EXTERNAL_WEAK__";
            Out << ";\n";
        }
    }

    // Output the global variable declarations
    if (!M.global_empty()) {
        Out << "\n\n/* Global Variable Declarations */\n";
        for (llvm::Module::global_iterator I = M.global_begin(), E = M.global_end(); I != E; ++I)
            if (!I->isDeclaration()) {
                // Ignore special globals, such as debug info.
                if (getGlobalVariableClass(&*I))
                    continue;

                if (I->hasLocalLinkage())
                    continue;
                else
                    Out << "extern ";

                // Thread Local Storage
                if (I->isThreadLocal())
                    Out << "__thread ";

                printType(Out, I->getType()->getElementType(), false, GetValueName(&*I));

                if (I->hasLinkOnceLinkage())
                    Out << " __attribute__((common))";
                else if (I->hasCommonLinkage()) // FIXME is this right?
                    Out << " __ATTRIBUTE_WEAK__";
                else if (I->hasWeakLinkage())
                    Out << " __ATTRIBUTE_WEAK__";
                else if (I->hasExternalWeakLinkage())
                    Out << " __EXTERNAL_WEAK__";
                if (I->hasHiddenVisibility())
                    Out << " __HIDDEN__";
                Out << ";\n";
            }
    }

    // Function declarations
    Out << "\n/* Function Declarations */\n";
    Out << "extern \"C\" {\n";

    // Store the intrinsics which will be declared/defined below.
    llvm::SmallVector<const llvm::Function *, 8> intrinsicsToDefine;

    for (llvm::Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
        // Don't print declarations for intrinsic functions.
        // Store the used intrinsics, which need to be explicitly defined.
        if (I->isIntrinsic()) {
            switch (I->getIntrinsicID()) {
            default:
                break;
            case llvm::Intrinsic::uadd_with_overflow:
            case llvm::Intrinsic::sadd_with_overflow:
            case llvm::Intrinsic::umul_with_overflow:
#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_7 /* 3.2, 3.3, 3.4, 3.5, 3.6, 3.7 */
                intrinsicsToDefine.push_back(I);
#else /* LLVM 3.8+ */
                intrinsicsToDefine.push_back(&*I);
#endif
                break;
            }
            continue;
        }

        if (I->getName() == "setjmp" || I->getName() == "abort" || I->getName() == "longjmp" ||
            I->getName() == "_setjmp" || I->getName() == "memset" || I->getName() == "memset_pattern16" ||
            I->getName() == "puts" || I->getName() == "printf" || I->getName() == "putchar" ||
            I->getName() == "fflush" ||
            // Memory allocation
            I->getName() == "malloc" || I->getName() == "posix_memalign" || I->getName() == "free" ||
            I->getName() == "_aligned_malloc" || I->getName() == "_aligned_free")
            continue;

        // Don't redeclare ispc's own intrinsics
        std::string name = I->getName();
        if (name.size() > 2 && name[0] == '_' && name[1] == '_')
            continue;

        if (I->hasExternalWeakLinkage())
            Out << "extern ";
#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_7 /* 3.2, 3.3, 3.4, 3.5, 3.6, 3.7 */
        printFunctionSignature(I, true);
#else /* LLVM 3.8+ */
        printFunctionSignature(&*I, true);
#endif
        if (I->hasWeakLinkage() || I->hasLinkOnceLinkage())
            Out << " __ATTRIBUTE_WEAK__";
        if (I->hasExternalWeakLinkage())
            Out << " __EXTERNAL_WEAK__";
#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_7 /* 3.2, 3.3, 3.4, 3.5, 3.6, 3.7 */
        if (StaticCtors.count(I))
            Out << " __ATTRIBUTE_CTOR__";
        if (StaticDtors.count(I))
#else /* LLVM 3.8+ */
        if (StaticCtors.count(&*I))
            Out << " __ATTRIBUTE_CTOR__";
        if (StaticDtors.count(&*I))
#endif
            Out << " __ATTRIBUTE_DTOR__";
        if (I->hasHiddenVisibility())
            Out << " __HIDDEN__";

        // This is MacOS specific feature, this should not appear on other platforms.
        if (I->hasName() && I->getName()[0] == 1)
            Out << " LLVM_ASM(\"" << I->getName().substr(1) << "\")";

        Out << ";\n";
    }
    Out << "}\n\n";

    if (!M.empty())
        Out << "\n\n/* Function Bodies */\n";

    // Emit some helper functions for dealing with FCMP instruction's
    // predicates
    Out << "template <typename A, typename B> static inline int llvm_fcmp_ord(A X, B Y) { ";
    Out << "return X == X && Y == Y; }\n";
    Out << "template <typename A, typename B> static inline int llvm_fcmp_uno(A X, B Y) { ";
    Out << "return X != X || Y != Y; }\n";
    Out << "template <typename A, typename B> static inline int llvm_fcmp_ueq(A X, B Y) { ";
    Out << "return X == Y || llvm_fcmp_uno(X, Y); }\n";
    Out << "template <typename A, typename B> static inline int llvm_fcmp_une(A X, B Y) { ";
    Out << "return X != Y; }\n";
    Out << "template <typename A, typename B> static inline int llvm_fcmp_ult(A X, B Y) { ";
    Out << "return X <  Y || llvm_fcmp_uno(X, Y); }\n";
    Out << "template <typename A, typename B> static inline int llvm_fcmp_ugt(A X, B Y) { ";
    Out << "return X >  Y || llvm_fcmp_uno(X, Y); }\n";
    Out << "template <typename A, typename B> static inline int llvm_fcmp_ule(A X, B Y) { ";
    Out << "return X <= Y || llvm_fcmp_uno(X, Y); }\n";
    Out << "template <typename A, typename B> static inline int llvm_fcmp_uge(A X, B Y) { ";
    Out << "return X >= Y || llvm_fcmp_uno(X, Y); }\n";
    Out << "template <typename A, typename B> static inline int llvm_fcmp_oeq(A X, B Y) { ";
    Out << "return X == Y ; }\n";
    Out << "template <typename A, typename B> static inline int llvm_fcmp_one(A X, B Y) { ";
    Out << "return X != Y && llvm_fcmp_ord(X, Y); }\n";
    Out << "template <typename A, typename B> static inline int llvm_fcmp_olt(A X, B Y) { ";
    Out << "return X <  Y ; }\n";
    Out << "template <typename A, typename B> static inline int llvm_fcmp_ogt(A X, B Y) { ";
    Out << "return X >  Y ; }\n";
    Out << "template <typename A, typename B> static inline int llvm_fcmp_ole(A X, B Y) { ";
    Out << "return X <= Y ; }\n";
    Out << "template <typename A, typename B> static inline int llvm_fcmp_oge(A X, B Y) { ";
    Out << "return X >= Y ; }\n";
    Out << "template <typename A> A *Memset(A *ptr, int count, size_t len) { ";
    Out << "return (A *)memset(ptr, count, len); }\n";

    // Emit definitions of the intrinsics.
    for (llvm::SmallVector<const llvm::Function *, 8>::const_iterator I = intrinsicsToDefine.begin(),
                                                                      E = intrinsicsToDefine.end();
         I != E; ++I) {
        printIntrinsicDefinition(**I, Out);
    }

    // Output the global variable definitions and contents...
    if (!M.global_empty()) {
        Out << "\n\n/* Global Variable Definitions and Initialization */\n";
        for (llvm::Module::global_iterator I = M.global_begin(), E = M.global_end(); I != E; ++I)
            if (!I->isDeclaration()) {
                // Ignore special globals, such as debug info.
#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_7 /* 3.2, 3.3, 3.4, 3.5, 3.6, 3.7 */
                if (getGlobalVariableClass(I))
#else /* LLVM 3.8+ */
                if (getGlobalVariableClass(&*I))
#endif
                    continue;

                if (I->hasLocalLinkage())
                    Out << "static ";
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_5 // LLVM 3.5+
                else if (I->hasDLLImportStorageClass())
                    Out << "__declspec(dllimport) ";
                else if (I->hasDLLExportStorageClass())
                    Out << "__declspec(dllexport) ";
#else
                else if (I->hasDLLImportLinkage())
                    Out << "__declspec(dllimport) ";
                else if (I->hasDLLExportLinkage())
                    Out << "__declspec(dllexport) ";
#endif
                // Thread Local Storage
                if (I->isThreadLocal())
                    Out << "__thread ";

                printType(Out, I->getType()->getElementType(), false,
#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_7 /* 3.2, 3.3, 3.4, 3.5, 3.6, 3.7 */
                          GetValueName(I));
#else /* LLVM 3.8+ */
                          GetValueName(&*I));
#endif

                if (I->hasLinkOnceLinkage())
                    Out << " __attribute__((common))";
                else if (I->hasWeakLinkage())
                    Out << " __ATTRIBUTE_WEAK__";
                else if (I->hasCommonLinkage())
                    Out << " __ATTRIBUTE_WEAK__";

                if (I->hasHiddenVisibility())
                    Out << " __HIDDEN__";

                // If the initializer is not null, emit the initializer.  If it is null,
                // we try to avoid emitting large amounts of zeros.  The problem with
                // this, however, occurs when the variable has weak linkage.  In this
                // case, the assembler will complain about the variable being both weak
                // and common, so we disable this optimization.
                // FIXME common linkage should avoid this problem.
                if (!I->getInitializer()->isNullValue()) {
                    Out << " = ";

                    // vec16_i64 should be handled separately
                    if (is_vec16_i64_ty(I->getType()->getElementType())) {
                        Out << "/* vec16_i64 should be loaded carefully on knc */\n";
                        Out << "\n#if defined(KNC) \n";
                        Out << "hilo2zmm";
                        Out << "\n#endif \n";
                    }

                    Out << "(";
                    writeOperand(I->getInitializer(), false);
                    Out << ")";
                } else if (I->hasWeakLinkage()) {
                    // We have to specify an initializer, but it doesn't have to be
                    // complete.  If the value is an aggregate, print out { 0 }, and let
                    // the compiler figure out the rest of the zeros.
                    Out << " = ";
                    if (I->getInitializer()->getType()->isStructTy() || I->getInitializer()->getType()->isVectorTy()) {
                        Out << "{ 0 }";
                    } else if (I->getInitializer()->getType()->isArrayTy()) {
                        // As with structs and vectors, but with an extra set of braces
                        // because arrays are wrapped in structs.
                        Out << "{ { 0 } }";
                    } else {
                        // Just print it out normally.
                        writeOperand(I->getInitializer(), false);
                    }
                }
                Out << ";\n";
            }
    }

    return false;
}

/// Output all floating point constants that cannot be printed accurately...
void CWriter::printFloatingPointConstants(llvm::Function &F) {
    // Scan the module for floating point constants.  If any FP constant is used
    // in the function, we want to redirect it here so that we do not depend on
    // the precision of the printed form, unless the printed form preserves
    // precision.
    //
    for (constant_scanner::constant_iterator I = constant_scanner::constant_begin(&F),
                                             E = constant_scanner::constant_end(&F);
         I != E; ++I)
        printFloatingPointConstants(*I);

    Out << '\n';
}

void CWriter::printFloatingPointConstants(const llvm::Constant *C) {
    // If this is a constant expression, recursively check for constant fp values.
    if (const llvm::ConstantExpr *CE = llvm::dyn_cast<llvm::ConstantExpr>(C)) {
        for (unsigned i = 0, e = CE->getNumOperands(); i != e; ++i)
            printFloatingPointConstants(CE->getOperand(i));
        return;
    }

    // Otherwise, check for a FP constant that we need to print.
    const llvm::ConstantFP *FPC = llvm::dyn_cast<llvm::ConstantFP>(C);
    if (FPC == 0 ||
        // Do not put in FPConstantMap if safe.
        isFPCSafeToPrint(FPC) ||
        // Already printed this constant?
        FPConstantMap.count(FPC))
        return;

    FPConstantMap[FPC] = FPCounter; // Number the FP constants

    if (FPC->getType() == llvm::Type::getDoubleTy(FPC->getContext())) {
        double Val = FPC->getValueAPF().convertToDouble();
        uint64_t i = FPC->getValueAPF().bitcastToAPInt().getZExtValue();
        Out << "static const ConstantDoubleTy FPConstant" << FPCounter++ << " = 0x" << llvm::utohexstr(i)
            << "ULL;    /* " << Val << " */\n";
    } else if (FPC->getType() == llvm::Type::getFloatTy(FPC->getContext())) {
        float Val = FPC->getValueAPF().convertToFloat();
        uint32_t i = (uint32_t)FPC->getValueAPF().bitcastToAPInt().getZExtValue();
        Out << "static const ConstantFloatTy FPConstant" << FPCounter++ << " = 0x" << llvm::utohexstr(i) << "U;    /* "
            << Val << " */\n";
    } else if (FPC->getType() == llvm::Type::getX86_FP80Ty(FPC->getContext())) {
        // api needed to prevent premature destruction
        llvm::APInt api = FPC->getValueAPF().bitcastToAPInt();
        const uint64_t *p = api.getRawData();
        Out << "static const ConstantFP80Ty FPConstant" << FPCounter++ << " = { 0x" << llvm::utohexstr(p[0])
            << "ULL, 0x" << llvm::utohexstr((uint16_t)p[1]) << ",{0,0,0}"
            << "}; /* Long double constant */\n";
    } else if (FPC->getType() == llvm::Type::getPPC_FP128Ty(FPC->getContext()) ||
               FPC->getType() == llvm::Type::getFP128Ty(FPC->getContext())) {
        llvm::APInt api = FPC->getValueAPF().bitcastToAPInt();
        const uint64_t *p = api.getRawData();
        Out << "static const ConstantFP128Ty FPConstant" << FPCounter++ << " = { 0x" << llvm::utohexstr(p[0]) << ", 0x"
            << llvm::utohexstr(p[1]) << "}; /* Long double constant */\n";

    } else {
        llvm_unreachable("Unknown float type!");
    }
}

// For any vector constants, generate code to declare static const arrays
// with their element values.  Doing so allows us to emit aligned vector
// loads to get their values, rather than tediously inserting the
// individual values into the vector.
void CWriter::printVectorConstants(llvm::Function &F) {
    for (constant_scanner::constant_iterator I = constant_scanner::constant_begin(&F),
                                             E = constant_scanner::constant_end(&F);
         I != E; ++I) {
        const llvm::ConstantDataVector *CDV = llvm::dyn_cast<llvm::ConstantDataVector>(*I);
        if (CDV == NULL)
            continue;

        // Don't bother if this is a splat of the same value; a (more
        // efficient?) __splat_* call will be generated for these.
        if (CDV->getSplatValue() != NULL)
            continue;

        // Don't align to anything more than 64 bytes
        int alignment = 4 * std::min(vectorWidth, 16);

        Out << "static const ";
        printSimpleType(Out, CDV->getElementType(), true, "");
        Out << "__attribute__ ((aligned(" << alignment << "))) ";
        Out << "VectorConstant" << VectorConstantIndex << "[] = { ";
        for (int i = 0; i < (int)CDV->getNumElements(); ++i) {
            printConstant(CDV->getElementAsConstant(i), false);
            Out << ", ";
        }
        Out << " };\n";

        VectorConstantMap[CDV] = VectorConstantIndex++;
    }
    Out << "\n";
}

/// printSymbolTable - Run through symbol table looking for type names.  If a
/// type name is found, emit its declaration...
///
void CWriter::printModuleTypes() {
    Out << "\n/* Helper union for bitcasts */\n";
    Out << "typedef union {\n";
    Out << "  unsigned int Int32;\n";
    Out << "  unsigned long long Int64;\n";
    Out << "  float Float;\n";
    Out << "  double Double;\n";
    Out << "} llvmBitCastUnion;\n";
    Out << "\n/* This is special class, designed for operations with long int.*/                       \n";
    Out << "namespace {                                                                                \n";
    Out << "template <int num_bits>                                                                    \n";
    Out << "struct iN {                                                                                \n";
    Out << "  int num[num_bits / (sizeof (int) * 8)];                                                  \n";
    Out << "                                                                                           \n";
    Out << "  iN () {}                                                                                 \n";
    Out << "                                                                                           \n";
    Out << "  iN (const char *val) {                                                                   \n";
    Out << "    if (val == NULL)                                                                       \n";
    Out << "      return;                                                                              \n";
    Out << "    int length = num_bits / (sizeof (int) * 8);                                            \n";
    Out << "    int val_len = 0;                                                                       \n";
    Out << "    for (val_len = 0; val[val_len]; (val_len)++);                                          \n";
    Out << "    for (int i = 0; (i < val_len && i < num_bits); i++)                                    \n";
    Out << "      num[i / (sizeof (int) * 8)] = (num[i / (sizeof (int) * 8)] << 1) | (val[i] - '0');   \n";
    Out << "  }                                                                                        \n";
    Out << "                                                                                           \n";
    Out << "  ~iN () {}                                                                                \n";
    Out << "                                                                                           \n";
    Out << "  iN operator >> (const iN rhs) {                                                          \n";
    Out << "    iN res;                                                                                \n";
    Out << "    int length = num_bits / (sizeof (int) * 8);                                            \n";
    Out << "    int cells_shift = rhs.num[0] / (sizeof(int) * 8);                                      \n";
    Out << "    int small_shift = rhs.num[0] % (sizeof(int) * 8);                                      \n";
    Out << "    for (int i = 0; i < (length - cells_shift); i++)                                       \n";
    Out << "      res.num[i] = this->num[cells_shift + i];                                             \n";
    Out << "    for (int i = 0; i < length - 1; i++) {                                                 \n";
    Out << "      res.num[i] = this->num[i] >> small_shift;                                            \n";
    Out << "      res.num[i]  = ((this->num[i + 1] << ((sizeof(int) * 8) - small_shift))) | res.num[i];\n";
    Out << "    }                                                                                      \n";
    Out << "    res.num[length - 1] = res.num[length - 1] >> small_shift;                              \n";
    Out << "    return res;                                                                            \n";
    Out << "  }                                                                                        \n";
    Out << "                                                                                           \n";
    Out << "  iN operator & (iN rhs) {                                                                 \n";
    Out << "    iN res;                                                                                \n";
    Out << "    int length = num_bits / (sizeof (int) * 8);                                            \n";
    Out << "    for (int i = 0; i < length; i++)                                                       \n";
    Out << "      res.num[i] = (this->num[i]) & (rhs.num[i]);                                          \n";
    Out << "    return res;                                                                            \n";
    Out << "  }                                                                                        \n";
    Out << "                                                                                           \n";
    Out << "  operator uint32_t() { return this->num[0]; }                                             \n";
    Out << "                                                                                           \n";
    Out << "  template <class T>                                                                       \n";
    Out << "  friend iN<num_bits> __cast_bits(iN<num_bits> to, T from) {                               \n";
    Out << "    for (int i = 0; i <" << vectorWidth << "; i++)                                         \n";
    Out << "      to.num[i] = ((int*)(&from))[i];                                                      \n";
    Out << "    return to;                                                                             \n";
    Out << "  }                                                                                        \n";
    Out << "                                                                                           \n";
    Out << "  template <class T>                                                                       \n";
    Out << "  friend T __cast_bits(T to, iN<num_bits> from) {                                          \n";
    Out << "    for (int i = 0; i <" << vectorWidth << "; i++)                                         \n";
    Out << "      ((int*)(&to))[i] = from.num[i];                                                      \n";
    Out << "    return to;                                                                             \n";
    Out << "  }                                                                                        \n";
    Out << "                                                                                           \n";
    Out << "  template <int ALIGN, class T>                                                            \n";
    Out << "  friend void __store(T *p, iN<num_bits> val) {                                            \n";
    Out << "    for (int i = 0; i <" << vectorWidth << "; i++)                                         \n";
    Out << "      ((int*)p)[i] = val.num[i];                                                           \n";
    Out << "  }                                                                                        \n";
    Out << "};                                                                                         \n";
    Out << "};\n";
    Out << "\n";

    // Get all of the struct types used in the module.
    std::vector<llvm::StructType *> StructTypes;
    llvm::TypeFinder typeFinder;
    typeFinder.run(*TheModule, false);
    for (llvm::TypeFinder::iterator iter = typeFinder.begin(); iter != typeFinder.end(); ++iter)
        StructTypes.push_back(*iter);

    // Get all of the array types used in the module
    std::vector<llvm::ArrayType *> ArrayTypes;
    std::vector<llvm::IntegerType *> IntegerTypes;
    std::vector<bool> IsVolatile;
    std::vector<int> Alignment;

    findUsedArrayAndLongIntTypes(TheModule, ArrayTypes, IntegerTypes, IsVolatile, Alignment);

    if (StructTypes.empty() && ArrayTypes.empty())
        return;

    Out << "/* Structure and array forward declarations */\n";

    unsigned NextTypeID = 0;

    // If any of them are missing names, add a unique ID to UnnamedStructIDs.
    // Print out forward declarations for structure types.
    for (unsigned i = 0, e = StructTypes.size(); i != e; ++i) {
        llvm::StructType *ST = StructTypes[i];

        if (ST->isLiteral() || ST->getName().empty())
            UnnamedStructIDs[ST] = NextTypeID++;

        std::string Name = getStructName(ST);

        Out << "struct " << Name << ";\n";
    }

    Out << "namespace {\n";
    for (unsigned i = 0, e = ArrayTypes.size(); i != e; ++i) {
        llvm::ArrayType *AT = ArrayTypes[i];
        ArrayIDs[AT] = NextTypeID++;
        std::string Name = getArrayName(AT);
        Out << "  struct " << Name << ";\n";
    }
    Out << "};\n";

    for (unsigned i = 0, e = IntegerTypes.size(); i != e; ++i) {
        llvm::IntegerType *IT = IntegerTypes[i];
        if (IT->getIntegerBitWidth() <= 64 || Alignment[i] == 0)
            continue;

        Out << "typedef struct __attribute__ ((packed, aligned(" << Alignment[i] << "))) {\n  ";
        IsVolatile[i] ? Out << "  volatile " : Out << "  ";
        printType(Out, IT, false, "data");
        Out << ";\n";
        Out << "} iN_" << IT->getIntegerBitWidth() << "_align_" << Alignment[i] << ";\n";
    }

    Out << '\n';

    // Keep track of which types have been printed so far.
    llvm::SmallPtrSet<llvm::Type *, 16> StructArrayPrinted;

    // Loop over all structures then push them into the stack so they are
    // printed in the correct order.
    //
    Out << "/* Structure and array contents */\n";
    for (unsigned i = 0, e = StructTypes.size(); i != e; ++i) {
        if (StructTypes[i]->isStructTy())
            // Only print out used types!
            printContainedStructs(StructTypes[i], StructArrayPrinted);
    }

    Out << "namespace {\n";
    for (unsigned i = 0, e = ArrayTypes.size(); i != e; ++i)
        printContainedArrays(ArrayTypes[i], StructArrayPrinted);

    Out << "};\n";
    Out << '\n';
}

// Push the struct onto the stack and recursively push all structs
// this one depends on.
//
// TODO:  Make this work properly with vector types
//
void CWriter::printContainedStructs(llvm::Type *Ty, llvm::SmallPtrSet<llvm::Type *, 16> &Printed) {
    // Don't walk through pointers.
    if (!(Ty->isStructTy() || Ty->isArrayTy()))
        return;

    // Print all contained types first.
    for (llvm::Type::subtype_iterator I = Ty->subtype_begin(), E = Ty->subtype_end(); I != E; ++I)
        printContainedStructs(*I, Printed);

    if (llvm::StructType *ST = llvm::dyn_cast<llvm::StructType>(Ty)) {
        // Check to see if we have already printed this struct.
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_6 // LLVM 3.6+
        if (!Printed.insert(Ty).second)
            return;
#else
        if (!Printed.insert(Ty))
            return;
#endif

        // Print structure type out.
        printType(Out, ST, false, getStructName(ST), true);
        Out << ";\n\n";
    }
    if (llvm::ArrayType *AT = llvm::dyn_cast<llvm::ArrayType>(Ty)) {
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_6 // LLVM 3.6+
        if (!Printed.insert(Ty).second)
            return;
#else
        if (!Printed.insert(Ty))
            return;
#endif

        Out << "namespace {\n";
        printType(Out, AT, false, getArrayName(AT), true);
        Out << ";\n}\n\n";
    }
}

void CWriter::printContainedArrays(llvm::ArrayType *ATy, llvm::SmallPtrSet<llvm::Type *, 16> &Printed) {
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_6 // LLVM 3.6+
    if (!Printed.insert(ATy).second)
        return;
#else
    if (!Printed.insert(ATy))
        return;
#endif

    llvm::ArrayType *ChildTy = llvm::dyn_cast<llvm::ArrayType>(ATy->getElementType());
    if (ChildTy != NULL)
        printContainedArrays(ChildTy, Printed);

    printType(Out, ATy, false, getArrayName(ATy), true);
    Out << ";\n\n";
}

void CWriter::printFunctionSignature(const llvm::Function *F, bool Prototype) {
    /// isStructReturn - Should this function actually return a struct by-value?
    bool isStructReturn = F->hasStructRetAttr();

    if (F->hasLocalLinkage())
        Out << "static ";
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_5 // LLVM 3.5+
    if (F->hasDLLImportStorageClass())
        Out << "__declspec(dllimport) ";
    if (F->hasDLLExportStorageClass())
        Out << "__declspec(dllexport) ";
#else
    if (F->hasDLLImportLinkage())
        Out << "__declspec(dllimport) ";
    if (F->hasDLLExportLinkage())
        Out << "__declspec(dllexport) ";
#endif
    switch (F->getCallingConv()) {
    case llvm::CallingConv::X86_StdCall:
        Out << "__attribute__((stdcall)) ";
        break;
    case llvm::CallingConv::X86_FastCall:
        Out << "__attribute__((fastcall)) ";
        break;
    case llvm::CallingConv::X86_ThisCall:
        Out << "__attribute__((thiscall)) ";
        break;
    default:
        break;
    }

    // Loop over the arguments, printing them...
    llvm::FunctionType *FT = llvm::cast<llvm::FunctionType>(F->getFunctionType());
#if ISPC_LLVM_VERSION == ISPC_LLVM_3_2
    const llvm::AttrListPtr &PAL = F->getAttributes();
#elif ISPC_LLVM_VERSION <= ISPC_LLVM_4_0
    const llvm::AttributeSet &PAL = F->getAttributes();
#else // LLVM 5.0+
    const llvm::AttributeList &PAL = F->getAttributes();
#endif

    std::string tstr;
    llvm::raw_string_ostream FunctionInnards(tstr);

    // Print out the name...
    FunctionInnards << GetValueName(F) << '(';

    bool PrintedArg = false;
    if (!F->isDeclaration()) {
        if (!F->arg_empty()) {
            llvm::Function::const_arg_iterator I = F->arg_begin(), E = F->arg_end();
            unsigned Idx = 1;

            // If this is a struct-return function, don't print the hidden
            // struct-return argument.
            if (isStructReturn) {
                assert(I != E && "Invalid struct return function!");
                ++I;
                ++Idx;
            }

            std::string ArgName;
            for (; I != E; ++I) {
                if (PrintedArg)
                    FunctionInnards << ", ";
                if (I->hasName() || !Prototype)
#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_7 /* 3.2, 3.3, 3.4, 3.5, 3.6, 3.7 */
                    ArgName = GetValueName(I);
#else /* LLVM 3.8+ */
                    ArgName = GetValueName(&*I);
#endif
                else
                    ArgName = "";
                llvm::Type *ArgTy = I->getType();
#if ISPC_LLVM_VERSION == ISPC_LLVM_3_2
                if (PAL.getParamAttributes(Idx).hasAttribute(llvm::Attributes::ByVal)) {
#elif ISPC_LLVM_VERSION <= ISPC_LLVM_4_0
                if (PAL.getParamAttributes(Idx).hasAttribute(llvm::AttributeSet::FunctionIndex,
                                                             llvm::Attribute::ByVal)) {
#else // LLVM 5.0+
                if (PAL.getParamAttributes(Idx).hasAttribute(llvm::Attribute::ByVal)) {
#endif
                    ArgTy = llvm::cast<llvm::PointerType>(ArgTy)->getElementType();
#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_7 /* 3.2, 3.3, 3.4, 3.5, 3.6, 3.7 */
                    ByValParams.insert(I);
#else /* LLVM 3.8+ */
                    ByValParams.insert(&*I);
#endif
                }
                printType(FunctionInnards, ArgTy,
#if ISPC_LLVM_VERSION == ISPC_LLVM_3_2
                          PAL.getParamAttributes(Idx).hasAttribute(llvm::Attributes::SExt),
#elif ISPC_LLVM_VERSION <= ISPC_LLVM_4_0
                          PAL.getParamAttributes(Idx).hasAttribute(llvm::AttributeSet::FunctionIndex,
                                                                   llvm::Attribute::SExt),
#else // LLVM 5.0+
                          PAL.getParamAttributes(Idx).hasAttribute(llvm::Attribute::SExt),
#endif
                          ArgName);
                PrintedArg = true;
                ++Idx;
            }
        }
    } else {
        // Loop over the arguments, printing them.
        llvm::FunctionType::param_iterator I = FT->param_begin(), E = FT->param_end();
        unsigned Idx = 1;

        // If this is a struct-return function, don't print the hidden
        // struct-return argument.
        if (isStructReturn) {
            assert(I != E && "Invalid struct return function!");
            ++I;
            ++Idx;
        }

        for (; I != E; ++I) {
            if (PrintedArg)
                FunctionInnards << ", ";
            llvm::Type *ArgTy = *I;
#if ISPC_LLVM_VERSION == ISPC_LLVM_3_2
            if (PAL.getParamAttributes(Idx).hasAttribute(llvm::Attributes::ByVal)) {
#elif ISPC_LLVM_VERSION <= ISPC_LLVM_4_0
            if (PAL.getParamAttributes(Idx).hasAttribute(llvm::AttributeSet::FunctionIndex, llvm::Attribute::ByVal)) {
#else // LLVM 5.0+
            if (PAL.getParamAttributes(Idx).hasAttribute(llvm::Attribute::ByVal)) {
#endif
                assert(ArgTy->isPointerTy());
                ArgTy = llvm::cast<llvm::PointerType>(ArgTy)->getElementType();
            }
            printType(FunctionInnards, ArgTy,
#if ISPC_LLVM_VERSION == ISPC_LLVM_3_2
                      PAL.getParamAttributes(Idx).hasAttribute(llvm::Attributes::SExt)
#elif ISPC_LLVM_VERSION <= ISPC_LLVM_4_0
                      PAL.getParamAttributes(Idx).hasAttribute(llvm::AttributeSet::FunctionIndex, llvm::Attribute::SExt)
#else // LLVM 5.0+
                      PAL.getParamAttributes(Idx).hasAttribute(llvm::Attribute::SExt)
#endif
            );
            PrintedArg = true;
            ++Idx;
        }
    }

    if (!PrintedArg && FT->isVarArg()) {
        FunctionInnards << "int vararg_dummy_arg";
        PrintedArg = true;
    }

    // Finish printing arguments... if this is a vararg function, print the ...,
    // unless there are no known types, in which case, we just emit ().
    //
    if (FT->isVarArg() && PrintedArg) {
        FunctionInnards << ",..."; // Output varargs portion of signature!
    } else if (!FT->isVarArg() && !PrintedArg) {
        FunctionInnards << "void"; // ret() -> ret(void) in C.
    }
    FunctionInnards << ')';

    // Get the return tpe for the function.
    llvm::Type *RetTy;
    if (!isStructReturn)
        RetTy = F->getReturnType();
    else {
        // If this is a struct-return function, print the struct-return type.
        RetTy = llvm::cast<llvm::PointerType>(FT->getParamType(0))->getElementType();
    }

    // Print out the return type and the signature built above.
    printType(Out, RetTy,
#if ISPC_LLVM_VERSION == ISPC_LLVM_3_2
              PAL.getParamAttributes(0).hasAttribute(llvm::Attributes::SExt),
#elif ISPC_LLVM_VERSION <= ISPC_LLVM_4_0
              PAL.getParamAttributes(0).hasAttribute(llvm::AttributeSet::ReturnIndex, llvm::Attribute::SExt),
#else // LLVM 5.0+
              PAL.getParamAttributes(0).hasAttribute(llvm::Attribute::SExt),
#endif
              FunctionInnards.str());
}

static inline bool isFPIntBitCast(const llvm::Instruction &I) {
    if (!llvm::isa<llvm::BitCastInst>(I))
        return false;
    llvm::Type *SrcTy = I.getOperand(0)->getType();
    llvm::Type *DstTy = I.getType();
    return (SrcTy->isFloatingPointTy() && DstTy->isIntegerTy()) || (DstTy->isFloatingPointTy() && SrcTy->isIntegerTy());
}

void CWriter::printFunction(llvm::Function &F) {
    /// isStructReturn - Should this function actually return a struct by-value?
    bool isStructReturn = F.hasStructRetAttr();

    printFunctionSignature(&F, false);
    Out << " {\n";

    // If this is a struct return function, handle the result with magic.
    if (isStructReturn) {
        llvm::Type *StructTy = llvm::cast<llvm::PointerType>(F.arg_begin()->getType())->getElementType();
        Out << "  ";
        printType(Out, StructTy, false, "StructReturn");
        Out << ";  /* Struct return temporary */\n";

        Out << "  ";
        printType(Out, F.arg_begin()->getType(), false, GetValueName(&*(F.arg_begin())));
        Out << " = &StructReturn;\n";
    }

    bool PrintedVar = false;

    // print local variable information for the function
    for (llvm::inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
        if (const llvm::AllocaInst *AI = isDirectAlloca(&*I)) {
            Out << "  ";
            printType(Out, AI->getAllocatedType(), false, GetValueName(AI));
            Out << ";    /* Address-exposed local */\n";
            PrintedVar = true;
        } else if (I->getType() != llvm::Type::getVoidTy(F.getContext()) && !isInlinableInst(*I)) {
            Out << "  ";
            printType(Out, I->getType(), false, GetValueName(&*I));
            Out << ";\n";

            if (llvm::isa<llvm::PHINode>(*I)) { // Print out PHI node temporaries as well...
                Out << "  ";
                printType(Out, I->getType(), false, GetValueName(&*I) + "__PHI");
                Out << ";\n";
            }
            PrintedVar = true;
        }
        // We need a temporary for the BitCast to use so it can pluck a value out
        // of a union to do the BitCast. This is separate from the need for a
        // variable to hold the result of the BitCast.
        if (isFPIntBitCast(*I)) {
            Out << "  llvmBitCastUnion " << GetValueName(&*I) << "__BITCAST_TEMPORARY;\n";
            PrintedVar = true;
        }
    }

    if (PrintedVar)
        Out << '\n';

    if (F.hasExternalLinkage() && F.getName() == "main")
        Out << "  CODE_FOR_MAIN();\n";

    // print the basic blocks
    for (llvm::Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
        if (llvm::Loop *L = LI->getLoopFor(&*BB)) {
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_9 // LLVM 3.9+
            if (L->getHeader()->getIterator() == BB && L->getParentLoop() == 0)
#else
            if (L->getHeader() == BB && L->getParentLoop() == 0)
#endif
                printLoop(L);
        } else {
            printBasicBlock(&*BB);
        }
    }

    Out << "}\n\n";
}

void CWriter::printLoop(llvm::Loop *L) {
    Out << "  do {     /* Syntactic loop '" << L->getHeader()->getName() << "' to make GCC happy */\n";
    for (unsigned i = 0, e = L->getBlocks().size(); i != e; ++i) {
        llvm::BasicBlock *BB = L->getBlocks()[i];
        llvm::Loop *BBLoop = LI->getLoopFor(BB);
        if (BBLoop == L)
            printBasicBlock(BB);
        else if (BB == BBLoop->getHeader() && BBLoop->getParentLoop() == L)
            printLoop(BBLoop);
    }
    Out << "  } while (1); /* end of syntactic loop '" << L->getHeader()->getName() << "' */\n";
}

void CWriter::printBasicBlock(llvm::BasicBlock *BB) {

    // Don't print the label for the basic block if there are no uses, or if
    // the only terminator use is the predecessor basic block's terminator.
    // We have to scan the use list because PHI nodes use basic blocks too but
    // do not require a label to be generated.
    //
    bool NeedsLabel = false;
    for (llvm::pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI)
        if (isGotoCodeNecessary(*PI, BB)) {
            NeedsLabel = true;
            break;
        }

    if (NeedsLabel)
        Out << GetValueName(BB) << ": {\n";

    // Output all of the instructions in the basic block...
    for (llvm::BasicBlock::iterator II = BB->begin(), E = --BB->end(); II != E; ++II) {
        if (!isInlinableInst(*II) && !isDirectAlloca(&*II)) {
            if (II->getType() != llvm::Type::getVoidTy(BB->getContext()) && !isInlineAsm(*II))
                outputLValue(&*II);
            else
                Out << "  ";
            writeInstComputationInline(*II);
            Out << ";\n";
        }
    }

    // Don't emit prefix or suffix for the terminator.
    visit(*BB->getTerminator());
    if (NeedsLabel)
        Out << "}\n"; // workaround g++ bug
}

// Specific Instruction type classes... note that all of the casts are
// necessary because we use the instruction classes as opaque types...
//
void CWriter::visitReturnInst(llvm::ReturnInst &I) {
    // If this is a struct return function, return the temporary struct.
    bool isStructReturn = I.getParent()->getParent()->hasStructRetAttr();

    if (isStructReturn) {
        Out << "  return StructReturn;\n";
        return;
    }

    // Don't output a void return if this is the last basic block in the function
    if (I.getNumOperands() == 0 && &*--I.getParent()->getParent()->end() == I.getParent() &&
        (!I.getParent()->size()) == 1) {
        return;
    }

    Out << "  return";
    if (I.getNumOperands()) {
        Out << ' ';
        writeOperand(I.getOperand(0));
    }
    Out << ";\n";
}

void CWriter::visitSwitchInst(llvm::SwitchInst &SI) {

    llvm::Value *Cond = SI.getCondition();

    Out << "  switch (";
    writeOperand(Cond);
    Out << ") {\n  default:\n";
    printPHICopiesForSuccessor(SI.getParent(), SI.getDefaultDest(), 2);
    printBranchToBlock(SI.getParent(), SI.getDefaultDest(), 2);
    Out << ";\n";

    for (llvm::SwitchInst::CaseIt i = SI.case_begin(), e = SI.case_end(); i != e; ++i) {
#if ISPC_LLVM_VERSION <= ISPC_LLVM_4_0
        llvm::ConstantInt *CaseVal = i.getCaseValue();
        llvm::BasicBlock *Succ = i.getCaseSuccessor();
#else // LLVM 5.0+
        llvm::ConstantInt *CaseVal = i->getCaseValue();
        llvm::BasicBlock *Succ = i->getCaseSuccessor();
#endif
        Out << "  case ";
        writeOperand(CaseVal);
        Out << ":\n";
        printPHICopiesForSuccessor(SI.getParent(), Succ, 2);
        printBranchToBlock(SI.getParent(), Succ, 2);

#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_5 // LLVM 3.5+
        if (llvm::Function::iterator(Succ) == std::next(llvm::Function::iterator(SI.getParent())))
#else
        if (llvm::Function::iterator(Succ) == llvm::next(llvm::Function::iterator(SI.getParent())))
#endif
            Out << "    break;\n";
    }

    Out << "  }\n";
}

void CWriter::visitIndirectBrInst(llvm::IndirectBrInst &IBI) {
    Out << "  goto *(void*)(";
    writeOperand(IBI.getOperand(0));
    Out << ");\n";
}

void CWriter::visitUnreachableInst(llvm::UnreachableInst &I) { Out << "  /*UNREACHABLE*/;\n"; }

bool CWriter::isGotoCodeNecessary(llvm::BasicBlock *From, llvm::BasicBlock *To) {
    /// FIXME: This should be reenabled, but loop reordering safe!!
    return true;

#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_5 // LLVM 3.5+
    if (std::next(llvm::Function::iterator(From)) != llvm::Function::iterator(To))
#else
    if (llvm::next(llvm::Function::iterator(From)) != llvm::Function::iterator(To))
#endif
        return true; // Not the direct successor, we need a goto.

    // llvm::isa<llvm::SwitchInst>(From->getTerminator())

    if (LI->getLoopFor(From) != LI->getLoopFor(To))
        return true;
    return false;
}

void CWriter::printPHICopiesForSuccessor(llvm::BasicBlock *CurBlock, llvm::BasicBlock *Successor, unsigned Indent) {
    for (llvm::BasicBlock::iterator I = Successor->begin(); llvm::isa<llvm::PHINode>(I); ++I) {
#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_7 /* 3.2, 3.3, 3.4, 3.5, 3.6, 3.7 */
        llvm::PHINode *PN = llvm::cast<llvm::PHINode>(I);
#else /* LLVM 3.8+ */
        llvm::PHINode *PN = llvm::cast<llvm::PHINode>(&*I);
#endif
        // Now we have to do the printing.
        llvm::Value *IV = PN->getIncomingValueForBlock(CurBlock);
        if (!llvm::isa<llvm::UndefValue>(IV)) {
            Out << std::string(Indent, ' ');
#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_7 /* 3.2, 3.3, 3.4, 3.5, 3.6, 3.7 */
            Out << "  " << GetValueName(I) << "__PHI = ";
#else /* LLVM 3.8+ */
            Out << "  " << GetValueName(&*I) << "__PHI = ";
#endif
            writeOperand(IV);
            Out << ";   /* for PHI node */\n";
        }
    }
}

void CWriter::printBranchToBlock(llvm::BasicBlock *CurBB, llvm::BasicBlock *Succ, unsigned Indent) {
    if (isGotoCodeNecessary(CurBB, Succ)) {
        Out << std::string(Indent, ' ') << "  goto ";
        writeOperand(Succ);
        Out << ";\n";
    }
}

// Branch instruction printing - Avoid printing out a branch to a basic block
// that immediately succeeds the current one.
//
void CWriter::visitBranchInst(llvm::BranchInst &I) {

    if (I.isConditional()) {
        if (isGotoCodeNecessary(I.getParent(), I.getSuccessor(0))) {
            Out << "  if (";
            writeOperand(I.getCondition());
            Out << ") {\n";

            printPHICopiesForSuccessor(I.getParent(), I.getSuccessor(0), 2);
            printBranchToBlock(I.getParent(), I.getSuccessor(0), 2);

            if (isGotoCodeNecessary(I.getParent(), I.getSuccessor(1))) {
                Out << "  } else {\n";
                printPHICopiesForSuccessor(I.getParent(), I.getSuccessor(1), 2);
                printBranchToBlock(I.getParent(), I.getSuccessor(1), 2);
            }
        } else {
            // First goto not necessary, assume second one is...
            Out << "  if (!";
            writeOperand(I.getCondition());
            Out << ") {\n";

            printPHICopiesForSuccessor(I.getParent(), I.getSuccessor(1), 2);
            printBranchToBlock(I.getParent(), I.getSuccessor(1), 2);
        }

        Out << "  }\n";
    } else {
        printPHICopiesForSuccessor(I.getParent(), I.getSuccessor(0), 0);
        printBranchToBlock(I.getParent(), I.getSuccessor(0), 0);
    }
    Out << "\n";
}

// PHI nodes get copied into temporary values at the end of predecessor basic
// blocks.  We now need to copy these temporary values into the REAL value for
// the PHI.
void CWriter::visitPHINode(llvm::PHINode &I) {
    writeOperand(&I);
    Out << "__PHI";
}

void CWriter::visitBinaryOperator(llvm::Instruction &I) {
    // binary instructions, shift instructions, setCond instructions.
    assert(!I.getType()->isPointerTy());

    if (llvm::isa<const llvm::VectorType>(I.getOperand(0)->getType())) {
        const char *intrinsic = NULL;
        switch (I.getOpcode()) {
        case llvm::Instruction::Add:
            intrinsic = "__add";
            break;
        case llvm::Instruction::FAdd:
            intrinsic = "__add";
            break;
        case llvm::Instruction::Sub:
            intrinsic = "__sub";
            break;
        case llvm::Instruction::FSub:
            intrinsic = "__sub";
            break;
        case llvm::Instruction::Mul:
            intrinsic = "__mul";
            break;
        case llvm::Instruction::FMul:
            intrinsic = "__mul";
            break;
        case llvm::Instruction::URem:
            intrinsic = "__urem";
            break;
        case llvm::Instruction::SRem:
            intrinsic = "__srem";
            break;
        case llvm::Instruction::FRem:
            intrinsic = "__frem";
            break;
        case llvm::Instruction::UDiv:
            intrinsic = "__udiv";
            break;
        case llvm::Instruction::SDiv:
            intrinsic = "__sdiv";
            break;
        case llvm::Instruction::FDiv:
            intrinsic = "__div";
            break;
        case llvm::Instruction::And:
            intrinsic = "__and";
            break;
        case llvm::Instruction::Or:
            intrinsic = "__or";
            break;
        case llvm::Instruction::Xor:
            intrinsic = "__xor";
            break;
        case llvm::Instruction::Shl:
            intrinsic = "__shl";
            break;
        case llvm::Instruction::LShr:
            intrinsic = "__lshr";
            break;
        case llvm::Instruction::AShr:
            intrinsic = "__ashr";
            break;
        default:
#ifndef NDEBUG
            llvm::errs() << "Invalid operator type!" << I;
#endif
            llvm_unreachable(0);
        }
        Out << intrinsic;
        Out << "(";
        writeOperand(I.getOperand(0));
        Out << ", ";
        if ((I.getOpcode() == llvm::Instruction::Shl || I.getOpcode() == llvm::Instruction::LShr ||
             I.getOpcode() == llvm::Instruction::AShr)) {
            llvm::Value *splat = NULL;
            if (LLVMVectorValuesAllEqual(I.getOperand(1), &splat)) {
                if (splat) {
                    // Avoid __extract_element(splat(value), 0), if possible.
                    writeOperand(splat);
                } else {
                    Out << "__extract_element(";
                    writeOperand(I.getOperand(1));
                    Out << ", 0) ";
                }
            } else
                writeOperand(I.getOperand(1));
        } else
            writeOperand(I.getOperand(1));
        Out << ")";
        return;
    }

    // We must cast the results of binary operations which might be promoted.
    bool needsCast = false;
    if ((I.getType() == llvm::Type::getInt8Ty(I.getContext())) ||
        (I.getType() == llvm::Type::getInt16Ty(I.getContext())) ||
        (I.getType() == llvm::Type::getFloatTy(I.getContext()))) {
        needsCast = true;
        Out << "((";
        printType(Out, I.getType(), false);
        Out << ")(";
    }

    // If this is a negation operation, print it out as such.  For FP, we don't
    // want to print "-0.0 - X".
#if ISPC_LLVM_VERSION > ISPC_LLVM_7_0 // LLVM 8.0+
    llvm::Value *X;
    if (match(&I, m_Neg(llvm::PatternMatch::m_Value(X)))) {
        Out << "-(";
        writeOperand(X);
        Out << ")";
    } else if (match(&I, m_FNeg(llvm::PatternMatch::m_Value(X)))) {
        Out << "-(";
        writeOperand(X);
        Out << ")";
    }
#else
    if (llvm::BinaryOperator::isNeg(&I)) {
        Out << "-(";
        writeOperand(llvm::BinaryOperator::getNegArgument(llvm::cast<llvm::BinaryOperator>(&I)));
        Out << ")";
    } else if (llvm::BinaryOperator::isFNeg(&I)) {
        Out << "-(";
        writeOperand(llvm::BinaryOperator::getFNegArgument(llvm::cast<llvm::BinaryOperator>(&I)));
        Out << ")";
    }
#endif
    else if (I.getOpcode() == llvm::Instruction::FRem) {
        // Output a call to fmod/fmodf instead of emitting a%b
        if (I.getType() == llvm::Type::getFloatTy(I.getContext()))
            Out << "fmodf(";
        else if (I.getType() == llvm::Type::getDoubleTy(I.getContext()))
            Out << "fmod(";
        else // all 3 flavors of long double
            Out << "fmodl(";
        writeOperand(I.getOperand(0));
        Out << ", ";
        writeOperand(I.getOperand(1));
        Out << ")";
    } else {

        // Write out the cast of the instruction's value back to the proper type
        // if necessary.
        bool NeedsClosingParens = writeInstructionCast(I);

        // Certain instructions require the operand to be forced to a specific type
        // so we use writeOperandWithCast here instead of writeOperand. Similarly
        // below for operand 1
        writeOperandWithCast(I.getOperand(0), I.getOpcode());

        switch (I.getOpcode()) {
        case llvm::Instruction::Add:
        case llvm::Instruction::FAdd:
            Out << " + ";
            break;
        case llvm::Instruction::Sub:
        case llvm::Instruction::FSub:
            Out << " - ";
            break;
        case llvm::Instruction::Mul:
        case llvm::Instruction::FMul:
            Out << " * ";
            break;
        case llvm::Instruction::URem:
        case llvm::Instruction::SRem:
        case llvm::Instruction::FRem:
            Out << " % ";
            break;
        case llvm::Instruction::UDiv:
        case llvm::Instruction::SDiv:
        case llvm::Instruction::FDiv:
            Out << " / ";
            break;
        case llvm::Instruction::And:
            Out << " & ";
            break;
        case llvm::Instruction::Or:
            Out << " | ";
            break;
        case llvm::Instruction::Xor:
            Out << " ^ ";
            break;
        case llvm::Instruction::Shl:
            Out << " << ";
            break;
        case llvm::Instruction::LShr:
        case llvm::Instruction::AShr:
            Out << " >> ";
            break;
        default:
#ifndef NDEBUG
            llvm::errs() << "Invalid operator type!" << I;
#endif
            llvm_unreachable(0);
        }

        writeOperandWithCast(I.getOperand(1), I.getOpcode());
        if (NeedsClosingParens)
            Out << "))";
    }

    if (needsCast) {
        Out << "))";
    }
}

static const char *lPredicateToString(llvm::CmpInst::Predicate p) {
    switch (p) {
    case llvm::ICmpInst::ICMP_EQ:
        return "__equal";
    case llvm::ICmpInst::ICMP_NE:
        return "__not_equal";
    case llvm::ICmpInst::ICMP_ULE:
        return "__unsigned_less_equal";
    case llvm::ICmpInst::ICMP_SLE:
        return "__signed_less_equal";
    case llvm::ICmpInst::ICMP_UGE:
        return "__unsigned_greater_equal";
    case llvm::ICmpInst::ICMP_SGE:
        return "__signed_greater_equal";
    case llvm::ICmpInst::ICMP_ULT:
        return "__unsigned_less_than";
    case llvm::ICmpInst::ICMP_SLT:
        return "__signed_less_than";
    case llvm::ICmpInst::ICMP_UGT:
        return "__unsigned_greater_than";
    case llvm::ICmpInst::ICMP_SGT:
        return "__signed_greater_than";

    case llvm::FCmpInst::FCMP_ORD:
        return "__ordered";
    case llvm::FCmpInst::FCMP_UNO:
        return "__unordered";
    case llvm::FCmpInst::FCMP_UEQ:
        return "__equal";
    case llvm::FCmpInst::FCMP_UNE:
        return "__not_equal";
    case llvm::FCmpInst::FCMP_ULT:
        return "__less_than";
    case llvm::FCmpInst::FCMP_ULE:
        return "__less_equal";
    case llvm::FCmpInst::FCMP_UGT:
        return "__greater_than";
    case llvm::FCmpInst::FCMP_UGE:
        return "__greater_equal";
    case llvm::FCmpInst::FCMP_OEQ:
        return "__equal";
    case llvm::FCmpInst::FCMP_ONE:
        return "__not_equal";
    case llvm::FCmpInst::FCMP_OLT:
        return "__less_than";
    case llvm::FCmpInst::FCMP_OLE:
        return "__less_equal";
    case llvm::FCmpInst::FCMP_OGT:
        return "__greater_than";
    case llvm::FCmpInst::FCMP_OGE:
        return "__greater_equal";

    default:
        llvm_unreachable(0);
        return NULL;
    }
}

static const char *lTypeToSuffix(llvm::Type *t) {
    llvm::VectorType *vt = llvm::dyn_cast<llvm::VectorType>(t);
    Assert(vt != NULL);
    t = vt->getElementType();

    switch (t->getTypeID()) {
    case llvm::Type::FloatTyID:
        return "float";
    case llvm::Type::DoubleTyID:
        return "double";
    case llvm::Type::IntegerTyID: {
        switch (llvm::cast<llvm::IntegerType>(t)->getBitWidth()) {
        case 1:
            return "i1";
        case 8:
            return "i8";
        case 16:
            return "i16";
        case 32:
            return "i32";
        case 64:
            return "i64";
        }
    }
    default:
        llvm_unreachable(0);
        return NULL;
    }
    return NULL;
}

void CWriter::visitICmpInst(llvm::ICmpInst &I) {
    bool isVector = llvm::isa<llvm::VectorType>(I.getOperand(0)->getType());

    if (isVector) {
        Out << lPredicateToString(I.getPredicate());
        Out << "_";
        Out << lTypeToSuffix(I.getOperand(0)->getType());
        Out << "(";
        writeOperand(I.getOperand(0));
        Out << ", ";
        writeOperand(I.getOperand(1));
        Out << ")";
        return;
    }

    // Write out the cast of the instruction's value back to the proper type
    // if necessary.
    bool NeedsClosingParens = writeInstructionCast(I);

    // Certain icmp predicate require the operand to be forced to a specific type
    // so we use writeOperandWithCast here instead of writeOperand. Similarly
    // below for operand 1
    writeOperandWithCast(I.getOperand(0), I);

    switch (I.getPredicate()) {
    case llvm::ICmpInst::ICMP_EQ:
        Out << " == ";
        break;
    case llvm::ICmpInst::ICMP_NE:
        Out << " != ";
        break;
    case llvm::ICmpInst::ICMP_ULE:
    case llvm::ICmpInst::ICMP_SLE:
        Out << " <= ";
        break;
    case llvm::ICmpInst::ICMP_UGE:
    case llvm::ICmpInst::ICMP_SGE:
        Out << " >= ";
        break;
    case llvm::ICmpInst::ICMP_ULT:
    case llvm::ICmpInst::ICMP_SLT:
        Out << " < ";
        break;
    case llvm::ICmpInst::ICMP_UGT:
    case llvm::ICmpInst::ICMP_SGT:
        Out << " > ";
        break;
    default:
#ifndef NDEBUG
        llvm::errs() << "Invalid icmp predicate!" << I;
#endif
        llvm_unreachable(0);
    }

    writeOperandWithCast(I.getOperand(1), I);
    if (NeedsClosingParens)
        Out << "))";
}

void CWriter::visitFCmpInst(llvm::FCmpInst &I) {
    bool isVector = llvm::isa<llvm::VectorType>(I.getOperand(0)->getType());

    if (I.getPredicate() == llvm::FCmpInst::FCMP_FALSE) {
        if (isVector)
            llvm::report_fatal_error("FIXME: vector FCMP_FALSE");
        else
            Out << "0";
        return;
    }
    if (I.getPredicate() == llvm::FCmpInst::FCMP_TRUE) {
        if (isVector)
            llvm::report_fatal_error("FIXME: vector FCMP_TRUE");
        else
            Out << "1";
        return;
    }

    if (isVector) {
        Out << lPredicateToString(I.getPredicate());
        Out << "_";
        Out << lTypeToSuffix(I.getOperand(0)->getType());
        Out << "(";
    } else {
        const char *op = 0;
        switch (I.getPredicate()) {
        default:
            llvm_unreachable("Illegal FCmp predicate");
        case llvm::FCmpInst::FCMP_ORD:
            op = "ord";
            break;
        case llvm::FCmpInst::FCMP_UNO:
            op = "uno";
            break;

        case llvm::FCmpInst::FCMP_UEQ:
            op = "ueq";
            break;
        case llvm::FCmpInst::FCMP_UNE:
            op = "une";
            break;
        case llvm::FCmpInst::FCMP_ULT:
            op = "ult";
            break;
        case llvm::FCmpInst::FCMP_ULE:
            op = "ule";
            break;
        case llvm::FCmpInst::FCMP_UGT:
            op = "ugt";
            break;
        case llvm::FCmpInst::FCMP_UGE:
            op = "uge";
            break;

        case llvm::FCmpInst::FCMP_OEQ:
            op = "oeq";
            break;
        case llvm::FCmpInst::FCMP_ONE:
            op = "one";
            break;
        case llvm::FCmpInst::FCMP_OLT:
            op = "olt";
            break;
        case llvm::FCmpInst::FCMP_OLE:
            op = "ole";
            break;
        case llvm::FCmpInst::FCMP_OGT:
            op = "ogt";
            break;
        case llvm::FCmpInst::FCMP_OGE:
            op = "oge";
            break;
        }

        Out << "llvm_fcmp_" << op << "(";
    }

    // Write the first operand
    writeOperand(I.getOperand(0));
    Out << ", ";
    // Write the second operand
    writeOperand(I.getOperand(1));
    Out << ")";
}

static const char *getFloatBitCastField(llvm::Type *Ty) {
    switch (Ty->getTypeID()) {
    default:
        llvm_unreachable("Invalid Type");
    case llvm::Type::FloatTyID:
        return "Float";
    case llvm::Type::DoubleTyID:
        return "Double";
    case llvm::Type::IntegerTyID: {
        unsigned NumBits = llvm::cast<llvm::IntegerType>(Ty)->getBitWidth();
        if (NumBits <= 32)
            return "Int32";
        else
            return "Int64";
    }
    }
}

void CWriter::visitCastInst(llvm::CastInst &I) {
    llvm::Type *DstTy = I.getType();
    llvm::Type *SrcTy = I.getOperand(0)->getType();
    if (isFPIntBitCast(I)) {
        Out << '(';
        // These int<->float and long<->double casts need to be handled specially
        Out << GetValueName(&I) << "__BITCAST_TEMPORARY." << getFloatBitCastField(I.getOperand(0)->getType()) << " = ";
        writeOperand(I.getOperand(0));
        Out << ", " << GetValueName(&I) << "__BITCAST_TEMPORARY." << getFloatBitCastField(I.getType());
        Out << ')';
        return;
    }

    if ((llvm::isa<llvm::VectorType>(DstTy)) && (!llvm::isa<llvm::VectorType>(SrcTy))) {
        writeOperand(I.getOperand(0));
        return;
    }

    Out << '(';
    bool closeParen = printCast(I.getOpcode(), SrcTy, DstTy);

    // Make a sext from i1 work by subtracting the i1 from 0 (an int).
    if (SrcTy == llvm::Type::getInt1Ty(I.getContext()) && I.getOpcode() == llvm::Instruction::SExt)
        Out << "0-";

    writeOperand(I.getOperand(0));

    if (DstTy == llvm::Type::getInt1Ty(I.getContext()) &&
        (I.getOpcode() == llvm::Instruction::Trunc || I.getOpcode() == llvm::Instruction::FPToUI ||
         I.getOpcode() == llvm::Instruction::FPToSI || I.getOpcode() == llvm::Instruction::PtrToInt)) {
        // Make sure we really get a trunc to bool by anding the operand with 1
        Out << "&1u";
    }
    Out << ')';
    if (closeParen)
        Out << ')';
}

void CWriter::visitSelectInst(llvm::SelectInst &I) {
    if (llvm::isa<llvm::VectorType>(I.getType())) {
        Out << "__select(";
        writeOperand(I.getCondition());
        Out << ", ";
        writeOperand(I.getTrueValue());
        Out << ", ";
        writeOperand(I.getFalseValue());
        Out << ")";
        return;
    }

    Out << "((";
    writeOperand(I.getCondition());
    Out << ") ? (";
    writeOperand(I.getTrueValue());
    Out << ") : (";
    writeOperand(I.getFalseValue());
    Out << "))";
}

// Returns the macro name or value of the max or min of an integer type
// (as defined in limits.h).
static void printLimitValue(llvm::IntegerType &Ty, bool isSigned, bool isMax, llvm::raw_ostream &Out) {
    const char *type = "";
    const char *sprefix = "";

    unsigned NumBits = Ty.getBitWidth();
    if (NumBits <= 8) {
        type = "CHAR";
        sprefix = "S";
    } else if (NumBits <= 16) {
        type = "SHRT";
    } else if (NumBits <= 32) {
        type = "INT";
    } else if (NumBits <= 64) {
        type = "LLONG";
    } else {
        llvm_unreachable("Bit widths > 64 not implemented yet");
    }

    if (isSigned)
        Out << sprefix << type << (isMax ? "_MAX" : "_MIN");
    else
        Out << "U" << type << (isMax ? "_MAX" : "0");
}

#ifndef NDEBUG
static bool isSupportedIntegerSize(llvm::IntegerType &T) {
    return T.getBitWidth() == 8 || T.getBitWidth() == 16 || T.getBitWidth() == 32 || T.getBitWidth() == 64;
}
#endif

void CWriter::printIntrinsicDefinition(const llvm::Function &F, llvm::raw_ostream &Out) {
    llvm::FunctionType *funT = F.getFunctionType();
    llvm::Type *retT = F.getReturnType();
    llvm::IntegerType *elemT = llvm::cast<llvm::IntegerType>(funT->getParamType(1));

    assert(isSupportedIntegerSize(*elemT) && "CBackend does not support arbitrary size integers.");
    assert(llvm::cast<llvm::StructType>(retT)->getElementType(0) == elemT && elemT == funT->getParamType(0) &&
           funT->getNumParams() == 2);

    switch (F.getIntrinsicID()) {
    default:
        llvm_unreachable("Unsupported Intrinsic.");
    case llvm::Intrinsic::uadd_with_overflow:
        // static inline Rty uadd_ixx(unsigned ixx a, unsigned ixx b) {
        //   Rty r;
        //   r.field0 = a + b;
        //   r.field1 = (r.field0 < a);
        //   return r;
        // }
        Out << "static inline ";
        printType(Out, retT);
        Out << GetValueName(&F);
        Out << "(";
        printSimpleType(Out, elemT, false);
        Out << "a,";
        printSimpleType(Out, elemT, false);
        Out << "b) {\n  ";
        printType(Out, retT);
        Out << "r;\n";
        Out << "  r.field0 = a + b;\n";
        Out << "  r.field1 = (r.field0 < a);\n";
        Out << "  return r;\n}\n";
        break;

    case llvm::Intrinsic::sadd_with_overflow:
        // static inline Rty sadd_ixx(ixx a, ixx b) {
        //   Rty r;
        //   r.field1 = (b > 0 && a > XX_MAX - b) ||
        //              (b < 0 && a < XX_MIN - b);
        //   r.field0 = r.field1 ? 0 : a + b;
        //   return r;
        // }
        Out << "static ";
        printType(Out, retT);
        Out << GetValueName(&F);
        Out << "(";
        printSimpleType(Out, elemT, true);
        Out << "a,";
        printSimpleType(Out, elemT, true);
        Out << "b) {\n  ";
        printType(Out, retT);
        Out << "r;\n";
        Out << "  r.field1 = (b > 0 && a > ";
        printLimitValue(*elemT, true, true, Out);
        Out << " - b) || (b < 0 && a < ";
        printLimitValue(*elemT, true, false, Out);
        Out << " - b);\n";
        Out << "  r.field0 = r.field1 ? 0 : a + b;\n";
        Out << "  return r;\n}\n";
        break;

    case llvm::Intrinsic::umul_with_overflow:
        Out << "static inline ";
        printType(Out, retT);
        Out << GetValueName(&F);
        Out << "(";
        printSimpleType(Out, elemT, false);
        Out << "a,";
        printSimpleType(Out, elemT, false);
        Out << "b) {\n  ";

        printType(Out, retT);
        Out << "r;\n";

        unsigned NumBits = llvm::cast<llvm::IntegerType>(elemT)->getBitWidth();
        std::stringstream str_type;
        if (NumBits <= 32)
            str_type << "uint" << 2 * NumBits << "_t";
        else {
            assert(NumBits <= 64 && "Bit widths > 128 not implemented yet");
            str_type << "llvmUInt128";
        }

        Out << "  " << str_type.str() << " result = (" << str_type.str() << ") a * (" << str_type.str() << ") b;\n";
        Out << "  r.field0 = result;\n";
        Out << "  r.field1 = result >> " << NumBits << ";\n";
        Out << "  return r;\n}\n";
        break;
    }
}

void CWriter::lowerIntrinsics(llvm::Function &F) {
    // This is used to keep track of intrinsics that get generated to a lowered
    // function. We must generate the prototypes before the function body which
    // will only be expanded on first use (by the loop below).
    std::vector<llvm::Function *> prototypesToGen;

    // Examine all the instructions in this function to find the intrinsics that
    // need to be lowered.
    for (llvm::Function::iterator BB = F.begin(), EE = F.end(); BB != EE; ++BB)
        for (llvm::BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E;)
            if (llvm::CallInst *CI = llvm::dyn_cast<llvm::CallInst>(I++))
                if (llvm::Function *F = CI->getCalledFunction())
                    switch (F->getIntrinsicID()) {
                    // We directly implement these intrinsics
                    case llvm::Intrinsic::not_intrinsic:
                    case llvm::Intrinsic::vastart:
                    case llvm::Intrinsic::vacopy:
                    case llvm::Intrinsic::vaend:
                    case llvm::Intrinsic::returnaddress:
                    case llvm::Intrinsic::frameaddress:
                    case llvm::Intrinsic::setjmp:
                    case llvm::Intrinsic::longjmp:
                    case llvm::Intrinsic::memset:
                    case llvm::Intrinsic::prefetch:
                    case llvm::Intrinsic::powi:
                    case llvm::Intrinsic::fabs:
                    case llvm::Intrinsic::x86_sse_cmp_ss:
                    case llvm::Intrinsic::x86_sse_cmp_ps:
                    case llvm::Intrinsic::x86_sse2_cmp_sd:
                    case llvm::Intrinsic::x86_sse2_cmp_pd:
                    case llvm::Intrinsic::ppc_altivec_lvsl:
                    case llvm::Intrinsic::uadd_with_overflow:
                    case llvm::Intrinsic::sadd_with_overflow:
                    case llvm::Intrinsic::trap:
                    case llvm::Intrinsic::objectsize:
                    case llvm::Intrinsic::readcyclecounter:
                    case llvm::Intrinsic::umul_with_overflow:
                    // Or we just ignore them because of their uselessness in C++ source
                    case llvm::Intrinsic::dbg_value:
                    case llvm::Intrinsic::dbg_declare:
                        break;
                    default:
                        // If this is an intrinsic that directly corresponds to a GCC
                        // builtin, we handle it.
                        const char *BuiltinName = "";
#define GET_GCC_BUILTIN_NAME
#define Intrinsic llvm::Intrinsic
#if ISPC_LLVM_VERSION == ISPC_LLVM_3_2
#include "llvm/Intrinsics.gen"
#elif ISPC_LLVM_VERSION <= ISPC_LLVM_6_0 /* LLVM 3.3-6.0 */
#include "llvm/IR/Intrinsics.gen"
#else /* LLVM 7.0+ */
// This looks completely broken, even in 3.2, need to figure out what's going on here
// and how to fix it (if needed).
//  #include "llvm/IR/Intrinsics.inc"
#endif
#undef Intrinsic
#undef GET_GCC_BUILTIN_NAME
                        // If we handle it, don't lower it.
                        if (BuiltinName[0])
                            break;

                        // All other intrinsic calls we must lower.
                        llvm::Instruction *Before = 0;
                        if (CI != &BB->front())
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_5 // LLVM 3.5+
                            Before = &*std::prev(llvm::BasicBlock::iterator(CI));
#else
                            Before = prior(llvm::BasicBlock::iterator(CI));
#endif

                        IL->LowerIntrinsicCall(CI);
                        if (Before) {  // Move iterator to instruction after call
#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_7 /* 3.2, 3.3, 3.4, 3.5, 3.6, 3.7 */
                            I = Before;
                            ++I;
#else /* LLVM 3.8+ */
                            I = Before->getIterator();
                            ++I;
#endif
                        } else {
                            I = BB->begin();
                        }
                        // If the intrinsic got lowered to another call, and that call has
                        // a definition then we need to make sure its prototype is emitted
                        // before any calls to it.
                        if (llvm::CallInst *Call = llvm::dyn_cast<llvm::CallInst>(I))
                            if (llvm::Function *NewF = Call->getCalledFunction())
                                if (!NewF->isDeclaration())
                                    prototypesToGen.push_back(NewF);

                        break;
                    }

    // We may have collected some prototypes to emit in the loop above.
    // Emit them now, before the function that uses them is emitted. But,
    // be careful not to emit them twice.
    std::vector<llvm::Function *>::iterator I = prototypesToGen.begin();
    std::vector<llvm::Function *>::iterator E = prototypesToGen.end();
    for (; I != E; ++I) {
        if (intrinsicPrototypesAlreadyGenerated.insert(*I).second) {
            Out << '\n';
            printFunctionSignature(*I, true);
            Out << ";\n";
        }
    }
}

void CWriter::visitCallInst(llvm::CallInst &I) {
    if (llvm::isa<llvm::InlineAsm>(I.getCalledValue()))
        return visitInlineAsm(I);

    bool WroteCallee = false;

    // Handle intrinsic function calls first...
    if (llvm::Function *F = I.getCalledFunction())
        if (llvm::Intrinsic::ID ID = (llvm::Intrinsic::ID)F->getIntrinsicID())
            if (visitBuiltinCall(I, ID, WroteCallee))
                return;

    llvm::Value *Callee = I.getCalledValue();

    llvm::PointerType *PTy = llvm::cast<llvm::PointerType>(Callee->getType());
    llvm::FunctionType *FTy = llvm::cast<llvm::FunctionType>(PTy->getElementType());

    // If this is a call to a struct-return function, assign to the first
    // parameter instead of passing it to the call.
#if ISPC_LLVM_VERSION == ISPC_LLVM_3_2
    const llvm::AttrListPtr &PAL = I.getAttributes();
#elif ISPC_LLVM_VERSION <= ISPC_LLVM_4_0
    const llvm::AttributeSet &PAL = I.getAttributes();
#else // LLVM 5.0+
    const llvm::AttributeList &PAL = I.getAttributes();
#endif

    bool hasByVal = I.hasByValArgument();
    bool isStructRet = (I.getNumArgOperands() > 0) && I.hasStructRetAttr();
    if (isStructRet) {
        writeOperandDeref(I.getArgOperand(0));
        Out << " = ";
    }

    if (I.isTailCall())
        Out << " /*tail*/ ";

    if (!WroteCallee) {
        // If this is an indirect call to a struct return function, we need to cast
        // the pointer. Ditto for indirect calls with byval arguments.
        bool NeedsCast = (hasByVal || isStructRet) && !llvm::isa<llvm::Function>(Callee);

        // GCC is a real PITA.  It does not permit codegening casts of functions to
        // function pointers if they are in a call (it generates a trap instruction
        // instead!).  We work around this by inserting a cast to void* in between
        // the function and the function pointer cast.  Unfortunately, we can't just
        // form the constant expression here, because the folder will immediately
        // nuke it.
        //
        // Note finally, that this is completely unsafe.  ANSI C does not guarantee
        // that void* and function pointers have the same size. :( To deal with this
        // in the common case, we handle casts where the number of arguments passed
        // match exactly.
        //
        if (llvm::ConstantExpr *CE = llvm::dyn_cast<llvm::ConstantExpr>(Callee))
            if (CE->isCast())
                if (llvm::Function *RF = llvm::dyn_cast<llvm::Function>(CE->getOperand(0))) {
                    NeedsCast = true;
                    Callee = RF;
                }

        if (Callee->getName() == "malloc" || Callee->getName() == "_aligned_malloc")
            Out << "(uint8_t *)";

        // This 'if' will fix 'soa-18.ispc' test (fails with optimizations off)
        // Yet the way the case is fixed is quite dirty and leads to many other fails

        // if (Callee->getName() == "__masked_store_i64") {
        //    llvm::CallSite CS(&I);
        //    llvm::CallSite::arg_iterator AI = CS.arg_begin();
        //    if (is_vec16_i64_ty(llvm::cast<llvm::PointerType>((*AI)->getType())->getElementType())) {
        //        Out << "/* Replacing store of vec16_i64 val into &vec16_i64 pointer with a simple copy */\n";
        //        // If we are trying to get a pointer to from a vec16_i64 var
        //        // It would be better to replace this instruction with a masked copy
        //        if (llvm::isa<llvm::GetElementPtrInst>(*AI)) {
        //            writeOperandDeref(*AI);
        //            Out << " = __select(";
        //            writeOperand(*(AI+2));
        //            Out << ", ";
        //            writeOperand(*(AI+1));
        //            Out << ", ";
        //            writeOperandDeref(*AI);
        //            Out << ")";
        //            return;
        //        }
        //    }
        //}

        if (NeedsCast) {
            // Ok, just cast the pointer type.
            Out << "((";
            if (isStructRet)
                printStructReturnPointerFunctionType(Out, PAL,
                                                     llvm::cast<llvm::PointerType>(I.getCalledValue()->getType()));
            else if (hasByVal)
                printType(Out, I.getCalledValue()->getType(), false, "", true, PAL);
            else
                printType(Out, I.getCalledValue()->getType());
            Out << ")(void*)";
        }
        writeOperand(Callee);
        if (NeedsCast)
            Out << ')';
    }

    Out << '(';

    bool PrintedArg = false;
    if (FTy->isVarArg() && !FTy->getNumParams()) {
        Out << "0 /*dummy arg*/";
        PrintedArg = true;
    }

    unsigned NumDeclaredParams = FTy->getNumParams();
    llvm::CallSite CS(&I);
    llvm::CallSite::arg_iterator AI = CS.arg_begin(), AE = CS.arg_end();
    unsigned ArgNo = 0;
    if (isStructRet) { // Skip struct return argument.
        ++AI;
        ++ArgNo;
    }

    for (; AI != AE; ++AI, ++ArgNo) {
        if (PrintedArg)
            Out << ", ";
        if (ArgNo == 0 && Callee->getName() == "posix_memalign") {
            // uint8_t** is incompatible with void** without explicit cast.
            // Should be do this any other functions?
            Out << "(void **)";
        } else if (ArgNo < NumDeclaredParams && (*AI)->getType() != FTy->getParamType(ArgNo)) {
            Out << '(';
            printType(Out, FTy->getParamType(ArgNo),
#if ISPC_LLVM_VERSION == ISPC_LLVM_3_2
                      PAL.getParamAttributes(ArgNo + 1).hasAttribute(llvm::Attributes::SExt)
#elif ISPC_LLVM_VERSION <= ISPC_LLVM_4_0
                      PAL.getParamAttributes(ArgNo + 1).hasAttribute(llvm::AttributeSet::FunctionIndex,
                                                                     llvm::Attribute::SExt)
#else // LLVM 5.0+
                      PAL.getParamAttributes(ArgNo + 1).hasAttribute(llvm::Attribute::SExt)
#endif
            );
            Out << ')';
        }
        // Check if the argument is expected to be passed by value.
#if ISPC_LLVM_VERSION <= ISPC_LLVM_4_0
        if (I.paramHasAttr(ArgNo + 1,
#else // LLVM 5.0+
        if (I.paramHasAttr(ArgNo,
#endif
#if ISPC_LLVM_VERSION == ISPC_LLVM_3_2
                           llvm::Attributes::ByVal
#else /* LLVM 3.3+ */
                           llvm::Attribute::ByVal
#endif
                           )) {
            writeOperandDeref(*AI);
        } else {
            writeOperand(*AI);
        }
        PrintedArg = true;
    }
    Out << ')';
}

/// visitBuiltinCall - Handle the call to the specified builtin.  Returns true
/// if the entire call is handled, return false if it wasn't handled, and
/// optionally set 'WroteCallee' if the callee has already been printed out.
bool CWriter::visitBuiltinCall(llvm::CallInst &I, llvm::Intrinsic::ID ID, bool &WroteCallee) {
    switch (ID) {
    default: {
        // If this is an intrinsic that directly corresponds to a GCC
        // builtin, we emit it here.
        const char *BuiltinName = "";
#define GET_GCC_BUILTIN_NAME
#define Intrinsic llvm::Intrinsic
#if ISPC_LLVM_VERSION == ISPC_LLVM_3_2
#include "llvm/Intrinsics.gen"
#elif ISPC_LLVM_VERSION <= ISPC_LLVM_6_0 /* LLVM 3.3-6.0 */
#include "llvm/IR/Intrinsics.gen"
#else /* LLVM 7.0+ */
// This looks completely broken, even in 3.2, need to figure out what's going on here
// and how to fix it (if needed).
//  #include "llvm/IR/Intrinsics.inc"
#endif
#undef Intrinsic
#undef GET_GCC_BUILTIN_NAME
        assert(BuiltinName[0] && "Unknown LLVM intrinsic!");

        Out << BuiltinName;
        WroteCallee = true;
        return false;
    }
    // Ignoring debug intrinsics
    case llvm::Intrinsic::dbg_value:
    case llvm::Intrinsic::dbg_declare:
        return true;
    case llvm::Intrinsic::vastart:
        Out << "0; ";

        Out << "va_start(*(va_list*)";
        writeOperand(I.getArgOperand(0));
        Out << ", ";
        // Output the last argument to the enclosing function.
        if (I.getParent()->getParent()->arg_empty())
            Out << "vararg_dummy_arg";
        else
            writeOperand(&*(std::prev(I.getParent()->getParent()->arg_end())));
        Out << ')';
        return true;
    case llvm::Intrinsic::vaend:
        if (!llvm::isa<llvm::ConstantPointerNull>(I.getArgOperand(0))) {
            Out << "0; va_end(*(va_list*)";
            writeOperand(I.getArgOperand(0));
            Out << ')';
        } else {
            Out << "va_end(*(va_list*)0)";
        }
        return true;
    case llvm::Intrinsic::vacopy:
        Out << "0; ";
        Out << "va_copy(*(va_list*)";
        writeOperand(I.getArgOperand(0));
        Out << ", *(va_list*)";
        writeOperand(I.getArgOperand(1));
        Out << ')';
        return true;
    case llvm::Intrinsic::returnaddress:
        Out << "__builtin_return_address(";
        writeOperand(I.getArgOperand(0));
        Out << ')';
        return true;
    case llvm::Intrinsic::frameaddress:
        Out << "__builtin_frame_address(";
        writeOperand(I.getArgOperand(0));
        Out << ')';
        return true;
    case llvm::Intrinsic::powi:
        Out << "__builtin_powi(";
        writeOperand(I.getArgOperand(0));
        Out << ", ";
        writeOperand(I.getArgOperand(1));
        Out << ')';
        return true;
    case llvm::Intrinsic::fabs:
        Out << "__builtin_fabs(";
        writeOperand(I.getArgOperand(0));
        Out << ')';
        return true;
    case llvm::Intrinsic::setjmp:
        Out << "setjmp(*(jmp_buf*)";
        writeOperand(I.getArgOperand(0));
        Out << ')';
        return true;
    case llvm::Intrinsic::longjmp:
        Out << "longjmp(*(jmp_buf*)";
        writeOperand(I.getArgOperand(0));
        Out << ", ";
        writeOperand(I.getArgOperand(1));
        Out << ')';
        return true;
    case llvm::Intrinsic::memset:
        Out << "Memset(";
        writeOperand(I.getArgOperand(0));
        Out << ", ";
        writeOperand(I.getArgOperand(1));
        Out << ", ";
        writeOperand(I.getArgOperand(2));
        Out << ')';
        return true;
    case llvm::Intrinsic::prefetch:
        Out << "LLVM_PREFETCH((const void *)";
        writeOperand(I.getArgOperand(0));
        Out << ", ";
        writeOperand(I.getArgOperand(1));
        Out << ", ";
        writeOperand(I.getArgOperand(2));
        Out << ")";
        return true;
    case llvm::Intrinsic::stacksave:
        // Emit this as: Val = 0; *((void**)&Val) = __builtin_stack_save()
        // to work around GCC bugs (see PR1809).
        Out << "0; *((void**)&" << GetValueName(&I) << ") = __builtin_stack_save()";
        return true;
    case llvm::Intrinsic::x86_sse_cmp_ss:
    case llvm::Intrinsic::x86_sse_cmp_ps:
    case llvm::Intrinsic::x86_sse2_cmp_sd:
    case llvm::Intrinsic::x86_sse2_cmp_pd:
        Out << '(';
        printType(Out, I.getType());
        Out << ')';
        // Multiple GCC builtins multiplex onto this intrinsic.
        switch (llvm::cast<llvm::ConstantInt>(I.getArgOperand(2))->getZExtValue()) {
        default:
            llvm_unreachable("Invalid llvm.x86.sse.cmp!");
        case 0:
            Out << "__builtin_ia32_cmpeq";
            break;
        case 1:
            Out << "__builtin_ia32_cmplt";
            break;
        case 2:
            Out << "__builtin_ia32_cmple";
            break;
        case 3:
            Out << "__builtin_ia32_cmpunord";
            break;
        case 4:
            Out << "__builtin_ia32_cmpneq";
            break;
        case 5:
            Out << "__builtin_ia32_cmpnlt";
            break;
        case 6:
            Out << "__builtin_ia32_cmpnle";
            break;
        case 7:
            Out << "__builtin_ia32_cmpord";
            break;
        }
        if (ID == llvm::Intrinsic::x86_sse_cmp_ps || ID == llvm::Intrinsic::x86_sse2_cmp_pd)
            Out << 'p';
        else
            Out << 's';
        if (ID == llvm::Intrinsic::x86_sse_cmp_ss || ID == llvm::Intrinsic::x86_sse_cmp_ps)
            Out << 's';
        else
            Out << 'd';

        Out << "(";
        writeOperand(I.getArgOperand(0));
        Out << ", ";
        writeOperand(I.getArgOperand(1));
        Out << ")";
        return true;
    case llvm::Intrinsic::ppc_altivec_lvsl:
        Out << '(';
        printType(Out, I.getType());
        Out << ')';
        Out << "__builtin_altivec_lvsl(0, (void*)";
        writeOperand(I.getArgOperand(0));
        Out << ")";
        return true;
    case llvm::Intrinsic::uadd_with_overflow:
    case llvm::Intrinsic::sadd_with_overflow:
    case llvm::Intrinsic::umul_with_overflow:
        Out << GetValueName(I.getCalledFunction()) << "(";
        writeOperand(I.getArgOperand(0));
        Out << ", ";
        writeOperand(I.getArgOperand(1));
        Out << ")";
        return true;
    case llvm::Intrinsic::trap:
        Out << "abort()";
        return true;
    case llvm::Intrinsic::objectsize:
        return true;
    case llvm::Intrinsic::readcyclecounter:
        Out << "__clock()";
        return true;
    }
}

// TODO: assumptions about what consume arguments from the call are likely wrong
//      handle communitivity
void CWriter::visitInlineAsm(llvm::CallInst &CI) { assert(!"Inline assembly not supported"); }

void CWriter::visitAllocaInst(llvm::AllocaInst &I) {
    Out << '(';
    printType(Out, I.getType());
    Out << ") alloca(sizeof(";
    printType(Out, I.getType()->getElementType());
    Out << ')';
    if (I.isArrayAllocation()) {
        Out << " * ";
        writeOperand(I.getOperand(0));
    }
    Out << ')';
}

void CWriter::printGEPExpression(llvm::Value *Ptr, llvm::gep_type_iterator I, llvm::gep_type_iterator E, bool Static) {

    // If there are no indices, just print out the pointer.
    if (I == E) {
        writeOperand(Ptr);
        return;
    }

    // Find out if the last index is into a vector.  If so, we have to print this
    // specially.  Since vectors can't have elements of indexable type, only the
    // last index could possibly be of a vector element.
    llvm::VectorType *LastIndexIsVector = 0;
    {
        for (llvm::gep_type_iterator TmpI = I; TmpI != E; ++TmpI)
#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_9
            LastIndexIsVector = llvm::dyn_cast<llvm::VectorType>(*TmpI);
#else // LLVM 4.0+
            LastIndexIsVector = llvm::dyn_cast<llvm::VectorType>(TmpI.getIndexedType());
#endif
    }

    Out << "(";

    // If the last index is into a vector, we can't print it as &a[i][j] because
    // we can't index into a vector with j in GCC.  Instead, emit this as
    // (((float*)&a[i])+j)
    if (LastIndexIsVector) {
        Out << "((";
        printType(Out, llvm::PointerType::getUnqual(LastIndexIsVector->getElementType()));
        Out << ")(";
    }

    Out << '&';

    llvm::Type *ParentTy = Ptr->getType();

    // If the first index is 0 (very typical) we can do a number of
    // simplifications to clean up the code.
    llvm::Value *FirstOp = I.getOperand();
    if (!llvm::isa<llvm::Constant>(FirstOp) || !llvm::cast<llvm::Constant>(FirstOp)->isNullValue()) {
        // First index isn't simple, print it the hard way.
        writeOperand(Ptr);
    } else {
        ParentTy = I.getIndexedType(); // Skip the zero index.
        ++I;

        // Okay, emit the first operand. If Ptr is something that is already address
        // exposed, like a global, avoid emitting (&foo)[0], just emit foo instead.
        if (isAddressExposed(Ptr)) {
            writeOperandInternal(Ptr, Static);
#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_9
        } else if (I != E && (*I)->isStructTy()) {
#else // LLVM 4.0+
        } else if (I != E && I.isStruct()) {
#endif
            // If we didn't already emit the first operand, see if we can print it as
            // P->f instead of "P[0].f"
            writeOperand(Ptr);
            Out << "->field" << llvm::cast<llvm::ConstantInt>(I.getOperand())->getZExtValue();
            ParentTy = I.getIndexedType();
            ++I; // eat the struct index as well.
        } else {
            // Instead of emitting P[0][1], emit (*P)[1], which is more idiomatic.
            Out << "(*";
            writeOperand(Ptr);
            Out << ")";
        }
    }

    for (; I != E; ++I) {
        if (ParentTy->isStructTy()) {
            Out << ".field" << llvm::cast<llvm::ConstantInt>(I.getOperand())->getZExtValue();
        } else if (ParentTy->isArrayTy()) {
            Out << ".array[";
            writeOperandWithCast(I.getOperand(), llvm::Instruction::GetElementPtr);
            Out << ']';
        } else if (!ParentTy->isVectorTy()) {
            Out << '[';
            writeOperandWithCast(I.getOperand(), llvm::Instruction::GetElementPtr);
            Out << ']';
        } else {
            // If the last index is into a vector, then print it out as "+j)".  This
            // works with the 'LastIndexIsVector' code above.
            if (llvm::isa<llvm::Constant>(I.getOperand()) &&
                llvm::cast<llvm::Constant>(I.getOperand())->isNullValue()) {
                Out << "))"; // avoid "+0".
            } else {
                Out << ")+(";
                writeOperandWithCast(I.getOperand(), llvm::Instruction::GetElementPtr);
                Out << "))";
            }
        }
        ParentTy = I.getIndexedType();
    }
    Out << ")";
}

void CWriter::writeMemoryAccess(llvm::Value *Operand, llvm::Type *OperandType, bool IsVolatile, unsigned Alignment) {
    assert(!llvm::isa<llvm::VectorType>(OperandType));
    bool IsUnaligned = Alignment && Alignment < TD->getABITypeAlignment(OperandType);

    llvm::IntegerType *ITy = llvm::dyn_cast<llvm::IntegerType>(OperandType);
    if (!IsUnaligned)
        Out << '*';
    if (IsVolatile || IsUnaligned) {
        Out << "((";
        if (IsUnaligned && ITy && (ITy->getBitWidth() > 64))
            Out << "iN_" << ITy->getBitWidth() << "_align_" << Alignment << " *)";
        else {
            if (IsUnaligned)
                Out << "struct __attribute__ ((packed, aligned(" << Alignment << "))) {";
            printType(Out, OperandType, false, IsUnaligned ? "data" : "volatile*");
            if (IsUnaligned) {
                Out << "; } ";
                if (IsVolatile)
                    Out << "volatile ";
                Out << "*";
            }
            Out << ")";
        }
    }

    writeOperand(Operand);

    if (IsVolatile || IsUnaligned) {
        Out << ')';
        if (IsUnaligned)
            Out << "->data";
    }
}

void CWriter::visitLoadInst(llvm::LoadInst &I) {
    llvm::VectorType *VT = llvm::dyn_cast<llvm::VectorType>(I.getType());
    if (VT != NULL) {
        Out << "__load<" << I.getAlignment() << ">(";
        writeOperand(I.getOperand(0));
        Out << ")";
        return;
    }

    writeMemoryAccess(I.getOperand(0), I.getType(), I.isVolatile(), I.getAlignment());
}

void CWriter::visitStoreInst(llvm::StoreInst &I) {
    llvm::VectorType *VT = llvm::dyn_cast<llvm::VectorType>(I.getOperand(0)->getType());
    if (VT != NULL) {
        Out << "__store<" << I.getAlignment() << ">(";
        writeOperand(I.getOperand(1));
        Out << ", ";
        writeOperand(I.getOperand(0));
        Out << ")";
        return;
    }

    writeMemoryAccess(I.getPointerOperand(), I.getOperand(0)->getType(), I.isVolatile(), I.getAlignment());
    Out << " = ";
    llvm::Value *Operand = I.getOperand(0);
    llvm::Constant *BitMask = 0;
    if (llvm::IntegerType *ITy = llvm::dyn_cast<llvm::IntegerType>(Operand->getType()))
        if (!ITy->isPowerOf2ByteWidth())
            // We have a bit width that doesn't match an even power-of-2 byte
            // size. Consequently we must & the value with the type's bit mask
            BitMask = llvm::ConstantInt::get(ITy, ITy->getBitMask());
    if (BitMask)
        Out << "((";
    writeOperand(Operand);
    if (BitMask) {
        Out << ") & ";
        printConstant(BitMask, false);
        Out << ")";
    }
}

void CWriter::visitGetElementPtrInst(llvm::GetElementPtrInst &I) {
    printGEPExpression(I.getPointerOperand(), gep_type_begin(I), gep_type_end(I), false);
}

void CWriter::visitVAArgInst(llvm::VAArgInst &I) {
    Out << "va_arg(*(va_list*)";
    writeOperand(I.getOperand(0));
    Out << ", ";
    printType(Out, I.getType());
    Out << ");\n ";
}

void CWriter::visitInsertElementInst(llvm::InsertElementInst &I) {
#if 0
  Type *EltTy = I.getType()->getElementType();
  writeOperand(I.getOperand(0));
  Out << ";\n  ";
  Out << "((";
  printType(Out, llvm::PointerType::getUnqual(EltTy));
  Out << ")(&" << GetValueName(&I) << "))[";
  writeOperand(I.getOperand(2));
  Out << "] = (";
  writeOperand(I.getOperand(1));
  Out << ")";
#else
    writeOperand(I.getOperand(0));
    Out << ";\n  ";
    Out << "__insert_element(&" << GetValueName(&I) << ", ";
    writeOperand(I.getOperand(2));
    Out << ", ";
    writeOperand(I.getOperand(1));
    Out << ")";
#endif
}

void CWriter::visitExtractElementInst(llvm::ExtractElementInst &I) {
    // We know that our operand is not inlined.
#if 0
  Out << "((";
  Type *EltTy =
    llvm::cast<llvm::VectorType>(I.getOperand(0)->getType())->getElementType();
  printType(Out, llvm::PointerType::getUnqual(EltTy));
  Out << ")(&" << GetValueName(I.getOperand(0)) << "))[";
  writeOperand(I.getOperand(1));
  Out << "]";
#else
    Out << "(__extract_element(";
    writeOperand(I.getOperand(0));
    Out << ", ";
    writeOperand(I.getOperand(1));
    Out << "))";
#endif
}

void CWriter::visitShuffleVectorInst(llvm::ShuffleVectorInst &SVI) {
    printType(Out, SVI.getType());
    Out << "(";
    llvm::VectorType *VT = SVI.getType();
    unsigned NumElts = VT->getNumElements();
    llvm::Type *EltTy = VT->getElementType();
    llvm::VectorType *OpTy = llvm::dyn_cast<llvm::VectorType>(SVI.getOperand(0)->getType());
    unsigned OpElts = OpTy->getNumElements();

    for (unsigned i = 0; i != NumElts; ++i) {
        if (i)
            Out << ", ";
        int SrcVal = SVI.getMaskValue(i);
        if ((unsigned)SrcVal >= 2 * OpElts) {
            Out << " 0/*undef*/ ";
        } else {
            llvm::Value *Op = SVI.getOperand((unsigned)SrcVal >= OpElts);
            SrcVal &= OpElts - 1;

            if (llvm::isa<llvm::ConstantVector>(Op)) {
                printConstant(llvm::cast<llvm::ConstantVector>(Op)->getOperand(SrcVal), false);
            } else if (llvm::isa<llvm::ConstantAggregateZero>(Op) || llvm::isa<llvm::UndefValue>(Op)) {
                Out << "0";
            } else {
                // Do an extractelement of this value from the appropriate input.
                Out << " \n#if defined(KNC) \n";
                if (OpElts != 1) { // all __vec16_* have overloaded operator []
                    Out << "(" << GetValueName(Op) << ")[" << SrcVal << "]";
                } else { // but __vec1_* don't have it
                    Out << "((";
                    printType(Out, llvm::PointerType::getUnqual(EltTy));
                    Out << ")(&" << GetValueName(Op) << "))[" << SrcVal << "]";
                }
                Out << " \n#else \n";
                Out << "((";
                printType(Out, llvm::PointerType::getUnqual(EltTy));
                Out << ")(&" << GetValueName(Op) << "))[" << SrcVal << "]";
                Out << " \n#endif \n";
            }
        }
    }
    Out << ")";
}

void CWriter::visitInsertValueInst(llvm::InsertValueInst &IVI) {
    // Start by copying the entire aggregate value into the result variable.
    writeOperand(IVI.getOperand(0));
    Out << ";\n  ";

    // Then do the insert to update the field.
    Out << GetValueName(&IVI);
    for (const unsigned *b = IVI.idx_begin(), *i = b, *e = IVI.idx_end(); i != e; ++i) {
        llvm::Type *IndexedTy =
            (b == i) ? IVI.getOperand(0)->getType()
                     : llvm::ExtractValueInst::getIndexedType(IVI.getOperand(0)->getType(), llvm::makeArrayRef(b, i));
        if (IndexedTy->isArrayTy())
            Out << ".array[" << *i << "]";
        else
            Out << ".field" << *i;
    }
    Out << " = ";
    writeOperand(IVI.getOperand(1));
}

void CWriter::visitExtractValueInst(llvm::ExtractValueInst &EVI) {
    Out << "(";
    if (llvm::isa<llvm::UndefValue>(EVI.getOperand(0))) {
        // FIXME: need to handle these--a 0 initializer won't do...
        assert(!llvm::isa<llvm::VectorType>(EVI.getType()));
        Out << "(";
        printType(Out, EVI.getType());
        Out << ") 0/*UNDEF*/";
    } else {
        Out << GetValueName(EVI.getOperand(0));
        for (const unsigned *b = EVI.idx_begin(), *i = b, *e = EVI.idx_end(); i != e; ++i) {
            llvm::Type *IndexedTy = (b == i) ? EVI.getOperand(0)->getType()
                                             : llvm::ExtractValueInst::getIndexedType(EVI.getOperand(0)->getType(),
                                                                                      llvm::makeArrayRef(b, i));
            if (IndexedTy->isArrayTy())
                Out << ".array[" << *i << "]";
            else
                Out << ".field" << *i;
        }
    }
    Out << ")";
}

void CWriter::visitAtomicRMWInst(llvm::AtomicRMWInst &AI) {
    Out << "(";
    Out << "__atomic_";
    switch (AI.getOperation()) {
    default:
        llvm_unreachable("Unhandled case in visitAtomicRMWInst!");
    case llvm::AtomicRMWInst::Add:
        Out << "add";
        break;
    case llvm::AtomicRMWInst::Sub:
        Out << "sub";
        break;
    case llvm::AtomicRMWInst::Xchg:
        Out << "xchg";
        break;
    case llvm::AtomicRMWInst::And:
        Out << "and";
        break;
    case llvm::AtomicRMWInst::Nand:
        Out << "nand";
        break;
    case llvm::AtomicRMWInst::Or:
        Out << "or";
        break;
    case llvm::AtomicRMWInst::Xor:
        Out << "xor";
        break;
    case llvm::AtomicRMWInst::Min:
        Out << "min";
        break;
    case llvm::AtomicRMWInst::Max:
        Out << "max";
        break;
    case llvm::AtomicRMWInst::UMin:
        Out << "umin";
        break;
    case llvm::AtomicRMWInst::UMax:
        Out << "umax";
        break;
    }
    Out << "(";
    writeOperand(AI.getOperand(0));
    Out << ", ";
    writeOperand(AI.getOperand(1));
    Out << "))";
}

void CWriter::visitAtomicCmpXchgInst(llvm::AtomicCmpXchgInst &ACXI) {
    Out << "(";
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_5 // LLVM 3.5+
    printType(Out, ACXI.getType(), false);
    Out << "::init("; // LLVM cmpxchg returns a struct, so we need make an assighment properly
#endif
    Out << "__atomic_cmpxchg(";
    writeOperand(ACXI.getPointerOperand());
    Out << ", ";
    writeOperand(ACXI.getCompareOperand());
    Out << ", ";
    writeOperand(ACXI.getNewValOperand());
    Out << ")";
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_5 // LLVM 3.5+
    Out << ", true /* There is no way to learn the value of this bit inside ISPC, so making it constant */)";
#endif
    Out << ")";
}

///////////////////////////////////////////////////////////////////////////
// SmearCleanupPass

class SmearCleanupPass : public llvm::BasicBlockPass {
  public:
    SmearCleanupPass(llvm::Module *m, int width) : BasicBlockPass(ID) {
        module = m;
        vectorWidth = width;
    }

#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_9 // <= 3.9
    const char *getPassName() const { return "Smear Cleanup Pass"; }
#else // LLVM 4.0+
    llvm::StringRef getPassName() const { return "Smear Cleanup Pass"; }
#endif
    bool runOnBasicBlock(llvm::BasicBlock &BB);

    static char ID;
    llvm::Module *module;
    unsigned int vectorWidth;

  private:
    unsigned int ChainLength(llvm::InsertElementInst *inst) const;
    llvm::Value *getInsertChainSmearValue(llvm::Instruction *inst) const;
    llvm::Value *getShuffleSmearValue(llvm::Instruction *inst) const;
};

char SmearCleanupPass::ID = 0;

unsigned int SmearCleanupPass::ChainLength(llvm::InsertElementInst *inst) const {
    unsigned int length = 0;
    while (inst != NULL) {
        ++length;
        inst = llvm::dyn_cast<llvm::InsertElementInst>(inst->getOperand(0));
    }
    return length;
}

llvm::Value *SmearCleanupPass::getInsertChainSmearValue(llvm::Instruction *inst) const {
    // TODO: we don't check indexes where we do insertion, so we may trigger
    // transformation for a wrong chain.
    // This way of doing broadcast is obsolete and should be probably removed
    // some day.

    llvm::InsertElementInst *insertInst = llvm::dyn_cast<llvm::InsertElementInst>(inst);
    if (!insertInst) {
        return NULL;
    }

    // We consider only chians of vectorWidth length.
    if (ChainLength(insertInst) != vectorWidth) {
        return NULL;
    }

    // FIXME: we only want to do this to vectors with width equal to
    // the target vector width.  But we can't easily get that here, so
    // for now we at least avoid one case where we definitely don't
    // want to do this.
    llvm::VectorType *vt = llvm::dyn_cast<llvm::VectorType>(insertInst->getType());
    if (vt->getNumElements() == 1) {
        return NULL;
    }

    llvm::Value *smearValue = NULL;
    while (insertInst != NULL) {
        // operand 1 is inserted value
        llvm::Value *insertValue = insertInst->getOperand(1);
        if (smearValue == NULL) {
            smearValue = insertValue;
        } else if (smearValue != insertValue) {
            return NULL;
        }

        // operand 0 is a vector to insert into.
        insertInst = llvm::dyn_cast<llvm::InsertElementInst>(insertInst->getOperand(0));
    }
    assert(smearValue != NULL);

    return smearValue;
}

llvm::Value *SmearCleanupPass::getShuffleSmearValue(llvm::Instruction *inst) const {
    llvm::ShuffleVectorInst *shuffleInst = llvm::dyn_cast<llvm::ShuffleVectorInst>(inst);
    if (!shuffleInst) {
        return NULL;
    }

    llvm::Constant *mask = llvm::dyn_cast<llvm::Constant>(shuffleInst->getOperand(2));

    // Check that the shuffle is a broadcast of the element of the first vector,
    // i.e. mask vector is vector with equal elements of expected size.
    if (!(mask &&
#if ISPC_LLVM_VERSION == ISPC_LLVM_3_2
          (mask->isNullValue() ||
           (shuffleInst->getMask()->getType()->isVectorTy() &&
            llvm::dyn_cast<llvm::ConstantVector>(shuffleInst->getMask())->getSplatValue() != 0)) &&
#else
          (mask->isNullValue() || (shuffleInst->getMask()->getSplatValue() != 0)) &&
#endif
          llvm::dyn_cast<llvm::VectorType>(mask->getType())->getNumElements() == vectorWidth)) {
        return NULL;
    }

    llvm::InsertElementInst *insertInst = llvm::dyn_cast<llvm::InsertElementInst>(shuffleInst->getOperand(0));

    // Check that it's an InsertElementInst that inserts a value to first element.
    if (!(insertInst && llvm::isa<llvm::Constant>(insertInst->getOperand(2)) &&
          llvm::dyn_cast<llvm::Constant>(insertInst->getOperand(2))->isNullValue())) {

        // We can't extract element from vec1
        llvm::VectorType *operandVec = llvm::dyn_cast<llvm::VectorType>(shuffleInst->getOperand(0)->getType());
        if (operandVec && operandVec->getNumElements() == 1)
            return NULL;

        // Insert ExtractElementInstr to get value for smear

        llvm::Function *extractFunc = module->getFunction("__extract_element");

        if (extractFunc == NULL) {
            // Declare the __extract_element function if needed; it takes a vector and
            // a scalar parameter and returns a scalar of the vector parameter type.
#if ISPC_LLVM_VERSION <= ISPC_LLVM_8_0
            llvm::Constant *ef =
#if ISPC_LLVM_VERSION <= ISPC_LLVM_4_0
                module->getOrInsertFunction(
                    "__extract_element", shuffleInst->getOperand(0)->getType()->getVectorElementType(),
                    shuffleInst->getOperand(0)->getType(), llvm::IntegerType::get(module->getContext(), 32), NULL);
#else // LLVM 5.0+
                module->getOrInsertFunction(
                    "__extract_element", shuffleInst->getOperand(0)->getType()->getVectorElementType(),
                    shuffleInst->getOperand(0)->getType(), llvm::IntegerType::get(module->getContext(), 32));
#endif
            extractFunc = llvm::dyn_cast<llvm::Function>(ef);
#else // LLVM 9.0+
            llvm::FunctionCallee ef = module->getOrInsertFunction(
                "__extract_element", shuffleInst->getOperand(0)->getType()->getVectorElementType(),
                shuffleInst->getOperand(0)->getType(), llvm::IntegerType::get(module->getContext(), 32));
            extractFunc = llvm::dyn_cast<llvm::Function>(ef.getCallee());
#endif
            assert(extractFunc != NULL);
            extractFunc->setDoesNotThrow();
            extractFunc->setOnlyReadsMemory();
        }

        if (extractFunc == NULL) {
            return NULL;
        }
        llvm::Instruction *extractCall =
            llvm::ExtractElementInst::Create(shuffleInst->getOperand(0),
#if ISPC_LLVM_VERSION == ISPC_LLVM_3_2
                                             // mask is of VectorType
                                             llvm::dyn_cast<llvm::ConstantVector>(mask)->getSplatValue(),
#else
                                             mask->getSplatValue(),
#endif
                                             "__extract_element", inst);
        return extractCall;
    }

    llvm::Value *result = insertInst->getOperand(1);

    return result;
}

bool SmearCleanupPass::runOnBasicBlock(llvm::BasicBlock &bb) {
    bool modifiedAny = false;

restart:
    for (llvm::BasicBlock::iterator iter = bb.begin(), e = bb.end(); iter != e; ++iter) {
        llvm::Value *smearValue = NULL;

        if (!(smearValue = getInsertChainSmearValue(&*iter)) && !(smearValue = getShuffleSmearValue(&*iter))) {
            continue;
        }

        llvm::Type *smearType = smearValue->getType();
        const char *smearFuncName = lGetTypedFunc("smear", smearType, vectorWidth);
        if (smearFuncName != NULL) {
            llvm::Function *smearFunc = module->getFunction(smearFuncName);
            if (smearFunc == NULL) {
                // Declare the smear function if needed; it takes a single
                // scalar parameter and returns a vector of the same
                // parameter type.
#if ISPC_LLVM_VERSION <= ISPC_LLVM_8_0
                llvm::Constant *sf =
#if ISPC_LLVM_VERSION <= ISPC_LLVM_4_0
                    module->getOrInsertFunction(smearFuncName, iter->getType(), smearType, NULL);
#else  // LLVM 5.0+
                    module->getOrInsertFunction(smearFuncName, iter->getType(), smearType);
#endif // LLVM 9.0+
                smearFunc = llvm::dyn_cast<llvm::Function>(sf);
#else
                llvm::FunctionCallee sf = module->getOrInsertFunction(smearFuncName, iter->getType(), smearType);
                smearFunc = llvm::dyn_cast<llvm::Function>(sf.getCallee());
#endif
                assert(smearFunc != NULL);
                smearFunc->setDoesNotThrow();
                smearFunc->setDoesNotAccessMemory();
            }

            assert(smearFunc != NULL);
            llvm::Value *args[1] = {smearValue};
            llvm::ArrayRef<llvm::Value *> argArray(&args[0], &args[1]);
            llvm::Instruction *smearCall = llvm::CallInst::Create(
                smearFunc, argArray, LLVMGetName(smearValue, "_smear"), (llvm::Instruction *)NULL);

            ReplaceInstWithInst(&*iter, smearCall);

            modifiedAny = true;
            goto restart;
        }
    }

    return modifiedAny;
}

///////////////////////////////////////////////////////////////////////////
// AndCmpCleanupPass

class AndCmpCleanupPass : public llvm::BasicBlockPass {
  public:
    AndCmpCleanupPass() : BasicBlockPass(ID) {}

#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_9 // <= 3.9
    const char *getPassName() const { return "AndCmp Cleanup Pass"; }
#else // LLVM 4.0+
    llvm::StringRef getPassName() const { return "AndCmp Cleanup Pass"; }
#endif
    bool runOnBasicBlock(llvm::BasicBlock &BB);

    static char ID;
};

char AndCmpCleanupPass::ID = 0;

// Look for ANDs of masks where one of the operands is a vector compare; we
// can turn these into specialized calls to masked vector compares and
// thence eliminate the AND.  For example, rather than emitting
// __and(__less(a, b), c), we will emit __less_and_mask(a, b, c).
bool AndCmpCleanupPass::runOnBasicBlock(llvm::BasicBlock &bb) {
    bool modifiedAny = false;

restart:
    for (llvm::BasicBlock::iterator iter = bb.begin(), e = bb.end(); iter != e; ++iter) {
        // See if we have an AND instruction
        llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(&*iter);
        if (bop == NULL || bop->getOpcode() != llvm::Instruction::And)
            continue;

        // Make sure it's a vector AND
        if (llvm::isa<llvm::VectorType>(bop->getType()) == false)
            continue;

        // We only care about ANDs of the mask type, not, e.g. ANDs of
        // int32s vectors.
        if (bop->getType() != LLVMTypes::MaskType)
            continue;

        // Now see if either of the operands to the AND is a comparison
        for (int i = 0; i < 2; ++i) {
            llvm::Value *op = bop->getOperand(i);
            llvm::CmpInst *opCmp = llvm::dyn_cast<llvm::CmpInst>(op);
            if (opCmp == NULL)
                continue;

            // We have a comparison.  However, we also need to make sure
            // that it's not comparing two mask values; those can't be
            // simplified to something simpler.
            if (opCmp->getOperand(0)->getType() == LLVMTypes::MaskType)
                break;

            // Success!  Go ahead and replace the AND with a call to the
            // "__and_mask" variant of the comparison function for this
            // operand.
            std::string funcName = lPredicateToString(opCmp->getPredicate());
            funcName += "_";
            funcName += lTypeToSuffix(opCmp->getOperand(0)->getType());
            funcName += "_and_mask";

            llvm::Function *andCmpFunc = m->module->getFunction(funcName);
            if (andCmpFunc == NULL) {
                // Declare the function if needed; the first two arguments
                // are the same as the two arguments to the compare we're
                // replacing and the third argument is the mask type.
                llvm::Type *cmpOpType = opCmp->getOperand(0)->getType();
#if ISPC_LLVM_VERSION <= ISPC_LLVM_8_0
                llvm::Constant *acf =
#if ISPC_LLVM_VERSION <= ISPC_LLVM_4_0
                    m->module->getOrInsertFunction(funcName, LLVMTypes::MaskType, cmpOpType, cmpOpType,
                                                   LLVMTypes::MaskType, NULL);
#else // LLVM 5.0+
                    m->module->getOrInsertFunction(funcName, LLVMTypes::MaskType, cmpOpType, cmpOpType,
                                                   LLVMTypes::MaskType);
#endif
                andCmpFunc = llvm::dyn_cast<llvm::Function>(acf);
#else
                llvm::FunctionCallee acf = m->module->getOrInsertFunction(funcName, LLVMTypes::MaskType, cmpOpType,
                                                                          cmpOpType, LLVMTypes::MaskType);
                andCmpFunc = llvm::dyn_cast<llvm::Function>(acf.getCallee());
#endif
                Assert(andCmpFunc != NULL);
                andCmpFunc->setDoesNotThrow();
                andCmpFunc->setDoesNotAccessMemory();
            }

            // Set up the function call to the *_and_mask function; the
            // mask value passed in is the other operand to the AND.
            llvm::Value *args[3] = {opCmp->getOperand(0), opCmp->getOperand(1), bop->getOperand(i ^ 1)};
            llvm::ArrayRef<llvm::Value *> argArray(&args[0], &args[3]);
            llvm::Instruction *cmpCall =
                llvm::CallInst::Create(andCmpFunc, argArray, LLVMGetName(bop, "_and_mask"), (llvm::Instruction *)NULL);

            // And replace the original AND instruction with it.
            llvm::ReplaceInstWithInst(&*iter, cmpCall);

            modifiedAny = true;
            goto restart;
        }
    }

    return modifiedAny;
}

///////////////////////////////////////////////////////////////////////////
// MaskOpsCleanupPass

/** This pass does various peephole improvements to mask modification
    operations.  In particular, it converts mask XORs with "all true" to
    calls to __not() and replaces operations like and(not(a), b) to
    __and_not1(a, b) (and similarly if the second operand has not applied
    to it...)
 */
class MaskOpsCleanupPass : public llvm::BasicBlockPass {
  public:
    MaskOpsCleanupPass(llvm::Module *m) : BasicBlockPass(ID) {
        llvm::Type *mt = LLVMTypes::MaskType;

        // Declare the __not, __and_not1, and __and_not2 functions that we
        // expect the target to end up providing.
        notFunc =
#if ISPC_LLVM_VERSION <= ISPC_LLVM_4_0
            llvm::dyn_cast<llvm::Function>(m->getOrInsertFunction("__not", mt, mt, NULL));
#elif ISPC_LLVM_VERSION <= ISPC_LLVM_8_0 // LLVM 5.0-LLVM 8.0
            llvm::dyn_cast<llvm::Function>(m->getOrInsertFunction("__not", mt, mt));
#else                                    // LLVM 9.0+
            llvm::dyn_cast<llvm::Function>(m->getOrInsertFunction("__not", mt, mt).getCallee());
#endif
        assert(notFunc != NULL);
#if ISPC_LLVM_VERSION == ISPC_LLVM_3_2
        notFunc->addFnAttr(llvm::Attributes::NoUnwind);
        notFunc->addFnAttr(llvm::Attributes::ReadNone);
#else /* LLVM 3.3+ */
        notFunc->addFnAttr(llvm::Attribute::NoUnwind);
        notFunc->addFnAttr(llvm::Attribute::ReadNone);
#endif

        andNotFuncs[0] =
#if ISPC_LLVM_VERSION <= ISPC_LLVM_4_0
            llvm::dyn_cast<llvm::Function>(m->getOrInsertFunction("__and_not1", mt, mt, mt, NULL));
#elif ISPC_LLVM_VERSION <= ISPC_LLVM_8_0 // LLVM 5.0-LLVM 8.0
            llvm::dyn_cast<llvm::Function>(m->getOrInsertFunction("__and_not1", mt, mt, mt));
#else                                    // LLVM 9.0+
            llvm::dyn_cast<llvm::Function>(m->getOrInsertFunction("__and_not1", mt, mt, mt).getCallee());
#endif
        assert(andNotFuncs[0] != NULL);
#if ISPC_LLVM_VERSION == ISPC_LLVM_3_2
        andNotFuncs[0]->addFnAttr(llvm::Attributes::NoUnwind);
        andNotFuncs[0]->addFnAttr(llvm::Attributes::ReadNone);
#else /* LLVM 3.3+ */
        andNotFuncs[0]->addFnAttr(llvm::Attribute::NoUnwind);
        andNotFuncs[0]->addFnAttr(llvm::Attribute::ReadNone);
#endif
        andNotFuncs[1] =
#if ISPC_LLVM_VERSION <= ISPC_LLVM_4_0
            llvm::dyn_cast<llvm::Function>(m->getOrInsertFunction("__and_not2", mt, mt, mt, NULL));
#elif ISPC_LLVM_VERSION <= ISPC_LLVM_8_0 // LLVM 5.0-LLVM 8.0
            llvm::dyn_cast<llvm::Function>(m->getOrInsertFunction("__and_not2", mt, mt, mt));
#else                                    // LLVM 9.0+
            llvm::dyn_cast<llvm::Function>(m->getOrInsertFunction("__and_not2", mt, mt, mt).getCallee());
#endif
        assert(andNotFuncs[1] != NULL);
#if ISPC_LLVM_VERSION == ISPC_LLVM_3_2
        andNotFuncs[1]->addFnAttr(llvm::Attributes::NoUnwind);
        andNotFuncs[1]->addFnAttr(llvm::Attributes::ReadNone);
#else /* LLVM 3.3+ */
        andNotFuncs[1]->addFnAttr(llvm::Attribute::NoUnwind);
        andNotFuncs[1]->addFnAttr(llvm::Attribute::ReadNone);
#endif
    }

#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_9 // <= 3.9
    const char *getPassName() const { return "MaskOps Cleanup Pass"; }
#else // LLVM 4.0+
    llvm::StringRef getPassName() const { return "MaskOps Cleanup Pass"; }
#endif
    bool runOnBasicBlock(llvm::BasicBlock &BB);

  private:
    llvm::Value *lGetNotOperand(llvm::Value *v) const;

    llvm::Function *notFunc, *andNotFuncs[2];

    static char ID;
};

char MaskOpsCleanupPass::ID = 0;

/** Returns true if the given value is a compile-time constant vector of
    i1s with all elements 'true'.
*/
static bool lIsAllTrue(llvm::Value *v) {
    if (llvm::ConstantVector *cv = llvm::dyn_cast<llvm::ConstantVector>(v)) {
        llvm::ConstantInt *ci;
        return (cv->getSplatValue() != NULL && (ci = llvm::dyn_cast<llvm::ConstantInt>(cv->getSplatValue())) != NULL &&
                ci->isOne());
    }

    if (llvm::ConstantDataVector *cdv = llvm::dyn_cast<llvm::ConstantDataVector>(v)) {
        llvm::ConstantInt *ci;
        return (cdv->getSplatValue() != NULL &&
                (ci = llvm::dyn_cast<llvm::ConstantInt>(cdv->getSplatValue())) != NULL && ci->isOne());
    }

    return false;
}

/** Checks to see if the given value is the NOT of some other value.  If
    so, it returns the operand of the NOT; otherwise returns NULL.
 */
llvm::Value *MaskOpsCleanupPass::lGetNotOperand(llvm::Value *v) const {
    if (llvm::CallInst *ci = llvm::dyn_cast<llvm::CallInst>(v))
        if (ci->getCalledFunction() == notFunc)
            // Direct call to __not()
            return ci->getArgOperand(0);

    if (llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(v))
        if (bop->getOpcode() == llvm::Instruction::Xor && lIsAllTrue(bop->getOperand(1)))
            // XOR of all-true vector.
            return bop->getOperand(0);

    return NULL;
}

bool MaskOpsCleanupPass::runOnBasicBlock(llvm::BasicBlock &bb) {
    bool modifiedAny = false;

restart:
    for (llvm::BasicBlock::iterator iter = bb.begin(), e = bb.end(); iter != e; ++iter) {
        llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(&*iter);
        if (bop == NULL)
            continue;

        if (bop->getType() != LLVMTypes::MaskType)
            continue;

        if (bop->getOpcode() == llvm::Instruction::Xor) {
            // Check for XOR with all-true values
            if (lIsAllTrue(bop->getOperand(1))) {
                llvm::Value *val = bop->getOperand(0);
                // Note that ArrayRef takes reference to an object, which must live
                // long enough, so passing return value of getOperand directly is
                // incorrect and it actually causes crashes with gcc 4.7 and later.
                llvm::ArrayRef<llvm::Value *> arg(val);
                llvm::CallInst *notCall = llvm::CallInst::Create(notFunc, arg, bop->getName());
                ReplaceInstWithInst(&*iter, notCall);
                modifiedAny = true;
                goto restart;
            }
        } else if (bop->getOpcode() == llvm::Instruction::And) {
            // Check each of the operands to see if they have NOT applied
            // to them.
            for (int i = 0; i < 2; ++i) {
                if (llvm::Value *notOp = lGetNotOperand(bop->getOperand(i))) {
                    // In notOp we have the target of the NOT operation;
                    // put it in its appropriate spot in the operand array.
                    // Copy in the other operand directly.
                    llvm::Value *args[2];
                    args[i] = notOp;
                    args[i ^ 1] = bop->getOperand(i ^ 1);
                    llvm::ArrayRef<llvm::Value *> argsRef(&args[0], 2);

                    // Call the appropriate __and_not* function.
                    llvm::CallInst *andNotCall = llvm::CallInst::Create(andNotFuncs[i], argsRef, bop->getName());

                    ReplaceInstWithInst(&*iter, andNotCall);
                    modifiedAny = true;
                    goto restart;
                }
            }
        }
    }

    return modifiedAny;
}

//===----------------------------------------------------------------------===//
//                       External Interface declaration
//===----------------------------------------------------------------------===//

bool WriteCXXFile(llvm::Module *module, const char *fn, int vectorWidth, const char *includeName) {

#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_6 // 3.2, 3.3, 3.4, 3.5, 3.6
    llvm::PassManager pm;
#else // LLVM 3.7+
    llvm::legacy::PassManager pm;
#endif
#if 0
    if (const llvm::TargetData *td = targetMachine->getTargetData())
        pm.add(new llvm::TargetData(*td));
    else
        pm.add(new llvm::TargetData(module));
#endif

#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_3 // 3.2, 3.3
    int flags = 0;
#else // LLVM 3.4+
    llvm::sys::fs::OpenFlags flags = llvm::sys::fs::F_None;
#endif

#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_5 // 3.2, 3.3, 3.4, 3.5
    std::string error;
#else // LLVM 3.6+
    std::error_code error;
#endif

#if ISPC_LLVM_VERSION <= ISPC_LLVM_5_0
    llvm::tool_output_file *of = new llvm::tool_output_file(fn, error, flags);
#else // LLVM 6.0+
    llvm::ToolOutputFile *of = new llvm::ToolOutputFile(fn, error, flags);
#endif

#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_5 // 3.2, 3.3, 3.4, 3.5
    if (error.size()) {
#else // LLVM 3.6+
    if (error) {
#endif
        fprintf(stderr, "Error opening output file \"%s\".\n", fn);
        return false;
    }

    llvm::formatted_raw_ostream fos(of->os());

    pm.add(llvm::createGCLoweringPass());
    pm.add(llvm::createLowerInvokePass());
    pm.add(llvm::createCFGSimplificationPass()); // clean up after lower invoke.
    pm.add(new SmearCleanupPass(module, vectorWidth));
    pm.add(new AndCmpCleanupPass());
    pm.add(new MaskOpsCleanupPass(module));
    pm.add(llvm::createDeadCodeEliminationPass()); // clean up after smear pass
                                                   // CO    pm.add(llvm::createPrintModulePass(&fos));
    pm.add(new CWriter(fos, includeName, vectorWidth));
#if ISPC_LLVM_VERSION == ISPC_LLVM_3_2
    // This interface is depricated for 3.3+
    pm.add(llvm::createGCInfoDeleter());
#endif
    // CO    pm.add(llvm::createVerifierPass());

    pm.run(*module);

    return true;
}
