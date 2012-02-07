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

/** @file opt.cpp
    @brief Implementations of various ispc optimization passes that operate
           on the LLVM IR.
*/

#include "opt.h"
#include "ctx.h"
#include "sym.h"
#include "module.h"
#include "util.h"
#include "llvmutil.h"

#include <stdio.h>
#include <map>
#include <set>

#include <llvm/Pass.h>
#include <llvm/Module.h>
#include <llvm/PassManager.h>
#include <llvm/PassRegistry.h>
#include <llvm/Assembly/PrintModulePass.h>
#include <llvm/Function.h>
#include <llvm/BasicBlock.h>
#include <llvm/Instructions.h>
#include <llvm/Intrinsics.h>
#include <llvm/Constants.h>
#include <llvm/Analysis/ConstantFolding.h>
#include <llvm/Target/TargetLibraryInfo.h>
#ifdef LLVM_2_9
    #include <llvm/Support/StandardPasses.h>
#endif // LLVM_2_9
#include <llvm/ADT/Triple.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Target/TargetData.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Analysis/Verifier.h>
#include <llvm/Analysis/Passes.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Analysis/DIBuilder.h>
#include <llvm/Analysis/DebugInfo.h>
#include <llvm/Support/Dwarf.h>
#ifdef ISPC_IS_LINUX
  #include <alloca.h>
#elif defined(ISPC_IS_WINDOWS)
  #include <malloc.h>
  #ifndef __MINGW32__
    #define alloca _alloca
  #endif
#endif // ISPC_IS_WINDOWS

static llvm::Pass *CreateIntrinsicsOptPass();
static llvm::Pass *CreateVSelMovmskOptPass();
static llvm::Pass *CreateDetectGSBaseOffsetsPass();
static llvm::Pass *CreateGSToLoadStorePass();
static llvm::Pass *CreatePseudoGSToGSPass();
static llvm::Pass *CreatePseudoMaskedStorePass();
static llvm::Pass *CreateMaskedStoreOptPass();
static llvm::Pass *CreateMaskedLoadOptPass();
static llvm::Pass *CreateIsCompileTimeConstantPass(bool isLastTry);
static llvm::Pass *CreateMakeInternalFuncsStaticPass();

///////////////////////////////////////////////////////////////////////////


/** This utility routine copies the metadata (if any) attached to the
    'from' instruction in the IR to the 'to' instruction.  

    For flexibility, this function takes an llvm::Value rather than an
    llvm::Instruction for the 'to' parameter; at some places in the code
    below, we sometimes use a llvm::Value to start out storing a value and
    then later store instructions.  If a llvm::Value is passed to this, the
    routine just returns without doing anything; if it is in fact an
    LLVM::Instruction, then the metadata can be copied to it.
 */
static void
lCopyMetadata(llvm::Value *vto, const llvm::Instruction *from) {
    llvm::Instruction *to = llvm::dyn_cast<llvm::Instruction>(vto);
    if (!to) 
        return;
                                                              
    llvm::SmallVector<std::pair<unsigned int, llvm::MDNode *>, 8> metadata;
    from->getAllMetadata(metadata);
    for (unsigned int i = 0; i < metadata.size(); ++i)
        to->setMetadata(metadata[i].first, metadata[i].second);
}


/** We have a protocol with the front-end LLVM IR code generation process
    that allows us to encode the source file position that corresponds with
    instructions.  (For example, this allows us to issue performance
    warnings related to things like scatter and gather after optimization
    has been performed, so that we aren't warning about scatters and
    gathers that have been improved to stores and loads by optimization
    passes.)  Note that this is slightly redundant with the source file
    position encoding generated for debugging symbols, though we don't
    always generate debugging information but we do always generate this
    position data.

    This function finds the SourcePos that the metadata in the instruction
    (if present) corresponds to.  See the implementation of
    FunctionEmitContext::addGSMetadata(), which encodes the source position during
    code generation.

    @param inst   Instruction to try to find the source position of
    @param pos    Output variable in which to store the position
    @returns      True if source file position metadata was present and *pos
                  has been set.  False otherwise.
*/
static bool
lGetSourcePosFromMetadata(const llvm::Instruction *inst, SourcePos *pos) {
    llvm::MDNode *filename = inst->getMetadata("filename");
    llvm::MDNode *first_line = inst->getMetadata("first_line");
    llvm::MDNode *first_column = inst->getMetadata("first_column");
    llvm::MDNode *last_line = inst->getMetadata("last_line");
    llvm::MDNode *last_column = inst->getMetadata("last_column");
    if (!filename || !first_line || !first_column || !last_line || !last_column)
        return false;

    // All of these asserts are things that FunctionEmitContext::addGSMetadata() is
    // expected to have done in its operation
    llvm::MDString *str = llvm::dyn_cast<llvm::MDString>(filename->getOperand(0));
    Assert(str);
    llvm::ConstantInt *first_lnum = 
        llvm::dyn_cast<llvm::ConstantInt>(first_line->getOperand(0));
    Assert(first_lnum);
    llvm::ConstantInt *first_colnum = 
        llvm::dyn_cast<llvm::ConstantInt>(first_column->getOperand(0));
    Assert(first_column);
    llvm::ConstantInt *last_lnum = 
        llvm::dyn_cast<llvm::ConstantInt>(last_line->getOperand(0));
    Assert(last_lnum);
    llvm::ConstantInt *last_colnum = 
        llvm::dyn_cast<llvm::ConstantInt>(last_column->getOperand(0));
    Assert(last_column);

    *pos = SourcePos(str->getString().data(), (int)first_lnum->getZExtValue(),
                     (int)first_colnum->getZExtValue(), (int)last_lnum->getZExtValue(),
                     (int)last_colnum->getZExtValue());
    return true;
}


static llvm::Instruction *
lCallInst(llvm::Function *func, llvm::Value *arg0, llvm::Value *arg1, 
          const char *name, llvm::Instruction *insertBefore = NULL) {
    llvm::Value *args[2] = { arg0, arg1 };
#if defined(LLVM_3_0) || defined(LLVM_3_0svn) || defined(LLVM_3_1svn)
    llvm::ArrayRef<llvm::Value *> newArgArray(&args[0], &args[2]);
    return llvm::CallInst::Create(func, newArgArray, name, insertBefore);
#else
    return llvm::CallInst::Create(func, &args[0], &args[2],
                                  name, insertBefore);
#endif
}


static llvm::Instruction *
lCallInst(llvm::Function *func, llvm::Value *arg0, llvm::Value *arg1, 
          llvm::Value *arg2, const char *name,
          llvm::Instruction *insertBefore = NULL) {
    llvm::Value *args[3] = { arg0, arg1, arg2 };
#if defined(LLVM_3_0) || defined(LLVM_3_0svn) || defined(LLVM_3_1svn)
    llvm::ArrayRef<llvm::Value *> newArgArray(&args[0], &args[3]);
    return llvm::CallInst::Create(func, newArgArray, name, insertBefore);
#else
    return llvm::CallInst::Create(func, &args[0], &args[3],
                                  name, insertBefore);
#endif
}


#if 0
static llvm::Instruction *
lCallInst(llvm::Function *func, llvm::Value *arg0, llvm::Value *arg1, 
          llvm::Value *arg2, llvm::Value *arg3, const char *name,
          llvm::Instruction *insertBefore = NULL) {
    llvm::Value *args[4] = { arg0, arg1, arg2, arg3 };
#if defined(LLVM_3_0) || defined(LLVM_3_0svn) || defined(LLVM_3_1svn)
    llvm::ArrayRef<llvm::Value *> newArgArray(&args[0], &args[4]);
    return llvm::CallInst::Create(func, newArgArray, name, insertBefore);
#else
    return llvm::CallInst::Create(func, &args[0], &args[4],
                                  name, insertBefore);
#endif
}
#endif

static llvm::Instruction *
lCallInst(llvm::Function *func, llvm::Value *arg0, llvm::Value *arg1, 
          llvm::Value *arg2, llvm::Value *arg3, llvm::Value *arg4,
          const char *name, llvm::Instruction *insertBefore = NULL) {
    llvm::Value *args[5] = { arg0, arg1, arg2, arg3, arg4 };
#if defined(LLVM_3_0) || defined(LLVM_3_0svn) || defined(LLVM_3_1svn)
    llvm::ArrayRef<llvm::Value *> newArgArray(&args[0], &args[5]);
    return llvm::CallInst::Create(func, newArgArray, name, insertBefore);
#else
    return llvm::CallInst::Create(func, &args[0], &args[5],
                                  name, insertBefore);
#endif
}

static llvm::Instruction *
lCallInst(llvm::Function *func, llvm::Value *arg0, llvm::Value *arg1, 
          llvm::Value *arg2, llvm::Value *arg3, llvm::Value *arg4,
          llvm::Value *arg5, const char *name, 
          llvm::Instruction *insertBefore = NULL) {
    llvm::Value *args[6] = { arg0, arg1, arg2, arg3, arg4, arg5 };
#if defined(LLVM_3_0) || defined(LLVM_3_0svn) || defined(LLVM_3_1svn)
    llvm::ArrayRef<llvm::Value *> newArgArray(&args[0], &args[6]);
    return llvm::CallInst::Create(func, newArgArray, name, insertBefore);
#else
    return llvm::CallInst::Create(func, &args[0], &args[6],
                                  name, insertBefore);
#endif
}

///////////////////////////////////////////////////////////////////////////

void
Optimize(llvm::Module *module, int optLevel) {
    if (g->debugPrint) {
        printf("*** Code going into optimization ***\n");
        module->dump();
    }

    llvm::PassManager optPM;
    llvm::FunctionPassManager funcPM(module);

    if (g->target.isa != Target::GENERIC) {
        llvm::TargetLibraryInfo *targetLibraryInfo =
            new llvm::TargetLibraryInfo(llvm::Triple(module->getTargetTriple()));
        optPM.add(targetLibraryInfo);
        optPM.add(new llvm::TargetData(module));
    }

#if defined(LLVM_3_0) || defined(LLVM_3_0svn) || defined(LLVM_3_1svn)
    optPM.add(llvm::createIndVarSimplifyPass());
#endif

    if (optLevel == 0) {
        // This is more or less the minimum set of optimizations that we
        // need to do to generate code that will actually run.  (We can't
        // run absolutely no optimizations, since the front-end needs us to
        // take the various __pseudo_* functions it has emitted and turn
        // them into something that can actually execute.
        optPM.add(llvm::createPromoteMemoryToRegisterPass());
        optPM.add(CreateDetectGSBaseOffsetsPass());
        if (g->opt.disableHandlePseudoMemoryOps == false) {
            optPM.add(CreatePseudoGSToGSPass());
            optPM.add(CreatePseudoMaskedStorePass());
        }
        optPM.add(CreateIsCompileTimeConstantPass(true));
        optPM.add(llvm::createFunctionInliningPass());
        optPM.add(CreateMakeInternalFuncsStaticPass());
        optPM.add(llvm::createCFGSimplificationPass());
        optPM.add(llvm::createGlobalDCEPass());
    }
    else {
        // Otherwise throw the kitchen sink of optimizations at the code.
        // This is almost certainly overkill and likely could be reduced,
        // but on the other hand trying to remove some of these has
        // historically caused performance slowdowns.  Benchmark carefully
        // if changing these around.
        //
        // Note in particular that a number of the ispc optimization
        // passes are run repeatedly along the way; they often can kick in
        // only later in the optimization process as things like constant
        // propagation have done their thing, and then when they do kick
        // in, they can often open up new opportunities for optimization...
        llvm::PassRegistry *registry = llvm::PassRegistry::getPassRegistry();
        llvm::initializeCore(*registry);
        llvm::initializeScalarOpts(*registry);
        llvm::initializeIPO(*registry);
        llvm::initializeAnalysis(*registry);
        llvm::initializeIPA(*registry);
        llvm::initializeTransformUtils(*registry);
        llvm::initializeInstCombine(*registry);
        llvm::initializeInstrumentation(*registry);
        llvm::initializeTarget(*registry);

        bool runSROA = true;

        // Early optimizations to try to reduce the total amount of code to
        // work with if we can
        optPM.add(llvm::createReassociatePass());
        optPM.add(llvm::createConstantPropagationPass());
        optPM.add(llvm::createConstantPropagationPass());
        optPM.add(llvm::createDeadInstEliminationPass());
        optPM.add(llvm::createCFGSimplificationPass());

        optPM.add(CreateDetectGSBaseOffsetsPass());
        if (!g->opt.disableMaskAllOnOptimizations) {
            optPM.add(CreateIntrinsicsOptPass());
            optPM.add(CreateVSelMovmskOptPass());
            optPM.add(CreateMaskedStoreOptPass());
            optPM.add(CreateMaskedLoadOptPass());
        }
        optPM.add(llvm::createDeadInstEliminationPass());

        // On to more serious optimizations
        if (runSROA)
            optPM.add(llvm::createScalarReplAggregatesPass());
        optPM.add(llvm::createInstructionCombiningPass());
        optPM.add(llvm::createCFGSimplificationPass());
        optPM.add(llvm::createPromoteMemoryToRegisterPass());
        optPM.add(llvm::createGlobalOptimizerPass());
        optPM.add(llvm::createReassociatePass());
        optPM.add(llvm::createIPConstantPropagationPass());
        optPM.add(llvm::createDeadArgEliminationPass());
        optPM.add(llvm::createInstructionCombiningPass());
        optPM.add(llvm::createCFGSimplificationPass());
        optPM.add(llvm::createPruneEHPass());
        optPM.add(llvm::createFunctionAttrsPass());
        optPM.add(llvm::createFunctionInliningPass());
        optPM.add(llvm::createConstantPropagationPass());
        optPM.add(llvm::createDeadInstEliminationPass());
        optPM.add(llvm::createCFGSimplificationPass());

        optPM.add(llvm::createArgumentPromotionPass());
        optPM.add(llvm::createSimplifyLibCallsPass());
        optPM.add(llvm::createInstructionCombiningPass());
        optPM.add(llvm::createJumpThreadingPass());
        optPM.add(llvm::createCFGSimplificationPass());
        if (runSROA)
            optPM.add(llvm::createScalarReplAggregatesPass());
        optPM.add(llvm::createInstructionCombiningPass());
        optPM.add(llvm::createTailCallEliminationPass());

        if (!g->opt.disableMaskAllOnOptimizations) {
            optPM.add(CreateIntrinsicsOptPass());
            optPM.add(CreateVSelMovmskOptPass());
            optPM.add(CreateMaskedStoreOptPass());
            optPM.add(CreateMaskedLoadOptPass());
        }
        if (g->opt.disableHandlePseudoMemoryOps == false)
            optPM.add(CreatePseudoMaskedStorePass());
        if (g->opt.disableGatherScatterOptimizations == false &&
            g->opt.disableHandlePseudoMemoryOps == false)
            optPM.add(CreateGSToLoadStorePass());
        if (g->opt.disableHandlePseudoMemoryOps == false) {
            optPM.add(CreatePseudoMaskedStorePass());
            optPM.add(CreatePseudoGSToGSPass());
        }
        if (!g->opt.disableMaskAllOnOptimizations) {
            optPM.add(CreateMaskedStoreOptPass());
            optPM.add(CreateMaskedLoadOptPass());
        }
        optPM.add(llvm::createFunctionInliningPass());
        optPM.add(llvm::createConstantPropagationPass());
        optPM.add(CreateIntrinsicsOptPass());
        optPM.add(CreateVSelMovmskOptPass());

#if defined(LLVM_2_9)
        llvm::createStandardModulePasses(&optPM, 3, 
                                         false /* opt size */,
                                         true /* unit at a time */, 
                                         g->opt.unrollLoops,
                                         true /* simplify lib calls */,
                                         false /* may have exceptions */,
                                         llvm::createFunctionInliningPass());
        llvm::createStandardLTOPasses(&optPM, true /* internalize pass */,
                                      true /* inline once again */,
                                      false /* verify after each pass */);
        llvm::createStandardFunctionPasses(&optPM, 3);

        optPM.add(CreateIsCompileTimeConstantPass(true));
        optPM.add(CreateIntrinsicsOptPass());
        optPM.add(CreateVSelMovmskOptPass());

        llvm::createStandardModulePasses(&optPM, 3, 
                                         false /* opt size */,
                                         true /* unit at a time */, 
                                         g->opt.unrollLoops,
                                         true /* simplify lib calls */,
                                         false /* may have exceptions */,
                                         llvm::createFunctionInliningPass());

#else
        funcPM.add(llvm::createTypeBasedAliasAnalysisPass());
        funcPM.add(llvm::createBasicAliasAnalysisPass());
        funcPM.add(llvm::createCFGSimplificationPass());
        if (runSROA)
            funcPM.add(llvm::createScalarReplAggregatesPass());
        funcPM.add(llvm::createEarlyCSEPass());
        funcPM.add(llvm::createLowerExpectIntrinsicPass());

        optPM.add(llvm::createTypeBasedAliasAnalysisPass());
        optPM.add(llvm::createBasicAliasAnalysisPass());
        optPM.add(llvm::createGlobalOptimizerPass());     
        optPM.add(llvm::createIPSCCPPass());              
        optPM.add(llvm::createDeadArgEliminationPass());  
        optPM.add(llvm::createInstructionCombiningPass());
        optPM.add(llvm::createCFGSimplificationPass());   
        optPM.add(llvm::createFunctionInliningPass());
        optPM.add(llvm::createArgumentPromotionPass());   
        if (runSROA)
            optPM.add(llvm::createScalarReplAggregatesPass(-1, false));
        optPM.add(llvm::createInstructionCombiningPass());  
        optPM.add(llvm::createCFGSimplificationPass());     
        optPM.add(llvm::createReassociatePass());           
        optPM.add(llvm::createLoopRotatePass());            
        optPM.add(llvm::createLICMPass());                  
        optPM.add(llvm::createLoopUnswitchPass(false));
        optPM.add(llvm::createInstructionCombiningPass());
        optPM.add(llvm::createIndVarSimplifyPass());        
        optPM.add(llvm::createLoopIdiomPass());             
        optPM.add(llvm::createLoopDeletionPass());          
        if (g->opt.unrollLoops)
            optPM.add(llvm::createLoopUnrollPass());          
        optPM.add(llvm::createGVNPass());                 
        optPM.add(llvm::createMemCpyOptPass());             
        optPM.add(llvm::createSCCPPass());                  
        optPM.add(llvm::createInstructionCombiningPass());
        optPM.add(llvm::createJumpThreadingPass());         
        optPM.add(llvm::createCorrelatedValuePropagationPass());
        optPM.add(llvm::createDeadStoreEliminationPass());  
        optPM.add(llvm::createAggressiveDCEPass());         
        optPM.add(llvm::createCFGSimplificationPass());     
        optPM.add(llvm::createInstructionCombiningPass());  
        optPM.add(llvm::createStripDeadPrototypesPass()); 
        optPM.add(llvm::createGlobalDCEPass());         
        optPM.add(llvm::createConstantMergePass());     

        optPM.add(CreateIsCompileTimeConstantPass(false));
        optPM.add(CreateIntrinsicsOptPass());
        optPM.add(CreateVSelMovmskOptPass());

        optPM.add(llvm::createGlobalOptimizerPass());
        optPM.add(llvm::createGlobalDCEPass()); 
        optPM.add(llvm::createArgumentPromotionPass());
        optPM.add(llvm::createInstructionCombiningPass());
        optPM.add(llvm::createJumpThreadingPass());
        if (runSROA)
            optPM.add(llvm::createScalarReplAggregatesPass());
        optPM.add(llvm::createFunctionAttrsPass()); 
        optPM.add(llvm::createGlobalsModRefPass()); 
        optPM.add(llvm::createLICMPass());      
        optPM.add(llvm::createGVNPass());       
        optPM.add(llvm::createMemCpyOptPass()); 
        optPM.add(llvm::createDeadStoreEliminationPass());
        optPM.add(llvm::createInstructionCombiningPass());
        optPM.add(llvm::createJumpThreadingPass());
        optPM.add(llvm::createCFGSimplificationPass());
        optPM.add(llvm::createGlobalDCEPass());

        optPM.add(CreateIsCompileTimeConstantPass(true));
        optPM.add(CreateIntrinsicsOptPass());
        optPM.add(CreateVSelMovmskOptPass());
            
        optPM.add(llvm::createArgumentPromotionPass());   
        if (runSROA)
            optPM.add(llvm::createScalarReplAggregatesPass(-1, false));
        optPM.add(llvm::createEarlyCSEPass());              
        optPM.add(llvm::createSimplifyLibCallsPass());    
        optPM.add(llvm::createJumpThreadingPass());         
        optPM.add(llvm::createCorrelatedValuePropagationPass()); 
        optPM.add(llvm::createCFGSimplificationPass());     
        optPM.add(llvm::createInstructionCombiningPass());  
        optPM.add(llvm::createCFGSimplificationPass());     
        optPM.add(llvm::createReassociatePass());           
        optPM.add(llvm::createGVNPass());                 
        optPM.add(llvm::createMemCpyOptPass());             
        optPM.add(llvm::createSCCPPass());                  
        optPM.add(llvm::createInstructionCombiningPass());
        optPM.add(llvm::createJumpThreadingPass());         
        optPM.add(llvm::createCorrelatedValuePropagationPass());
        optPM.add(llvm::createDeadStoreEliminationPass());  
        optPM.add(llvm::createAggressiveDCEPass());         
        optPM.add(llvm::createCFGSimplificationPass());     
        optPM.add(llvm::createInstructionCombiningPass());  
        optPM.add(llvm::createStripDeadPrototypesPass()); 
        optPM.add(llvm::createGlobalDCEPass());         
        optPM.add(llvm::createConstantMergePass());     
#endif
        optPM.add(CreateMakeInternalFuncsStaticPass());
        optPM.add(llvm::createGlobalDCEPass());
    }

    // Finish up by making sure we didn't mess anything up in the IR along
    // the way.
    optPM.add(llvm::createVerifierPass());
    
    for (llvm::Module::iterator fiter = module->begin(); fiter != module->end();
         ++fiter)
        funcPM.run(*fiter);

    optPM.run(*module);

    if (g->debugPrint) {
        printf("\n*****\nFINAL OUTPUT\n*****\n");
        module->dump();
    }
}



///////////////////////////////////////////////////////////////////////////
// IntrinsicsOpt

/** This is a relatively simple optimization pass that does a few small
    optimizations that LLVM's x86 optimizer doesn't currently handle.
    (Specifically, MOVMSK of a constant can be replaced with the
    corresponding constant value, BLENDVPS and AVX masked load/store with
    either an 'all on' or 'all off' masks can be replaced with simpler
    operations.

    @todo The better thing to do would be to submit a patch to LLVM to get
    these; they're presumably pretty simple patterns to match.  
*/
class IntrinsicsOpt : public llvm::BasicBlockPass {
public:
    IntrinsicsOpt();

    const char *getPassName() const { return "Intrinsics Cleanup Optimization"; }
    bool runOnBasicBlock(llvm::BasicBlock &BB);

    static char ID;

private:
    struct MaskInstruction {
        MaskInstruction(llvm::Function *f) { function = f; }
        llvm::Function *function;
    };
    std::vector<MaskInstruction> maskInstructions;

    /** Structure that records everything we need to know about a blend
        instruction for this optimization pass.
     */
    struct BlendInstruction {
        BlendInstruction(llvm::Function *f, int ao, int o0, int o1, int of)
            : function(f), allOnMask(ao), op0(o0), op1(o1), opFactor(of) { }
        /** Function pointer for the blend instruction */ 
        llvm::Function *function;
        /** Mask value for an "all on" mask for this instruction */
        int allOnMask;
        /** The operand number in the llvm CallInst corresponds to the
            first operand to blend with. */
        int op0;
        /** The operand number in the CallInst corresponding to the second
            operand to blend with. */
        int op1;
        /** The operand in the call inst where the blending factor is
            found. */
        int opFactor;
    };
    std::vector<BlendInstruction> blendInstructions;

    bool matchesMaskInstruction(llvm::Function *function);
    BlendInstruction *matchingBlendInstruction(llvm::Function *function);
};

char IntrinsicsOpt::ID = 0;


IntrinsicsOpt::IntrinsicsOpt() 
    : BasicBlockPass(ID) {

    // All of the mask instructions we may encounter.  Note that even if
    // compiling for AVX, we may still encounter the regular 4-wide SSE
    // MOVMSK instruction.
    llvm::Function *sseMovmsk = 
        llvm::Intrinsic::getDeclaration(m->module, llvm::Intrinsic::x86_sse_movmsk_ps);
    maskInstructions.push_back(sseMovmsk);
    maskInstructions.push_back(m->module->getFunction("__movmsk"));
#if defined(LLVM_3_0) || defined(LLVM_3_0svn) || defined(LLVM_3_1svn)
    llvm::Function *avxMovmsk = 
        llvm::Intrinsic::getDeclaration(m->module, llvm::Intrinsic::x86_avx_movmsk_ps_256);
    Assert(avxMovmsk != NULL);
    maskInstructions.push_back(avxMovmsk);
#endif

    // And all of the blend instructions
    blendInstructions.push_back(BlendInstruction(
        llvm::Intrinsic::getDeclaration(m->module, llvm::Intrinsic::x86_sse41_blendvps),
        0xf, 0, 1, 2));
#if defined(LLVM_3_0) || defined(LLVM_3_0svn) || defined(LLVM_3_1svn)
    blendInstructions.push_back(BlendInstruction(
        llvm::Intrinsic::getDeclaration(m->module, llvm::Intrinsic::x86_avx_blendv_ps_256),
        0xff, 0, 1, 2));
#endif
}


/** Given an llvm::Value represinting a vector mask, see if the value is a
    constant.  If so, return the integer mask found by taking the high bits
    of the mask values in turn and concatenating them into a single integer.
    In other words, given the 4-wide mask: < 0xffffffff, 0, 0, 0xffffffff >, 
    we have 0b1001 = 9.
 */
static int
lGetMask(llvm::Value *factor) {
    /* FIXME: This will break if we ever do 32-wide compilation, in which case
       it don't be possible to distinguish between -1 for "don't know" and
       "known and all bits on". */
    Assert(g->target.vectorWidth < 32);

#ifdef LLVM_3_1svn
    llvm::ConstantDataVector *cv = llvm::dyn_cast<llvm::ConstantDataVector>(factor);
#else
    llvm::ConstantVector *cv = llvm::dyn_cast<llvm::ConstantVector>(factor);
#endif
    if (cv) {
        int mask = 0;
        llvm::SmallVector<llvm::Constant *, ISPC_MAX_NVEC> elements;
#ifdef LLVM_3_1svn
        for (int i = 0; i < (int)cv->getNumElements(); ++i)
            elements.push_back(cv->getElementAsConstant(i));
#else
        cv->getVectorElements(elements);
#endif

        for (unsigned int i = 0; i < elements.size(); ++i) {
            llvm::APInt intMaskValue;
            // SSE has the "interesting" approach of encoding blending
            // masks as <n x float>.
            llvm::ConstantFP *cf = llvm::dyn_cast<llvm::ConstantFP>(elements[i]);
            if (cf) {
                llvm::APFloat apf = cf->getValueAPF();
                intMaskValue = apf.bitcastToAPInt();
            }
            else {
                // Otherwise get it as an int
                llvm::ConstantInt *ci = llvm::dyn_cast<llvm::ConstantInt>(elements[i]);
                Assert(ci != NULL);  // vs return -1 if NULL?
                intMaskValue = ci->getValue();
            }
            // Is the high-bit set?  If so, OR in the appropriate bit in
            // the result mask
            if (intMaskValue.countLeadingOnes() > 0)
                mask |= (1 << i);
        }
        return mask;
    }
    else if (llvm::isa<llvm::ConstantAggregateZero>(factor))
        return 0;
    else {
#if 0
        llvm::ConstantExpr *ce = llvm::dyn_cast<llvm::ConstantExpr>(factor);
        if (ce != NULL) {
            llvm::TargetMachine *targetMachine = g->target.GetTargetMachine();
            const llvm::TargetData *td = targetMachine->getTargetData();
            llvm::Constant *c = llvm::ConstantFoldConstantExpression(ce, td);
            c->dump();
            factor = c;
        }
        // else we should be able to handle it above...
        Assert(!llvm::isa<llvm::Constant>(factor));
#endif
        return -1;
    }
}


/** Given an llvm::Value, return true if we can determine that it's an
    undefined value.  This only makes a weak attempt at chasing this down,
    only detecting flat-out undef values, and bitcasts of undef values.

    @todo Is it worth working harder to find more of these?  It starts to
    get tricky, since having an undef operand doesn't necessarily mean that
    the result will be undefined.  (And for that matter, is there an LLVM
    call that will do this for us?)
 */
static bool
lIsUndef(llvm::Value *value) {
    if (llvm::isa<llvm::UndefValue>(value))
        return true;

    llvm::BitCastInst *bci = llvm::dyn_cast<llvm::BitCastInst>(value);
    if (bci)
        return lIsUndef(bci->getOperand(0));

    return false;
}


bool
IntrinsicsOpt::runOnBasicBlock(llvm::BasicBlock &bb) {
#if defined(LLVM_3_0) || defined(LLVM_3_0svn) || defined(LLVM_3_1svn)
    llvm::Function *avxMaskedLoad32 = 
        llvm::Intrinsic::getDeclaration(m->module, llvm::Intrinsic::x86_avx_maskload_ps_256);
    llvm::Function *avxMaskedLoad64 = 
        llvm::Intrinsic::getDeclaration(m->module, llvm::Intrinsic::x86_avx_maskload_pd_256);
    llvm::Function *avxMaskedStore32 = 
        llvm::Intrinsic::getDeclaration(m->module, llvm::Intrinsic::x86_avx_maskstore_ps_256);
    llvm::Function *avxMaskedStore64 = 
        llvm::Intrinsic::getDeclaration(m->module, llvm::Intrinsic::x86_avx_maskstore_pd_256);
    Assert(avxMaskedLoad32 != NULL && avxMaskedStore32 != NULL);
    Assert(avxMaskedLoad64 != NULL && avxMaskedStore64 != NULL);
#endif

    bool modifiedAny = false;
 restart:
    for (llvm::BasicBlock::iterator iter = bb.begin(), e = bb.end(); iter != e; ++iter) {
        llvm::CallInst *callInst = llvm::dyn_cast<llvm::CallInst>(&*iter);
        if (callInst == NULL || callInst->getCalledFunction() == NULL)
            continue;

        BlendInstruction *blend = matchingBlendInstruction(callInst->getCalledFunction());
        if (blend != NULL) {
            llvm::Value *v[2] = { callInst->getArgOperand(blend->op0), 
                                  callInst->getArgOperand(blend->op1) };
            llvm::Value *factor = callInst->getArgOperand(blend->opFactor);

            // If the values are the same, then no need to blend..
            if (v[0] == v[1]) {
                llvm::ReplaceInstWithValue(iter->getParent()->getInstList(), 
                                           iter, v[0]);
                modifiedAny = true;
                goto restart;
            }

            // If one of the two is undefined, we're allowed to replace
            // with the value of the other.  (In other words, the only
            // valid case is that the blend factor ends up having a value
            // that only selects from the defined one of the two operands,
            // otherwise the result is undefined and any value is fine,
            // ergo the defined one is an acceptable result.)
            if (lIsUndef(v[0])) {
                llvm::ReplaceInstWithValue(iter->getParent()->getInstList(), 
                                           iter, v[1]);
                modifiedAny = true;
                goto restart;
            }
            if (lIsUndef(v[1])) {
                llvm::ReplaceInstWithValue(iter->getParent()->getInstList(), 
                                           iter, v[0]);
                modifiedAny = true;
                goto restart;
            }

            int mask = lGetMask(factor);
            llvm::Value *value = NULL;
            if (mask == 0)
                // Mask all off -> replace with the first blend value
                value = v[0];
            else if (mask == blend->allOnMask)
                // Mask all on -> replace with the second blend value
                value = v[1];

            if (value != NULL) {
                llvm::ReplaceInstWithValue(iter->getParent()->getInstList(), 
                                           iter, value);
                modifiedAny = true;
                goto restart;
            }
        }
        else if (matchesMaskInstruction(callInst->getCalledFunction())) {
            llvm::Value *factor = callInst->getArgOperand(0);
            int mask = lGetMask(factor);
            if (mask != -1) {
                // If the vector-valued mask has a known value, replace it
                // with the corresponding integer mask from its elements
                // high bits.
                llvm::Value *value = LLVMInt32(mask);
                llvm::ReplaceInstWithValue(iter->getParent()->getInstList(),
                                           iter, value);
                modifiedAny = true;
                goto restart;
            }
        }
#if defined(LLVM_3_0) || defined(LLVM_3_0svn) || defined(LLVM_3_1svn)
        else if (callInst->getCalledFunction() == avxMaskedLoad32 ||
                 callInst->getCalledFunction() == avxMaskedLoad64) {
            llvm::Value *factor = callInst->getArgOperand(1);
            int mask = lGetMask(factor);
            if (mask == 0) {
                // nothing being loaded, replace with undef value
                llvm::Type *returnType = callInst->getType();
                Assert(llvm::isa<llvm::VectorType>(returnType));
                llvm::Value *undefValue = llvm::UndefValue::get(returnType);
                llvm::ReplaceInstWithValue(iter->getParent()->getInstList(),
                                           iter, undefValue);
                modifiedAny = true;
                goto restart;
            }
            else if (mask == 0xff) {
                // all lanes active; replace with a regular load
                llvm::Type *returnType = callInst->getType();
                Assert(llvm::isa<llvm::VectorType>(returnType));
                // cast the i8 * to the appropriate type
                llvm::Value *castPtr = 
                    new llvm::BitCastInst(callInst->getArgOperand(0),
                                          llvm::PointerType::get(returnType, 0), 
                                          "ptr2vec", callInst);
                lCopyMetadata(castPtr, callInst);
                int align = callInst->getCalledFunction() == avxMaskedLoad32 ? 4 : 8;
                llvm::Instruction *loadInst = 
                    new llvm::LoadInst(castPtr, "load", false /* not volatile */,
                                       align, (llvm::Instruction *)NULL);
                lCopyMetadata(loadInst, callInst);
                llvm::ReplaceInstWithInst(callInst, loadInst);
                modifiedAny = true;
                goto restart;
            }
        }
        else if (callInst->getCalledFunction() == avxMaskedStore32 ||
                 callInst->getCalledFunction() == avxMaskedStore64) {
            // NOTE: mask is the 2nd parameter, not the 3rd one!!
            llvm::Value *factor = callInst->getArgOperand(1);
            int mask = lGetMask(factor);
            if (mask == 0) {
                // nothing actually being stored, just remove the inst
                callInst->eraseFromParent();
                modifiedAny = true;
                goto restart;
            }
            else if (mask == 0xff) {
                // all lanes storing, so replace with a regular store
                llvm::Value *rvalue = callInst->getArgOperand(2);
                llvm::Type *storeType = rvalue->getType();
                llvm::Value *castPtr = 
                    new llvm::BitCastInst(callInst->getArgOperand(0),
                                          llvm::PointerType::get(storeType, 0), 
                                          "ptr2vec", callInst);
                lCopyMetadata(castPtr, callInst);

                llvm::StoreInst *storeInst = 
                    new llvm::StoreInst(rvalue, castPtr, (llvm::Instruction *)NULL);
                int align = callInst->getCalledFunction() == avxMaskedStore32 ? 4 : 8;
                storeInst->setAlignment(align);
                lCopyMetadata(storeInst, callInst);
                llvm::ReplaceInstWithInst(callInst, storeInst);

                modifiedAny = true;
                goto restart;
            }
        }
#endif
    }
    return modifiedAny;
}


bool
IntrinsicsOpt::matchesMaskInstruction(llvm::Function *function) {
    for (unsigned int i = 0; i < maskInstructions.size(); ++i)
        if (maskInstructions[i].function != NULL &&
            function == maskInstructions[i].function)
            return true;
    return false;
}


IntrinsicsOpt::BlendInstruction *
IntrinsicsOpt::matchingBlendInstruction(llvm::Function *function) {
    for (unsigned int i = 0; i < blendInstructions.size(); ++i)
        if (blendInstructions[i].function != NULL &&
            function == blendInstructions[i].function)
            return &blendInstructions[i];
    return NULL;
}


static llvm::Pass *
CreateIntrinsicsOptPass() {
    return new IntrinsicsOpt;
}


///////////////////////////////////////////////////////////////////////////

/** This simple optimization pass looks for a vector select instruction
    with an all-on or all-off constant mask, simplifying it to the
    appropriate operand if so.

    @todo The better thing to do would be to submit a patch to LLVM to get
    these; they're presumably pretty simple patterns to match.  
*/
class VSelMovmskOpt : public llvm::BasicBlockPass {
public:
    VSelMovmskOpt()
        : BasicBlockPass(ID) { }

    const char *getPassName() const { return "Vector Select Optimization"; }
    bool runOnBasicBlock(llvm::BasicBlock &BB);

    static char ID;
};

char VSelMovmskOpt::ID = 0;


bool
VSelMovmskOpt::runOnBasicBlock(llvm::BasicBlock &bb) {
    bool modifiedAny = false;

 restart:
    for (llvm::BasicBlock::iterator iter = bb.begin(), e = bb.end(); iter != e; ++iter) {
    // vector select wasn't available before 3.1...
#if defined(LLVM_3_1svn)
        llvm::SelectInst *selectInst = llvm::dyn_cast<llvm::SelectInst>(&*iter);
        if (selectInst != NULL && selectInst->getType()->isVectorTy()) {
            llvm::Value *factor = selectInst->getOperand(0);
            int mask = lGetMask(factor);
            int allOnMask = (1 << g->target.vectorWidth) - 1;
            llvm::Value *value = NULL;
            if (mask == allOnMask)
                // Mask all on -> replace with the first select value
                value = selectInst->getOperand(1);
            else if (mask == 0)
                // Mask all off -> replace with the second select blend value
                value = selectInst->getOperand(1);

            if (value != NULL) {
                llvm::ReplaceInstWithValue(iter->getParent()->getInstList(), 
                                           iter, value);
                modifiedAny = true;
                goto restart;
            }
        }
#endif // LLVM_3_1svn

        llvm::CallInst *callInst = llvm::dyn_cast<llvm::CallInst>(&*iter);
        if (callInst == NULL)
            continue;

        llvm::Function *calledFunc = callInst->getCalledFunction();
        if (calledFunc == NULL || calledFunc != m->module->getFunction("__movmsk"))
            continue;

        int mask = lGetMask(callInst->getArgOperand(0));
        if (mask != -1) {
#if 0
            fprintf(stderr, "mask %d\n", mask);
            callInst->getArgOperand(0)->dump();
            fprintf(stderr, "-----------\n");
#endif
            llvm::ReplaceInstWithValue(iter->getParent()->getInstList(), 
                                       iter, LLVMInt32(mask));
            modifiedAny = true;
            goto restart;
        }
    }

    return modifiedAny;
}


static llvm::Pass *
CreateVSelMovmskOptPass() {
    return new VSelMovmskOpt;
}


///////////////////////////////////////////////////////////////////////////
// DetectGSBaseOffsetsPass

/** When the front-end emits gathers and scatters, it generates an array of
    vector-width pointers to represent the set of addresses to read from or
    write to.  This optimization detects cases when the base pointer is a
    uniform pointer or when the indexing is into an array that can be
    converted into scatters/gathers from a single base pointer and an array
    of offsets.

    See for example the comments discussing the __pseudo_gather functions
    in builtins.cpp for more information about this.
 */
class DetectGSBaseOffsetsPass : public llvm::BasicBlockPass {
public:
    static char ID;
    DetectGSBaseOffsetsPass() : BasicBlockPass(ID) { }

    const char *getPassName() const { return "Gather/Scatter Flattening"; }
    bool runOnBasicBlock(llvm::BasicBlock &BB);
};

char DetectGSBaseOffsetsPass::ID = 0;



/** Check to make sure that this value is actually a pointer in the end.
    We need to make sure that given an expression like vec(offset) +
    ptr2int(ptr), lGetBasePointer() doesn't return vec(offset) for the base
    pointer such that we then treat ptr2int(ptr) as an offset.  This ends
    up being important so that we don't generate LLVM GEP instructions like
    "gep inttoptr 8, i64 %ptr", which in turn can lead to incorrect code
    since LLVM's pointer aliasing analysis assumes that operands after the
    first one to a GEP aren't pointers.
 */
static llvm::Value *
lCheckForActualPointer(llvm::Value *v) {
    if (v == NULL)
        return NULL;
    else if (llvm::isa<LLVM_TYPE_CONST llvm::PointerType>(v->getType()))
        return v;
    else if (llvm::isa<llvm::PtrToIntInst>(v))
        return v;
    else {
        llvm::ConstantExpr *uce = 
            llvm::dyn_cast<llvm::ConstantExpr>(v);
        if (uce != NULL &&
            uce->getOpcode() == llvm::Instruction::PtrToInt)
            return v;
        return NULL;
    }
}


/** Given a llvm::Value representing a varying pointer, this function
    checks to see if all of the elements of the vector have the same value
    (i.e. there's a common base pointer).  If so, it returns the common
    pointer value; otherwise it returns NULL.
 */
static llvm::Value *
lGetBasePointer(llvm::Value *v) {
    llvm::InsertElementInst *ie = llvm::dyn_cast<llvm::InsertElementInst>(v);
    if (ie != NULL) {
        llvm::Value *elements[ISPC_MAX_NVEC];
        LLVMFlattenInsertChain(ie, g->target.vectorWidth, elements);

        // Make sure none of the elements is undefined.
        // TODO: it's probably ok to allow undefined elements and return
        // the base pointer if all of the other elements have the same
        // value.
        for (int i = 0; i < g->target.vectorWidth; ++i)
            if (elements[i] == NULL)
                return NULL;

        // Do all of the elements have the same value?
        for (int i = 0; i < g->target.vectorWidth-1; ++i)
            if (elements[i] != elements[i+1])
                return NULL;

        return lCheckForActualPointer(elements[0]);
    }

    // This case comes up with global/static arrays
    llvm::ConstantVector *cv = llvm::dyn_cast<llvm::ConstantVector>(v);
    if (cv != NULL)
        return lCheckForActualPointer(cv->getSplatValue());

    return NULL;
}


/** Given the two operands to a constant add expression, see if we have the
    form "base pointer + offset", whee op0 is the base pointer and op1 is
    the offset; if so return the base and the offset. */
static llvm::Constant *
lGetConstantAddExprBaseOffset(llvm::Constant *op0, llvm::Constant *op1, 
                              llvm::Constant **delta) {
    llvm::ConstantExpr *op = llvm::dyn_cast<llvm::ConstantExpr>(op0);
    if (op == NULL || op->getOpcode() != llvm::Instruction::PtrToInt)
        // the first operand isn't a pointer
        return NULL;

    llvm::ConstantInt *opDelta = llvm::dyn_cast<llvm::ConstantInt>(op1);
    if (opDelta == NULL)
        // the second operand isn't an integer operand
        return NULL;

    *delta = opDelta;
    return op0;
}


/** Given a varying pointer in ptrs, this function checks to see if it can
    be determined to be indexing from a common uniform base pointer.  If
    so, the function returns the base pointer llvm::Value and initializes
    *offsets with an int vector of the per-lane offsets
 */
static llvm::Value *
lGetBasePtrAndOffsets(llvm::Value *ptrs, llvm::Value **offsets,
                      llvm::Instruction *insertBefore) {
    llvm::Value *base = lGetBasePointer(ptrs);
    if (base != NULL) {
        // We have a straight up varying pointer with no indexing that's
        // actually all the same value.
        if (g->target.is32Bit)
            *offsets = LLVMInt32Vector(0);
        else
            *offsets = LLVMInt64Vector((int64_t)0);
        return base;
    }

    llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(ptrs);
    if (bop != NULL && bop->getOpcode() == llvm::Instruction::Add) {
        // If we have a common pointer plus something, then we're also
        // good.
        if ((base = lGetBasePtrAndOffsets(bop->getOperand(0), 
                                          offsets, insertBefore)) != NULL) {
            *offsets = 
                llvm::BinaryOperator::Create(llvm::Instruction::Add, *offsets,
                                             bop->getOperand(1), "new_offsets",
                                             insertBefore);
            return base;
        }
        else if ((base = lGetBasePtrAndOffsets(bop->getOperand(1), 
                                               offsets, insertBefore)) != NULL) {
            *offsets = 
                llvm::BinaryOperator::Create(llvm::Instruction::Add, *offsets,
                                             bop->getOperand(0), "new_offsets",
                                             insertBefore);
            return base;
        }
    }

    llvm::ConstantVector *cv = llvm::dyn_cast<llvm::ConstantVector>(ptrs);
    if (cv != NULL) {
        // Indexing into global arrays can lead to this form, with
        // ConstantVectors..
        llvm::SmallVector<llvm::Constant *, ISPC_MAX_NVEC> elements;
#ifdef LLVM_3_1svn
        for (int i = 0; i < (int)cv->getNumOperands(); ++i) {
            llvm::Constant *c = 
                llvm::dyn_cast<llvm::Constant>(cv->getOperand(i));
            if (c == NULL)
                return NULL;
            elements.push_back(c);
        }
#else
        cv->getVectorElements(elements);
#endif

        llvm::Constant *delta[ISPC_MAX_NVEC];
        for (unsigned int i = 0; i < elements.size(); ++i) {
            // For each element, try to decompose it into either a straight
            // up base pointer, or a base pointer plus an integer value.
            llvm::ConstantExpr *ce = llvm::dyn_cast<llvm::ConstantExpr>(elements[i]);
            if (ce == NULL)
                return NULL;

            delta[i] = NULL;
            llvm::Value *elementBase = NULL; // base pointer for this element
            if (ce->getOpcode() == llvm::Instruction::PtrToInt) {
                // If the element is just a ptr to int instruction, treat
                // it as having an offset of zero
                elementBase = ce;
                delta[i] = g->target.is32Bit ? LLVMInt32(0) : LLVMInt64(0);
            }
            else if (ce->getOpcode() == llvm::Instruction::Add) {
                // Try both orderings of the operands to see if we can get
                // a pointer+offset out of them.
                elementBase =
                    lGetConstantAddExprBaseOffset(ce->getOperand(0), 
                                                  ce->getOperand(1),
                                                  &delta[i]);
                if (elementBase == NULL)
                    elementBase = 
                        lGetConstantAddExprBaseOffset(ce->getOperand(1), 
                                                      ce->getOperand(0),
                                                      &delta[i]);
            }

            // We weren't able to find a base pointer in the above.  (We
            // don't expect this to happen; if it does, it may be necessary
            // to handle more cases in the decomposition above.)
            if (elementBase == NULL)
                return NULL;

            Assert(delta[i] != NULL);
            if (base == NULL)
                // The first time we've found a base pointer
                base = elementBase;
            else if (base != elementBase)
                // Different program instances have different base
                // pointers, so no luck.
                return NULL;
        }

        Assert(base != NULL);
#ifdef LLVM_2_9
        *offsets = llvm::ConstantVector::get(delta);
#else
        llvm::ArrayRef<llvm::Constant *> deltas(&delta[0], 
                                                &delta[elements.size()]);
        *offsets = llvm::ConstantVector::get(deltas);
#endif
        return base;
    }

    return NULL;
}


static llvm::Value *
lGetZeroOffsetVector(llvm::Value *origVec) {
    if (origVec->getType() == LLVMTypes::Int32VectorType)
        return LLVMInt32Vector((int32_t)0);
    else
        return LLVMInt64Vector((int64_t)0);
}


#if 0
static void
lPrint(llvm::Value *v, int indent = 0) {
    if (llvm::isa<llvm::PHINode>(v))
        return;

    fprintf(stderr, "%*c", indent, ' ');
    v->dump();

    llvm::Instruction *inst = llvm::dyn_cast<llvm::Instruction>(v);
    if (inst != NULL) {
        for (int i = 0; i < (int)inst->getNumOperands(); ++i) {
            llvm::Value *op = inst->getOperand(i);
            if (llvm::isa<llvm::Constant>(op) == false)
                lPrint(op, indent+4);
        }
    }
}
#endif


/** Given a vector expression in vec, separate it into a compile-time
    constant component and a variable component, returning the two parts in
    *constOffset and *variableOffset.  (It should be the case that the sum
    of these two is exactly equal to the original vector.)

    This routine only handles some (important) patterns; in some cases it
    will fail and return components that are actually compile-time
    constants in *variableOffset.

    Finally, if there aren't any constant (or, respectivaly, variable)
    components, the corresponding return value may be set to NULL.
 */
static void
lExtractConstantOffset(llvm::Value *vec, llvm::Value **constOffset,
                       llvm::Value **variableOffset, 
                       llvm::Instruction *insertBefore) {
    if (llvm::isa<llvm::ConstantVector>(vec) ||
#ifdef LLVM_3_1svn
        llvm::isa<llvm::ConstantDataVector>(vec) ||
#endif
        llvm::isa<llvm::ConstantAggregateZero>(vec)) {
        *constOffset = vec;
        *variableOffset = NULL;
        return;
    }

    llvm::SExtInst *sext = llvm::dyn_cast<llvm::SExtInst>(vec);
    if (sext != NULL) {
        // Check the sext target.
        llvm::Value *co, *vo;
        lExtractConstantOffset(sext->getOperand(0), &co, &vo, insertBefore);

        // make new sext instructions for the two parts
        if (co == NULL)
            *constOffset = NULL;
        else
            *constOffset = new llvm::SExtInst(co, sext->getType(), 
                                              "const_offset_sext", insertBefore);
        if (vo == NULL)
            *variableOffset = NULL;
        else
            *variableOffset = new llvm::SExtInst(vo, sext->getType(), 
                                                 "variable_offset_sext", 
                                                 insertBefore);
        return;
    }

    // FIXME? handle bitcasts / type casts here

    llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(vec);
    if (bop != NULL) {
        llvm::Value *op0 = bop->getOperand(0);
        llvm::Value *op1 = bop->getOperand(1);
        llvm::Value *c0, *v0, *c1, *v1;

        if (bop->getOpcode() == llvm::Instruction::Add) {
            lExtractConstantOffset(op0, &c0, &v0, insertBefore);
            lExtractConstantOffset(op1, &c1, &v1, insertBefore);

            if (c0 == NULL)
                *constOffset = c1;
            else if (c1 == NULL)
                *constOffset = c0;
            else
                *constOffset = 
                    llvm::BinaryOperator::Create(llvm::Instruction::Add, c0, c1,
                                                 "const_op", insertBefore);

            if (v0 == NULL)
                *variableOffset = v1;
            else if (v1 == NULL)
                *variableOffset = v0;
            else
                *variableOffset = 
                    llvm::BinaryOperator::Create(llvm::Instruction::Add, v0, v1,
                                                 "variable_op", insertBefore);
            return;
        }
        else if (bop->getOpcode() == llvm::Instruction::Mul) {
            lExtractConstantOffset(op0, &c0, &v0, insertBefore);
            lExtractConstantOffset(op1, &c1, &v1, insertBefore);

            // Given the product of constant and variable terms, we have:
            // (c0 + v0) * (c1 + v1) == (c0 c1) + (v0 c1 + c0 v1 + v0 v1)
            // Note that the first term is a constant and the last three are
            // variable.
            if (c0 != NULL && c1 != NULL)
                *constOffset =
                    llvm::BinaryOperator::Create(llvm::Instruction::Mul, c0, c1,
                                                 "const_mul", insertBefore);
            else
                *constOffset = NULL;

            llvm::Value *va = NULL, *vb = NULL, *vc = NULL;
            if (v0 != NULL && c1 != NULL)
                va = llvm::BinaryOperator::Create(llvm::Instruction::Mul, v0, c1,
                                                  "va_mul", insertBefore);
            if (c0 != NULL && v1 != NULL)
                vb = llvm::BinaryOperator::Create(llvm::Instruction::Mul, c0, v1,
                                                  "vb_mul", insertBefore);
            if (v0 != NULL && v1 != NULL)
                vc = llvm::BinaryOperator::Create(llvm::Instruction::Mul, v0, v1,
                                                  "vc_mul", insertBefore);

            
            llvm::Value *vab = NULL;
            if (va != NULL && vb != NULL)
                vab = llvm::BinaryOperator::Create(llvm::Instruction::Add, va, vb,
                                                   "vab_add", insertBefore);
            else if (va != NULL)
                vab = va;
            else
                vab = vb;

            if (vab != NULL && vc != NULL)
                *variableOffset = 
                    llvm::BinaryOperator::Create(llvm::Instruction::Add, vab, vc,
                                                 "vabc_add", insertBefore);
            else if (vab != NULL)
                *variableOffset = vab;
            else
                *variableOffset = vc;

            return;
        }
    }

    // Nothing matched, just return what we have as a variable component
    *constOffset = NULL;
    *variableOffset = vec;
}


/* Returns true if the given value is a constant vector of integers with
   the value 2, 4, 8 in all of the elements.  (Returns the splatted value
   in *splat, if so). */
static bool
lIs248Splat(llvm::Value *v, int *splat) {
#ifdef LLVM_3_1svn
    llvm::ConstantDataVector *cvec = 
        llvm::dyn_cast<llvm::ConstantDataVector>(v);
#else
    llvm::ConstantVector *cvec = llvm::dyn_cast<llvm::ConstantVector>(v);
#endif
    if (cvec == NULL)
        return false;

    llvm::Constant *splatConst = cvec->getSplatValue();
    if (splatConst == NULL)
        return false;

    llvm::ConstantInt *ci = 
        llvm::dyn_cast<llvm::ConstantInt>(splatConst);
    if (ci == NULL)
        return false;

    int64_t splatVal = ci->getSExtValue();
    if (splatVal != 2 && splatVal != 4 && splatVal != 8)
        return false;

    *splat = (int)splatVal;
    return true;
}
        

/** Given a vector of integer offsets to a base pointer being used for a
    gather or a scatter, see if its root operation is a multiply by a
    vector of some value by all 2s/4s/8s.  If not, return NULL.

    If it is return an i32 value of 2, 4, 8 from the function and modify
    *vec so that it points to the operand that is being multiplied by
    2/4/8.

    We go through all this trouble so that we can pass the i32 scale factor
    to the {gather,scatter}_base_offsets function as a separate scale
    factor for the offsets.  This in turn is used in a way so that the LLVM
    x86 code generator matches it to apply x86's free scale by 2x, 4x, or
    8x to one of two registers being added together for an addressing
    calculation.
 */
static llvm::Value *
lExtractOffsetVector248Scale(llvm::Value **vec) {
    llvm::SExtInst *sext = llvm::dyn_cast<llvm::SExtInst>(*vec);
    if (sext != NULL) {
        llvm::Value *sextOp = sext->getOperand(0);
        // Check the sext target.
        llvm::Value *scale = lExtractOffsetVector248Scale(&sextOp);
        if (scale == NULL)
            return NULL;

        // make a new sext instruction so that we end up with the right
        // type
        *vec = new llvm::SExtInst(sextOp, sext->getType(), "offset_sext", sext);
        return scale;
    }


    // If we don't have a binary operator, then just give up
    llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(*vec);
    if (bop == NULL)
        return LLVMInt32(1);

    llvm::Value *op0 = bop->getOperand(0), *op1 = bop->getOperand(1);
    if (bop->getOpcode() == llvm::Instruction::Add) {
        if (llvm::isa<llvm::ConstantAggregateZero>(op0)) {
            *vec = op1;
            return lExtractOffsetVector248Scale(vec);
        }
        else if (llvm::isa<llvm::ConstantAggregateZero>(op1)) {
            *vec = op0;
            return lExtractOffsetVector248Scale(vec);
        }
        else {
            llvm::Value *s0 = lExtractOffsetVector248Scale(&op0);
            llvm::Value *s1 = lExtractOffsetVector248Scale(&op1);
            if (s0 == s1) {
                *vec = llvm::BinaryOperator::Create(llvm::Instruction::Add,
                                                    op0, op1, "new_add", bop);
                return s0;
            }
            else
                return LLVMInt32(1);
        }
    }
    else if (bop->getOpcode() == llvm::Instruction::Mul) {
        // Check each operand for being one of the scale factors we care about.
        int splat;
        if (lIs248Splat(op0, &splat)) {
            *vec = op1;
            return LLVMInt32(splat);
        }
        else if (lIs248Splat(op1, &splat)) {
            *vec = op0;
            return LLVMInt32(splat);
        }
        else
            return LLVMInt32(1);
    }
    else
        return LLVMInt32(1);
}

#if 0
static llvm::Value *
lExtractUniforms(llvm::Value **vec, llvm::Instruction *insertBefore) {
    fprintf(stderr, " lextract: ");
    (*vec)->dump();
    fprintf(stderr, "\n");

    if (llvm::isa<llvm::ConstantVector>(*vec) ||
#ifdef LLVM_3_1svn
        llvm::isa<llvm::ConstantDataVector>(*vec) ||
#endif
        llvm::isa<llvm::ConstantAggregateZero>(*vec))
        return NULL;

    llvm::SExtInst *sext = llvm::dyn_cast<llvm::SExtInst>(*vec);
    if (sext != NULL) {
        llvm::Value *sextOp = sext->getOperand(0);
        // Check the sext target.
        llvm::Value *unif = lExtractUniforms(&sextOp, insertBefore);
        if (unif == NULL)
            return NULL;

        // make a new sext instruction so that we end up with the right
        // type
        *vec = new llvm::SExtInst(sextOp, sext->getType(), "offset_sext", sext);
        return unif;
    }

    std::vector<llvm::PHINode *> phis;
    if (LLVMVectorValuesAllEqual(*vec, g->target.vectorWidth, phis)) {
        // FIXME: we may want to redo all of the expression here, in scalar
        // form (if at all possible), for code quality...
        llvm::Value *unif = 
            llvm::ExtractElementInst::Create(*vec, LLVMInt32(0),
                                             "first_uniform", insertBefore);
        *vec = NULL;
        return unif;
    }

    llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(*vec);
    if (bop == NULL)
        return NULL;

    llvm::Value *op0 = bop->getOperand(0), *op1 = bop->getOperand(1);
    if (bop->getOpcode() == llvm::Instruction::Add) {
        llvm::Value *s0 = lExtractUniforms(&op0, insertBefore);
        llvm::Value *s1 = lExtractUniforms(&op1, insertBefore);
        if (s0 == NULL && s1 == NULL)
            return NULL;

        if (op0 == NULL)
            *vec = op1;
        else if (op1 == NULL)
            *vec = op0;
        else
            *vec = llvm::BinaryOperator::Create(llvm::Instruction::Add,
                                                op0, op1, "new_add", insertBefore);

        if (s0 == NULL) 
            return s1;
        else if (s1 == NULL)
            return s0;
        else
            return llvm::BinaryOperator::Create(llvm::Instruction::Add, s0, s1,
                                                "add_unif", insertBefore);
    }
#if 0
    else if (bop->getOpcode() == llvm::Instruction::Mul) {
        // Check each operand for being one of the scale factors we care about.
        int splat;
        if (lIs248Splat(op0, &splat)) {
            *vec = op1;
            return LLVMInt32(splat);
        }
        else if (lIs248Splat(op1, &splat)) {
            *vec = op0;
            return LLVMInt32(splat);
        }
        else
            return LLVMInt32(1);
    }
#endif
    else
        return NULL;
}


static void
lExtractUniformsFromOffset(llvm::Value **basePtr, llvm::Value **offsetVector, 
                           llvm::Value *offsetScale, 
                           llvm::Instruction *insertBefore) {
#if 1
    (*basePtr)->dump();
    printf("\n");
    (*offsetVector)->dump();
    printf("\n");
    offsetScale->dump();
    printf("-----\n");
#endif

    llvm::Value *uniformDelta = lExtractUniforms(offsetVector, insertBefore);
    if (uniformDelta == NULL)
        return;

    llvm::Value *index[1] = { uniformDelta };
#if defined(LLVM_3_0) || defined(LLVM_3_0svn) || defined(LLVM_3_1svn)
    llvm::ArrayRef<llvm::Value *> arrayRef(&index[0], &index[1]);
    *basePtr = llvm::GetElementPtrInst::Create(*basePtr, arrayRef, "new_base",
                                               insertBefore);
#else
    *basePtr = llvm::GetElementPtrInst::Create(*basePtr, &index[0], 
                                               &index[1], "new_base",
                                               insertBefore);
#endif

    // this should only happen if we have only uniforms, but that in turn
    // shouldn't be a gather/scatter!
    Assert(*offsetVector != NULL);
}
#endif

struct GSInfo {
    GSInfo(const char *pgFuncName, const char *pgboFuncName, 
           const char *pgbo32FuncName, bool ig) 
        : isGather(ig) {
        func = m->module->getFunction(pgFuncName);
        baseOffsetsFunc = m->module->getFunction(pgboFuncName);
        baseOffsets32Func = m->module->getFunction(pgbo32FuncName);
    }
    llvm::Function *func;
    llvm::Function *baseOffsetsFunc, *baseOffsets32Func;
    const bool isGather;
};


bool
DetectGSBaseOffsetsPass::runOnBasicBlock(llvm::BasicBlock &bb) {
    GSInfo gsFuncs[] = {
        GSInfo("__pseudo_gather32_8",  "__pseudo_gather_base_offsets32_8",
               "__pseudo_gather_base_offsets32_8", true),
        GSInfo("__pseudo_gather32_16", "__pseudo_gather_base_offsets32_16", 
               "__pseudo_gather_base_offsets32_16", true),
        GSInfo("__pseudo_gather32_32", "__pseudo_gather_base_offsets32_32", 
               "__pseudo_gather_base_offsets32_32", true),
        GSInfo("__pseudo_gather32_64", "__pseudo_gather_base_offsets32_64", 
               "__pseudo_gather_base_offsets32_64", true),
        GSInfo("__pseudo_scatter32_8",  "__pseudo_scatter_base_offsets32_8", 
               "__pseudo_scatter_base_offsets32_8", false),
        GSInfo("__pseudo_scatter32_16", "__pseudo_scatter_base_offsets32_16", 
               "__pseudo_scatter_base_offsets32_16", false),
        GSInfo("__pseudo_scatter32_32", "__pseudo_scatter_base_offsets32_32", 
               "__pseudo_scatter_base_offsets32_32", false),
        GSInfo("__pseudo_scatter32_64", "__pseudo_scatter_base_offsets32_64", 
               "__pseudo_scatter_base_offsets32_64", false),
        GSInfo("__pseudo_gather64_8",  "__pseudo_gather_base_offsets64_8", 
               "__pseudo_gather_base_offsets32_8", true),
        GSInfo("__pseudo_gather64_16", "__pseudo_gather_base_offsets64_16", 
               "__pseudo_gather_base_offsets32_16", true),
        GSInfo("__pseudo_gather64_32", "__pseudo_gather_base_offsets64_32", 
               "__pseudo_gather_base_offsets32_32", true),
        GSInfo("__pseudo_gather64_64", "__pseudo_gather_base_offsets64_64", 
               "__pseudo_gather_base_offsets32_64", true),
        GSInfo("__pseudo_scatter64_8",  "__pseudo_scatter_base_offsets64_8", 
               "__pseudo_scatter_base_offsets32_8", false),
        GSInfo("__pseudo_scatter64_16", "__pseudo_scatter_base_offsets64_16", 
               "__pseudo_scatter_base_offsets32_16", false),
        GSInfo("__pseudo_scatter64_32", "__pseudo_scatter_base_offsets64_32", 
               "__pseudo_scatter_base_offsets32_32", false),
        GSInfo("__pseudo_scatter64_64", "__pseudo_scatter_base_offsets64_64", 
               "__pseudo_scatter_base_offsets32_64", false),
    };
    int numGSFuncs = sizeof(gsFuncs) / sizeof(gsFuncs[0]);
    for (int i = 0; i < numGSFuncs; ++i)
        Assert(gsFuncs[i].func != NULL && gsFuncs[i].baseOffsetsFunc != NULL &&
               gsFuncs[i].baseOffsets32Func != NULL);

    bool modifiedAny = false;
 restart:
    // Iterate through all of the instructions in the basic block.
    for (llvm::BasicBlock::iterator iter = bb.begin(), e = bb.end(); iter != e; ++iter) {
        llvm::CallInst *callInst = llvm::dyn_cast<llvm::CallInst>(&*iter);
        // If we don't have a call to one of the
        // __pseudo_{gather,scatter}_* functions, then just go on to the
        // next instruction.
        if (callInst == NULL)
            continue;
        GSInfo *info = NULL;
        for (int i = 0; i < numGSFuncs; ++i)
            if (gsFuncs[i].func != NULL &&
                callInst->getCalledFunction() == gsFuncs[i].func) {
                info = &gsFuncs[i];
                break;
            }
        if (info == NULL)
            continue;

        // Try to transform the array of pointers to a single base pointer
        // and an array of int32 offsets.  (All the hard work is done by
        // lGetBasePtrAndOffsets).
        llvm::Value *ptrs = callInst->getArgOperand(0);
        llvm::Value *offsetVector = NULL;
        llvm::Value *basePtr = lGetBasePtrAndOffsets(ptrs, &offsetVector, 
                                                     callInst);

        if (basePtr == NULL || offsetVector == NULL)
            // It's actually a fully general gather/scatter with a varying
            // set of base pointers, so leave it as is and continune onward
            // to the next instruction...
            continue;

        // Try to decompose the offset vector into a compile time constant
        // component and a varying component.  The constant component is
        // passed as a separate parameter to the gather/scatter functions,
        // which in turn allows their implementations to end up emitting
        // x86 instructions with constant offsets encoded in them.
        llvm::Value *constOffset, *variableOffset;
        lExtractConstantOffset(offsetVector, &constOffset, &variableOffset, 
                               callInst);
        if (constOffset == NULL)
            constOffset = lGetZeroOffsetVector(offsetVector);
        if (variableOffset == NULL)
            variableOffset = lGetZeroOffsetVector(offsetVector);

        // See if the varying component is scaled by 2, 4, or 8.  If so,
        // extract that scale factor and rewrite variableOffset to remove
        // it.  (This also is pulled out so that we can match the scales by
        // 2/4/8 offered by x86 addressing operators.)
        llvm::Value *offsetScale = lExtractOffsetVector248Scale(&variableOffset);

        // Cast the base pointer to a void *, since that's what the
        // __pseudo_*_base_offsets_* functions want.
        basePtr = new llvm::IntToPtrInst(basePtr, LLVMTypes::VoidPointerType,
                                         "base2void", callInst);
        lCopyMetadata(basePtr, callInst);

        llvm::Function *gatherScatterFunc = info->baseOffsetsFunc;

        if (g->opt.force32BitAddressing) {
            // If we're doing 32-bit addressing on a 64-bit target, here we
            // will see if we can call one of the 32-bit variants of the
            // pseudo gather/scatter functions.  Specifically, if the
            // offset vector turns out to be an i32 value that was sext'ed
            // to be i64 immediately before the scatter/gather, then we
            // walk past the sext to get the i32 offset values and then
            // call out to the corresponding 32-bit gather/scatter
            // function.
            llvm::SExtInst *sext = llvm::dyn_cast<llvm::SExtInst>(variableOffset);
            if (sext != NULL && 
                sext->getOperand(0)->getType() == LLVMTypes::Int32VectorType) {
                variableOffset = sext->getOperand(0);
                gatherScatterFunc = info->baseOffsets32Func;
                if (constOffset->getType() != LLVMTypes::Int32VectorType)
                    constOffset = 
                        new llvm::TruncInst(constOffset, LLVMTypes::Int32VectorType,
                                            "trunc_const_offset", callInst);
            }
        }

        if (info->isGather) {
            llvm::Value *mask = callInst->getArgOperand(1);

            // Generate a new function call to the next pseudo gather
            // base+offsets instruction.  Note that we're passing a NULL
            // llvm::Instruction to llvm::CallInst::Create; this means that
            // the instruction isn't inserted into a basic block and that
            // way we can then call ReplaceInstWithInst().
            llvm::Instruction *newCall = 
                lCallInst(gatherScatterFunc, basePtr, variableOffset, offsetScale,
                          constOffset, mask, "newgather", NULL);
            lCopyMetadata(newCall, callInst);
            llvm::ReplaceInstWithInst(callInst, newCall);
        }
        else {
            llvm::Value *storeValue = callInst->getArgOperand(1);
            llvm::Value *mask = callInst->getArgOperand(2);

            // Generate a new function call to the next pseudo scatter
            // base+offsets instruction.  See above for why passing NULL
            // for the Instruction * is intended.
            llvm::Instruction *newCall = 
                lCallInst(gatherScatterFunc, basePtr, variableOffset, offsetScale,
                          constOffset, storeValue, mask, "", NULL);
            lCopyMetadata(newCall, callInst);
            llvm::ReplaceInstWithInst(callInst, newCall);
        }
        modifiedAny = true;
        goto restart;
    }

    return modifiedAny;
}


static llvm::Pass *
CreateDetectGSBaseOffsetsPass() {
    return new DetectGSBaseOffsetsPass;
}


///////////////////////////////////////////////////////////////////////////
// MaskedStoreOptPass

/** Masked stores are generally more complex than regular stores; for
    example, they require multiple instructions to simulate under SSE.
    This optimization detects cases where masked stores can be replaced
    with regular stores or removed entirely, for the cases of an 'all on'
    mask and an 'all off' mask, respectively.
*/
class MaskedStoreOptPass : public llvm::BasicBlockPass {
public:
    static char ID;
    MaskedStoreOptPass() : BasicBlockPass(ID) { }

    const char *getPassName() const { return "Masked Store Scalarize"; }
    bool runOnBasicBlock(llvm::BasicBlock &BB);
};


char MaskedStoreOptPass::ID = 0;

struct MSInfo {
    MSInfo(const char *name, const int a) 
        : align(a) {
        func = m->module->getFunction(name);
        Assert(func != NULL);
    }
    llvm::Function *func;
    const int align;
};
        

bool
MaskedStoreOptPass::runOnBasicBlock(llvm::BasicBlock &bb) {
    MSInfo msInfo[] = {
        MSInfo("__pseudo_masked_store_8",  1),
        MSInfo("__pseudo_masked_store_16", 2),
        MSInfo("__pseudo_masked_store_32", 4),
        MSInfo("__pseudo_masked_store_64", 8),
        MSInfo("__masked_store_blend_8",  1),
        MSInfo("__masked_store_blend_16", 2),
        MSInfo("__masked_store_blend_32", 4),
        MSInfo("__masked_store_blend_64", 8),
        MSInfo("__masked_store_8",  1),
        MSInfo("__masked_store_16", 2),
        MSInfo("__masked_store_32", 4),
        MSInfo("__masked_store_64", 8)
    };

    bool modifiedAny = false;
 restart:
    // Iterate over all of the instructions to look for one of the various
    // masked store functions
    for (llvm::BasicBlock::iterator iter = bb.begin(), e = bb.end(); iter != e; ++iter) {
        llvm::CallInst *callInst = llvm::dyn_cast<llvm::CallInst>(&*iter);
        if (callInst == NULL)
            continue;

        llvm::Function *called = callInst->getCalledFunction();
        if (called == NULL)
            continue;

        int nMSFuncs = sizeof(msInfo) / sizeof(msInfo[0]);
        MSInfo *info = NULL;
        for (int i = 0; i < nMSFuncs; ++i) {
            if (msInfo[i].func != NULL && called == msInfo[i].func) {
                info = &msInfo[i];
                break;
            }
        }
        if (info == NULL)
            continue;

        // Got one; grab the operands
        llvm::Value *lvalue = callInst->getArgOperand(0);
        llvm::Value *rvalue  = callInst->getArgOperand(1);
        llvm::Value *mask = callInst->getArgOperand(2);

        int allOnMask = (1 << g->target.vectorWidth) - 1;

        int maskAsInt = lGetMask(mask);
        if (maskAsInt == 0) {
            // Zero mask - no-op, so remove the store completely.  (This
            // may in turn lead to being able to optimize out instructions
            // that compute the rvalue...)
            callInst->eraseFromParent();
            modifiedAny = true;
            goto restart;
        }
        else if (maskAsInt == allOnMask) {
            // The mask is all on, so turn this into a regular store
            LLVM_TYPE_CONST llvm::Type *rvalueType = rvalue->getType();
            LLVM_TYPE_CONST llvm::Type *ptrType = 
                llvm::PointerType::get(rvalueType, 0);

            lvalue = new llvm::BitCastInst(lvalue, ptrType, "lvalue_to_ptr_type", callInst);
            lCopyMetadata(lvalue, callInst);
            llvm::Instruction *store = 
                new llvm::StoreInst(rvalue, lvalue, false /* not volatile */,
                                    info->align);
            lCopyMetadata(store, callInst);
            llvm::ReplaceInstWithInst(callInst, store);

            modifiedAny = true;
            goto restart;
        }
    }

    return modifiedAny;
}


static llvm::Pass *
CreateMaskedStoreOptPass() {
    return new MaskedStoreOptPass;
}


///////////////////////////////////////////////////////////////////////////
// MaskedLoadOptPass

/** Masked load improvements for the all on/all off mask cases.
*/
class MaskedLoadOptPass : public llvm::BasicBlockPass {
public:
    static char ID;
    MaskedLoadOptPass() : BasicBlockPass(ID) { }

    const char *getPassName() const { return "Masked Load Improvements"; }
    bool runOnBasicBlock(llvm::BasicBlock &BB);
};


char MaskedLoadOptPass::ID = 0;

struct MLInfo {
    MLInfo(const char *name, const int a) 
        : align(a) {
        func = m->module->getFunction(name);
        Assert(func != NULL);
    }
    llvm::Function *func;
    const int align;
};
        

bool
MaskedLoadOptPass::runOnBasicBlock(llvm::BasicBlock &bb) {
    MLInfo mlInfo[] = {
        MLInfo("__masked_load_8",  1),
        MLInfo("__masked_load_16", 2),
        MLInfo("__masked_load_32", 4),
        MLInfo("__masked_load_64", 8)
    };

    bool modifiedAny = false;
 restart:
    // Iterate over all of the instructions to look for one of the various
    // masked load functions
    for (llvm::BasicBlock::iterator iter = bb.begin(), e = bb.end(); iter != e; ++iter) {
        llvm::CallInst *callInst = llvm::dyn_cast<llvm::CallInst>(&*iter);
        if (!callInst)
            continue;

        llvm::Function *called = callInst->getCalledFunction();
        if (called == NULL)
            continue;

        int nFuncs = sizeof(mlInfo) / sizeof(mlInfo[0]);
        MLInfo *info = NULL;
        for (int i = 0; i < nFuncs; ++i) {
            if (mlInfo[i].func != NULL && called == mlInfo[i].func) {
                info = &mlInfo[i];
                break;
            }
        }
        if (info == NULL)
            continue;

        // Got one; grab the operands
        llvm::Value *ptr = callInst->getArgOperand(0);
        llvm::Value *mask  = callInst->getArgOperand(1);
        int allOnMask = (1 << g->target.vectorWidth) - 1;

        int maskAsInt = lGetMask(mask);
        if (maskAsInt == 0) {
            // Zero mask - no-op, so replace the load with an undef value
            llvm::ReplaceInstWithValue(iter->getParent()->getInstList(),
                                       iter, llvm::UndefValue::get(callInst->getType()));
            modifiedAny = true;
            goto restart;
        }
        else if (maskAsInt == allOnMask) {
            // The mask is all on, so turn this into a regular load
            LLVM_TYPE_CONST llvm::Type *ptrType = 
                llvm::PointerType::get(callInst->getType(), 0);
            ptr = new llvm::BitCastInst(ptr, ptrType, "ptr_cast_for_load", 
                                        callInst);
            llvm::Instruction *load = 
                new llvm::LoadInst(ptr, callInst->getName(), false /* not volatile */,
                                   info->align, (llvm::Instruction *)NULL);
            lCopyMetadata(load, callInst);
            llvm::ReplaceInstWithInst(callInst, load);
            modifiedAny = true;
            goto restart;
        }
    }
    return modifiedAny;
}


static llvm::Pass *
CreateMaskedLoadOptPass() {
    return new MaskedLoadOptPass;
}


///////////////////////////////////////////////////////////////////////////
// PseudoMaskedStorePass

/** When the front-end needs to do a masked store, it emits a
    __pseudo_masked_store* call as a placeholder.  This pass lowers these
    calls to either __masked_store* or __masked_store_blend* calls.  
*/
class PseudoMaskedStorePass : public llvm::BasicBlockPass {
public:
    static char ID;
    PseudoMaskedStorePass() : BasicBlockPass(ID) { }

    const char *getPassName() const { return "Lower Masked Stores"; }
    bool runOnBasicBlock(llvm::BasicBlock &BB);
};


char PseudoMaskedStorePass::ID = 0;


/** This routine attempts to determine if the given pointer in lvalue is
    pointing to stack-allocated memory.  It's conservative in that it
    should never return true for non-stack allocated memory, but may return
    false for memory that actually is stack allocated.  The basic strategy
    is to traverse through the operands and see if the pointer originally
    comes from an AllocaInst.
*/
static bool
lIsSafeToBlend(llvm::Value *lvalue) {
    llvm::BitCastInst *bc = llvm::dyn_cast<llvm::BitCastInst>(lvalue);
    if (bc != NULL)
        return lIsSafeToBlend(bc->getOperand(0));
    else {
        llvm::AllocaInst *ai = llvm::dyn_cast<llvm::AllocaInst>(lvalue);
        if (ai) {
            LLVM_TYPE_CONST llvm::Type *type = ai->getType();
            LLVM_TYPE_CONST llvm::PointerType *pt = 
                llvm::dyn_cast<LLVM_TYPE_CONST llvm::PointerType>(type);
            assert(pt != NULL);
            type = pt->getElementType();
            LLVM_TYPE_CONST llvm::ArrayType *at;
            while ((at = llvm::dyn_cast<LLVM_TYPE_CONST llvm::ArrayType>(type))) {
                type = at->getElementType();
            }
            LLVM_TYPE_CONST llvm::VectorType *vt = 
                llvm::dyn_cast<LLVM_TYPE_CONST llvm::VectorType>(type);
            return (vt != NULL && 
                    (int)vt->getNumElements() == g->target.vectorWidth);
        }
        else {
            llvm::GetElementPtrInst *gep = 
                llvm::dyn_cast<llvm::GetElementPtrInst>(lvalue);
            if (gep != NULL)
                return lIsSafeToBlend(gep->getOperand(0));
            else
                return false;
        }
    }
}


struct LMSInfo {
    LMSInfo(const char *pname, const char *bname, const char *msname) {
        pseudoFunc = m->module->getFunction(pname);
        blendFunc = m->module->getFunction(bname);
        maskedStoreFunc = m->module->getFunction(msname);
        Assert(pseudoFunc != NULL && blendFunc != NULL && 
               maskedStoreFunc != NULL);
    }
    llvm::Function *pseudoFunc;
    llvm::Function *blendFunc;
    llvm::Function *maskedStoreFunc;
};


bool
PseudoMaskedStorePass::runOnBasicBlock(llvm::BasicBlock &bb) {
    LMSInfo msInfo[] = {
        LMSInfo("__pseudo_masked_store_8", "__masked_store_blend_8", 
                "__masked_store_8"),
        LMSInfo("__pseudo_masked_store_16", "__masked_store_blend_16", 
                "__masked_store_16"),
        LMSInfo("__pseudo_masked_store_32", "__masked_store_blend_32", 
                "__masked_store_32"),
        LMSInfo("__pseudo_masked_store_64", "__masked_store_blend_64", 
                "__masked_store_64")
    };

    bool modifiedAny = false;
 restart:
    for (llvm::BasicBlock::iterator iter = bb.begin(), e = bb.end(); iter != e; ++iter) {
        // Iterate through all of the instructions and look for
        // __pseudo_masked_store_* calls.
        llvm::CallInst *callInst = llvm::dyn_cast<llvm::CallInst>(&*iter);
        if (callInst == NULL)
            continue;
        LMSInfo *info = NULL;
        for (unsigned int i = 0; i < sizeof(msInfo) / sizeof(msInfo[0]); ++i) {
            if (msInfo[i].pseudoFunc != NULL &&
                callInst->getCalledFunction() == msInfo[i].pseudoFunc) {
                info = &msInfo[i];
                break;
            }
        }
        if (info == NULL)
            continue;

        llvm::Value *lvalue = callInst->getArgOperand(0);
        llvm::Value *rvalue  = callInst->getArgOperand(1);
        llvm::Value *mask = callInst->getArgOperand(2);

        // We need to choose between doing the load + blend + store trick,
        // or serializing the masked store.  Even on targets with a native
        // masked store instruction, this is preferable since it lets us
        // keep values in registers rather than going out to the stack.
        bool doBlend = (!g->opt.disableBlendedMaskedStores &&
                        lIsSafeToBlend(lvalue));

        // Generate the call to the appropriate masked store function and
        // replace the __pseudo_* one with it.
        llvm::Function *fms = doBlend ? info->blendFunc : info->maskedStoreFunc;
        llvm::Instruction *inst = lCallInst(fms, lvalue, rvalue, mask, "", callInst);
        lCopyMetadata(inst, callInst);

        callInst->eraseFromParent();
        modifiedAny = true;
        goto restart;
    }

    return modifiedAny;
}


static llvm::Pass *
CreatePseudoMaskedStorePass() {
    return new PseudoMaskedStorePass;
}

///////////////////////////////////////////////////////////////////////////
// GSToLoadStorePass

/** After earlier optimization passes have run, we are sometimes able to
    determine that gathers/scatters are actually accessing memory in a more
    regular fashion and then change the operation to something simpler and
    more efficient.  For example, if all of the lanes in a gather are
    reading from the same location, we can instead do a scalar load and
    broadcast.  This pass examines gathers and scatters and tries to
    simplify them if at all possible.

    @todo Currently, this only looks for all program instances going to the
    same location and all going to a linear sequence of locations in
    memory.  There are a number of other cases that might make sense to
    look for, including things that could be handled with a vector load +
    shuffle or things that could be handled with hybrids of e.g. 2 4-wide
    vector loads with AVX, etc.
*/
class GSToLoadStorePass : public llvm::BasicBlockPass {
public:
    static char ID;
    GSToLoadStorePass() : BasicBlockPass(ID) { }

    const char *getPassName() const { return "Gather/Scatter Improvements"; }
    bool runOnBasicBlock(llvm::BasicBlock &BB);
};


char GSToLoadStorePass::ID = 0;



/** Given a vector of compile-time constant integer values, test to see if
    they are a linear sequence of constant integers starting from an
    arbirary value but then having a step of value "stride" between
    elements.
 */
static bool
lVectorIsLinearConstantInts(
#ifdef LLVM_3_1svn
                            llvm::ConstantDataVector *cv, 
#else
                            llvm::ConstantVector *cv, 
#endif
                            int vectorLength, 
                            int stride) {
    // Flatten the vector out into the elements array
    llvm::SmallVector<llvm::Constant *, ISPC_MAX_NVEC> elements;
#ifdef LLVM_3_1svn
    for (int i = 0; i < (int)cv->getNumElements(); ++i)
        elements.push_back(cv->getElementAsConstant(i));
#else
    cv->getVectorElements(elements);
#endif
    Assert((int)elements.size() == vectorLength);

    llvm::ConstantInt *ci = llvm::dyn_cast<llvm::ConstantInt>(elements[0]);
    if (ci == NULL)
        // Not a vector of integers
        return false;

    int64_t prevVal = ci->getSExtValue();

    // For each element in the array, see if it is both a ConstantInt and
    // if the difference between it and the value of the previous element
    // is stride.  If not, fail.
    for (int i = 1; i < vectorLength; ++i) {
        ci = llvm::dyn_cast<llvm::ConstantInt>(elements[i]);
        if (ci == NULL) 
            return false;

        int64_t nextVal = ci->getSExtValue();
        if (prevVal + stride != nextVal)
            return false;

        prevVal = nextVal;
    }
    return true;
}


static bool lVectorIsLinear(llvm::Value *v, int vectorLength, int stride,
                            std::vector<llvm::PHINode *> &seenPhis);

/** Checks to see if (op0 * op1) is a linear vector where the result is a
    vector with values that increase by stride.
 */
static bool
lCheckMulForLinear(llvm::Value *op0, llvm::Value *op1, int vectorLength, 
                   int stride, std::vector<llvm::PHINode *> &seenPhis) {
    // Is the first operand a constant integer value splatted across all of
    // the lanes?
#ifdef LLVM_3_1svn
    llvm::ConstantDataVector *cv = llvm::dyn_cast<llvm::ConstantDataVector>(op0);
#else
    llvm::ConstantVector *cv = llvm::dyn_cast<llvm::ConstantVector>(op0);
#endif
    if (cv == NULL)
        return false;

    llvm::Constant *csplat = cv->getSplatValue();
    if (csplat == NULL)
        return false;

    llvm::ConstantInt *splat = llvm::dyn_cast<llvm::ConstantInt>(csplat);
    if (splat == NULL)
        return false;

    // If the splat value doesn't evenly divide the stride we're looking
    // for, there's no way that we can get the linear sequence we're
    // looking or.
    int64_t splatVal = splat->getSExtValue();
    if (splatVal == 0 || splatVal > stride || (stride % splatVal) != 0)
        return false;

    // Check to see if the other operand is a linear vector with stride
    // given by stride/splatVal.
    return lVectorIsLinear(op1, vectorLength, (int)(stride / splatVal), 
                           seenPhis);
}


/** Given vector of integer-typed values, see if the elements of the array
    have a step of 'stride' between their values.  This function tries to
    handle as many possibilities as possible, including things like all
    elements equal to some non-constant value plus an integer offset, etc.
*/
static bool
lVectorIsLinear(llvm::Value *v, int vectorLength, int stride,
                std::vector<llvm::PHINode *> &seenPhis) {
    // First try the easy case: if the values are all just constant
    // integers and have the expected stride between them, then we're done.
#ifdef LLVM_3_1svn
    llvm::ConstantDataVector *cv = llvm::dyn_cast<llvm::ConstantDataVector>(v);
#else
    llvm::ConstantVector *cv = llvm::dyn_cast<llvm::ConstantVector>(v);
#endif
    if (cv != NULL)
        return lVectorIsLinearConstantInts(cv, vectorLength, stride);

    llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(v);
    if (bop != NULL) {
        // FIXME: is it right to pass the seenPhis to the all equal check as well??
        llvm::Value *op0 = bop->getOperand(0), *op1 = bop->getOperand(1);

        if (bop->getOpcode() == llvm::Instruction::Add)
            // There are two cases to check if we have an add:
            //
            // programIndex + unif -> ascending linear seqeuence
            // unif + programIndex -> ascending linear sequence
            return ((lVectorIsLinear(op0, vectorLength, stride, seenPhis) &&
                     LLVMVectorValuesAllEqual(op1, vectorLength, seenPhis)) ||
                    (lVectorIsLinear(op1, vectorLength, stride, seenPhis) &&
                     LLVMVectorValuesAllEqual(op0, vectorLength, seenPhis)));
        else if (bop->getOpcode() == llvm::Instruction::Sub)
            // For subtraction, we only match:
            //
            // programIndex - unif -> ascending linear seqeuence
            //
            // In the future, we could also look for:
            // unif - programIndex -> *descending* linear seqeuence
            // And generate code for that as a vector load + shuffle.
            return (lVectorIsLinear(bop->getOperand(0), vectorLength,
                                    stride, seenPhis) &&
                    LLVMVectorValuesAllEqual(bop->getOperand(1), vectorLength,
                                          seenPhis));
        else if (bop->getOpcode() == llvm::Instruction::Mul)
            // Multiplies are a bit trickier, so are handled in a separate
            // function.
            return (lCheckMulForLinear(op0, op1, vectorLength, stride, seenPhis) ||
                    lCheckMulForLinear(op1, op0, vectorLength, stride, seenPhis));
        else
            return false;
    }

    llvm::CastInst *ci = llvm::dyn_cast<llvm::CastInst>(v);
    if (ci != NULL)
        return lVectorIsLinear(ci->getOperand(0), vectorLength,
                               stride, seenPhis);

    if (llvm::isa<llvm::CallInst>(v) || llvm::isa<llvm::LoadInst>(v))
        return false;

    llvm::PHINode *phi = llvm::dyn_cast<llvm::PHINode>(v);
    if (phi != NULL) {
        for (unsigned int i = 0; i < seenPhis.size(); ++i)
            if (seenPhis[i] == phi)
                return true;

        seenPhis.push_back(phi);

        unsigned int numIncoming = phi->getNumIncomingValues();
        // Check all of the incoming values: if all of them are all equal,
        // then we're good.
        for (unsigned int i = 0; i < numIncoming; ++i) {
            if (!lVectorIsLinear(phi->getIncomingValue(i), vectorLength, stride,
                                 seenPhis)) {
                seenPhis.pop_back();
                return false;
            }
        }

        seenPhis.pop_back();
        return true;
    }

    // TODO: is any reason to worry about these?
    if (llvm::isa<llvm::InsertElementInst>(v))
        return false;

    // TODO: we could also handle shuffles, but we haven't yet seen any
    // cases where doing so would detect cases where actually have a linear
    // vector.
    llvm::ShuffleVectorInst *shuffle = llvm::dyn_cast<llvm::ShuffleVectorInst>(v);
    if (shuffle != NULL)
        return false;

#if 0
    fprintf(stderr, "linear check: ");
    v->dump();
    fprintf(stderr, "\n");
    llvm::Instruction *inst = llvm::dyn_cast<llvm::Instruction>(v);
    if (inst) {
        inst->getParent()->dump();
        fprintf(stderr, "\n");
        fprintf(stderr, "\n");
    }
#endif

    return false;
}


struct GatherImpInfo {
    GatherImpInfo(const char *pName, const char *lbName, const char *lmName,
                  int a) 
        : align(a) {
        pseudoFunc = m->module->getFunction(pName);
        loadBroadcastFunc = m->module->getFunction(lbName);
        loadMaskedFunc = m->module->getFunction(lmName);

        Assert(pseudoFunc != NULL && loadBroadcastFunc != NULL &&
               loadMaskedFunc != NULL);
    }
    llvm::Function *pseudoFunc;
    llvm::Function *loadBroadcastFunc;
    llvm::Function *loadMaskedFunc;
    const int align;
};


static llvm::Value *
lComputeCommonPointer(llvm::Value *base, llvm::Value *offsets,
                      llvm::Instruction *insertBefore) {
    llvm::Value *firstOffset = 
        llvm::ExtractElementInst::Create(offsets, LLVMInt32(0), "first_offset",
                                         insertBefore);

    llvm::Value *offsetIndex[1] = { firstOffset };
#if defined(LLVM_3_0) || defined(LLVM_3_0svn) || defined(LLVM_3_1svn)
    llvm::ArrayRef<llvm::Value *> arrayRef(&offsetIndex[0], &offsetIndex[1]);
    return
        llvm::GetElementPtrInst::Create(base, arrayRef, "ptr", insertBefore);
#else
    return
        llvm::GetElementPtrInst::Create(base, &offsetIndex[0], &offsetIndex[1],
                                        "ptr", insertBefore);
#endif
}


struct ScatterImpInfo {
    ScatterImpInfo(const char *pName, const char *msName, 
                   LLVM_TYPE_CONST llvm::Type *vpt, int a)
        : align(a) {
        pseudoFunc = m->module->getFunction(pName);
        maskedStoreFunc = m->module->getFunction(msName);
        vecPtrType = vpt;
        Assert(pseudoFunc != NULL && maskedStoreFunc != NULL);
    }
    llvm::Function *pseudoFunc;
    llvm::Function *maskedStoreFunc;
    LLVM_TYPE_CONST llvm::Type *vecPtrType;
    const int align;
};
    

bool
GSToLoadStorePass::runOnBasicBlock(llvm::BasicBlock &bb) {
    GatherImpInfo gInfo[] = {
        GatherImpInfo("__pseudo_gather_base_offsets32_8", "__load_and_broadcast_8",
                      "__masked_load_8", 1),
        GatherImpInfo("__pseudo_gather_base_offsets32_16", "__load_and_broadcast_16",
                      "__masked_load_16", 2),
        GatherImpInfo("__pseudo_gather_base_offsets32_32", "__load_and_broadcast_32",
                      "__masked_load_32", 4),
        GatherImpInfo("__pseudo_gather_base_offsets32_64", "__load_and_broadcast_64",
                      "__masked_load_64", 8),
        GatherImpInfo("__pseudo_gather_base_offsets64_8", "__load_and_broadcast_8",
                      "__masked_load_8", 1),
        GatherImpInfo("__pseudo_gather_base_offsets64_16", "__load_and_broadcast_16",
                      "__masked_load_16", 2),
        GatherImpInfo("__pseudo_gather_base_offsets64_32", "__load_and_broadcast_32",
                      "__masked_load_32", 4),
        GatherImpInfo("__pseudo_gather_base_offsets64_64", "__load_and_broadcast_64",
                      "__masked_load_64", 8)
    };
    ScatterImpInfo sInfo[] = {
        ScatterImpInfo("__pseudo_scatter_base_offsets32_8",  "__pseudo_masked_store_8", 
                       LLVMTypes::Int8VectorPointerType, 1),
        ScatterImpInfo("__pseudo_scatter_base_offsets32_16", "__pseudo_masked_store_16",
                       LLVMTypes::Int16VectorPointerType, 2),
        ScatterImpInfo("__pseudo_scatter_base_offsets32_32", "__pseudo_masked_store_32",
                       LLVMTypes::Int32VectorPointerType, 4),
        ScatterImpInfo("__pseudo_scatter_base_offsets32_64", "__pseudo_masked_store_64",
                       LLVMTypes::Int64VectorPointerType, 8),
        ScatterImpInfo("__pseudo_scatter_base_offsets64_8",  "__pseudo_masked_store_8", 
                       LLVMTypes::Int8VectorPointerType, 1),
        ScatterImpInfo("__pseudo_scatter_base_offsets64_16", "__pseudo_masked_store_16",
                       LLVMTypes::Int16VectorPointerType, 2),
        ScatterImpInfo("__pseudo_scatter_base_offsets64_32", "__pseudo_masked_store_32",
                       LLVMTypes::Int32VectorPointerType, 4),
        ScatterImpInfo("__pseudo_scatter_base_offsets64_64", "__pseudo_masked_store_64",
                       LLVMTypes::Int64VectorPointerType, 8)
    };

    bool modifiedAny = false;

 restart:
    for (llvm::BasicBlock::iterator iter = bb.begin(), e = bb.end(); iter != e; ++iter) {
        // Iterate over all of the instructions and look for calls to
        // __pseudo_*_base_offsets_* calls.
        llvm::CallInst *callInst = llvm::dyn_cast<llvm::CallInst>(&*iter);
        if (callInst == NULL)
            continue;

        llvm::Function *calledFunc = callInst->getCalledFunction();
        if (calledFunc == NULL)
            continue;

        GatherImpInfo *gatherInfo = NULL;
        ScatterImpInfo *scatterInfo = NULL;
        for (unsigned int i = 0; i < sizeof(gInfo) / sizeof(gInfo[0]); ++i) {
            if (gInfo[i].pseudoFunc != NULL &&
                calledFunc == gInfo[i].pseudoFunc) {
                gatherInfo = &gInfo[i];
                break;
            }
        }
        for (unsigned int i = 0; i < sizeof(sInfo) / sizeof(sInfo[0]); ++i) {
            if (sInfo[i].pseudoFunc != NULL &&
                calledFunc == sInfo[i].pseudoFunc) {
                scatterInfo = &sInfo[i];
                break;
            }
        }
        if (gatherInfo == NULL && scatterInfo == NULL)
            continue;

        SourcePos pos;
        bool ok = lGetSourcePosFromMetadata(callInst, &pos);
        Assert(ok);     

        llvm::Value *base = callInst->getArgOperand(0);
        llvm::Value *varyingOffsets = callInst->getArgOperand(1);
        llvm::Value *offsetScale = callInst->getArgOperand(2);
        llvm::Value *constOffsets = callInst->getArgOperand(3);
        llvm::Value *storeValue = (scatterInfo != NULL) ? callInst->getArgOperand(4) : NULL;
        llvm::Value *mask = callInst->getArgOperand((gatherInfo != NULL) ? 4 : 5);

        // Compute the full offset vector: offsetScale * varyingOffsets + constOffsets
        llvm::ConstantInt *offsetScaleInt = 
            llvm::dyn_cast<llvm::ConstantInt>(offsetScale);
        Assert(offsetScaleInt != NULL);
        uint64_t scaleValue = offsetScaleInt->getZExtValue();

        std::vector<llvm::Constant *> scales;
        for (int i = 0; i < g->target.vectorWidth; ++i) {
            if (varyingOffsets->getType() == LLVMTypes::Int64VectorType)
                scales.push_back(LLVMInt64(scaleValue));
            else
                scales.push_back(LLVMInt32(scaleValue));
        }
        llvm::Constant *offsetScaleVec = llvm::ConstantVector::get(scales);

        llvm::Value *scaledVarying = 
            llvm::BinaryOperator::Create(llvm::Instruction::Mul, offsetScaleVec,
                                         varyingOffsets, "scaled_varying", callInst);
        llvm::Value *fullOffsets =
            llvm::BinaryOperator::Create(llvm::Instruction::Add, scaledVarying,
                                         constOffsets, "varying+const_offsets",
                                         callInst);

        std::vector<llvm::PHINode *> seenPhis;
        if (LLVMVectorValuesAllEqual(fullOffsets, g->target.vectorWidth, seenPhis)) {
            // If all the offsets are equal, then compute the single
            // pointer they all represent based on the first one of them
            // (arbitrarily).
            llvm::Value *ptr = lComputeCommonPointer(base, fullOffsets, callInst);
            lCopyMetadata(ptr, callInst);

            if (gatherInfo != NULL) {
                // A gather with everyone going to the same location is
                // handled as a scalar load and broadcast across the lanes.
                // Note that we do still have to pass the mask to the
                // __load_and_broadcast_* functions, since they shouldn't
                // access memory if the mask is all off (the location may
                // be invalid in that case).
                Debug(pos, "Transformed gather to scalar load and broadcast!");
                llvm::Instruction *newCall = 
                    lCallInst(gatherInfo->loadBroadcastFunc, ptr, mask, 
                              "load_braodcast");
                lCopyMetadata(newCall, callInst);
                llvm::ReplaceInstWithInst(callInst, newCall);

                modifiedAny = true;
                goto restart;
            }
            else {
                // A scatter with everyone going to the same location is
                // undefined (if there's more than one program instance in
                // the gang).  Issue a warning.
                if (g->target.vectorWidth > 1)
                    Warning(pos, "Undefined behavior: all program instances are "
                            "writing to the same location!");

                // We could do something similar to the gather case, where
                // we arbitrarily write one of the values, but we need to
                // a) check to be sure the mask isn't all off and b) pick
                // the value from an executing program instance in that
                // case.  We'll just let a bunch of the program instances
                // do redundant writes, since this isn't important to make
                // fast anyway...
            }
        }
        else {
            int step = gatherInfo ? gatherInfo->align : scatterInfo->align;

            std::vector<llvm::PHINode *> seenPhis;
            if (step > 0 && lVectorIsLinear(fullOffsets, g->target.vectorWidth, 
                                            step, seenPhis)) {
                // We have a linear sequence of memory locations being accessed
                // starting with the location given by the offset from
                // offsetElements[0], with stride of 4 or 8 bytes (for 32 bit
                // and 64 bit gather/scatters, respectively.)
                llvm::Value *ptr = lComputeCommonPointer(base, fullOffsets, callInst);
                lCopyMetadata(ptr, callInst);

                if (gatherInfo != NULL) {
                    Debug(pos, "Transformed gather to unaligned vector load!");
                    llvm::Instruction *newCall = 
                        lCallInst(gatherInfo->loadMaskedFunc, ptr, mask, "masked_load");
                    lCopyMetadata(newCall, callInst);
                    llvm::ReplaceInstWithInst(callInst, newCall);
                }
                else {
                    Debug(pos, "Transformed scatter to unaligned vector store!");
                    ptr = new llvm::BitCastInst(ptr, scatterInfo->vecPtrType, "ptrcast", 
                                                callInst);
                    llvm::Instruction *newCall =
                        lCallInst(scatterInfo->maskedStoreFunc, ptr, storeValue, 
                                  mask, "");
                    lCopyMetadata(newCall, callInst);
                    llvm::ReplaceInstWithInst(callInst, newCall);
                }

                modifiedAny = true;
                goto restart;
            }
        }
    }

    return modifiedAny;
}


static llvm::Pass *
CreateGSToLoadStorePass() {
    return new GSToLoadStorePass;
}


///////////////////////////////////////////////////////////////////////////
// PseudoGSToGSPass

/** For any gathers and scatters remaining after the GSToLoadStorePass
    runs, we need to turn them into actual native gathers and scatters.
    This task is handled by the PseudoGSToGSPass here.
 */
class PseudoGSToGSPass : public llvm::BasicBlockPass {
public:
    static char ID;
    PseudoGSToGSPass() : BasicBlockPass(ID) { }

    const char *getPassName() const { return "Gather/Scatter Improvements"; }
    bool runOnBasicBlock(llvm::BasicBlock &BB);
};


char PseudoGSToGSPass::ID = 0;


struct LowerGSInfo {
    LowerGSInfo(const char *pName, const char *aName, bool ig)
        : isGather(ig) {
        pseudoFunc = m->module->getFunction(pName);
        actualFunc = m->module->getFunction(aName);
        Assert(pseudoFunc != NULL && actualFunc != NULL);
    }
    llvm::Function *pseudoFunc;
    llvm::Function *actualFunc;
    const bool isGather;
};


bool
PseudoGSToGSPass::runOnBasicBlock(llvm::BasicBlock &bb) {
    LowerGSInfo lgsInfo[] = {
        LowerGSInfo("__pseudo_gather_base_offsets32_8",  "__gather_base_offsets32_i8",  true),
        LowerGSInfo("__pseudo_gather_base_offsets32_16", "__gather_base_offsets32_i16", true),
        LowerGSInfo("__pseudo_gather_base_offsets32_32", "__gather_base_offsets32_i32", true),
        LowerGSInfo("__pseudo_gather_base_offsets32_64", "__gather_base_offsets32_i64", true),

        LowerGSInfo("__pseudo_gather_base_offsets64_8",  "__gather_base_offsets64_i8",  true),
        LowerGSInfo("__pseudo_gather_base_offsets64_16", "__gather_base_offsets64_i16", true),
        LowerGSInfo("__pseudo_gather_base_offsets64_32", "__gather_base_offsets64_i32", true),
        LowerGSInfo("__pseudo_gather_base_offsets64_64", "__gather_base_offsets64_i64", true),

        LowerGSInfo("__pseudo_gather32_8",  "__gather32_i8",  true),
        LowerGSInfo("__pseudo_gather32_16", "__gather32_i16", true),
        LowerGSInfo("__pseudo_gather32_32", "__gather32_i32", true),
        LowerGSInfo("__pseudo_gather32_64", "__gather32_i64", true),

        LowerGSInfo("__pseudo_gather64_8",  "__gather64_i8",  true),
        LowerGSInfo("__pseudo_gather64_16", "__gather64_i16", true),
        LowerGSInfo("__pseudo_gather64_32", "__gather64_i32", true),
        LowerGSInfo("__pseudo_gather64_64", "__gather64_i64", true),

        LowerGSInfo("__pseudo_scatter_base_offsets32_8",  "__scatter_base_offsets32_i8",  false),
        LowerGSInfo("__pseudo_scatter_base_offsets32_16", "__scatter_base_offsets32_i16", false),
        LowerGSInfo("__pseudo_scatter_base_offsets32_32", "__scatter_base_offsets32_i32", false),
        LowerGSInfo("__pseudo_scatter_base_offsets32_64", "__scatter_base_offsets32_i64", false),

        LowerGSInfo("__pseudo_scatter_base_offsets64_8",  "__scatter_base_offsets64_i8",  false),
        LowerGSInfo("__pseudo_scatter_base_offsets64_16", "__scatter_base_offsets64_i16", false),
        LowerGSInfo("__pseudo_scatter_base_offsets64_32", "__scatter_base_offsets64_i32", false),
        LowerGSInfo("__pseudo_scatter_base_offsets64_64", "__scatter_base_offsets64_i64", false),

        LowerGSInfo("__pseudo_scatter32_8",  "__scatter32_i8",  false),
        LowerGSInfo("__pseudo_scatter32_16", "__scatter32_i16", false),
        LowerGSInfo("__pseudo_scatter32_32", "__scatter32_i32", false),
        LowerGSInfo("__pseudo_scatter32_64", "__scatter32_i64", false),

        LowerGSInfo("__pseudo_scatter64_8",  "__scatter64_i8",  false),
        LowerGSInfo("__pseudo_scatter64_16", "__scatter64_i16", false),
        LowerGSInfo("__pseudo_scatter64_32", "__scatter64_i32", false),
        LowerGSInfo("__pseudo_scatter64_64", "__scatter64_i64", false),
    };

    bool modifiedAny = false;

 restart:
    for (llvm::BasicBlock::iterator iter = bb.begin(), e = bb.end(); iter != e; ++iter) {
        // Loop over the instructions and find calls to the
        // __pseudo_*_base_offsets_* functions.
        llvm::CallInst *callInst = llvm::dyn_cast<llvm::CallInst>(&*iter);
        if (callInst == NULL)
            continue;

        llvm::Function *calledFunc = callInst->getCalledFunction();
        if (calledFunc == NULL)
            continue;

        LowerGSInfo *info = NULL;
        for (unsigned int i = 0; i < sizeof(lgsInfo) / sizeof(lgsInfo[0]); ++i) {
            if (lgsInfo[i].pseudoFunc != NULL &&
                calledFunc == lgsInfo[i].pseudoFunc) {
                info = &lgsInfo[i];
                break;
            }
        }
        if (info == NULL)
            continue;

        // Get the source position from the metadata attached to the call
        // instruction so that we can issue PerformanceWarning()s below.
        SourcePos pos;
        bool ok = lGetSourcePosFromMetadata(callInst, &pos);
        Assert(ok);     

        callInst->setCalledFunction(info->actualFunc);
        if (g->target.vectorWidth > 1) {
            if (info->isGather)
                PerformanceWarning(pos, "Gather required to compute value in expression.");
            else
                PerformanceWarning(pos, "Scatter required for storing value.");
        }
        modifiedAny = true;
        goto restart;
    }

    return modifiedAny;
}


static llvm::Pass *
CreatePseudoGSToGSPass() {
    return new PseudoGSToGSPass;
}


///////////////////////////////////////////////////////////////////////////
// IsCompileTimeConstantPass

/** LLVM IR implementations of target-specific functions may include calls
    to the functions "bool __is_compile_time_constant_*(...)"; these allow
    them to have specialied code paths for where the corresponding value is
    known at compile time.  For masks, for example, this allows them to not
    incur the cost of a MOVMSK call at runtime to compute its value in
    cases where the mask value isn't known until runtime.

    This pass resolves these calls into either 'true' or 'false' values so
    that later optimization passes can operate with these as constants.

    See stdlib.m4 for a number of uses of this idiom. 
 */

class IsCompileTimeConstantPass : public llvm::BasicBlockPass {
public:
    static char ID;
    IsCompileTimeConstantPass(bool last = false) : BasicBlockPass(ID) {
        isLastTry = last;
    }

    const char *getPassName() const { return "Resolve \"is compile time constant\""; }
    bool runOnBasicBlock(llvm::BasicBlock &BB);

    bool isLastTry;
};

char IsCompileTimeConstantPass::ID = 0;

bool
IsCompileTimeConstantPass::runOnBasicBlock(llvm::BasicBlock &bb) {
    llvm::Function *funcs[] = {
        m->module->getFunction("__is_compile_time_constant_mask"),
        m->module->getFunction("__is_compile_time_constant_uniform_int32"),
        m->module->getFunction("__is_compile_time_constant_varying_int32")
    };

    bool modifiedAny = false;
 restart:
    for (llvm::BasicBlock::iterator i = bb.begin(), e = bb.end(); i != e; ++i) {
        // Iterate through the instructions looking for calls to the
        // __is_compile_time_constant_*() functions
        llvm::CallInst *callInst = llvm::dyn_cast<llvm::CallInst>(&*i);
        if (callInst == NULL)
            continue;

        int j;
        int nFuncs = sizeof(funcs) / sizeof(funcs[0]);
        for (j = 0; j < nFuncs; ++j) {
            if (funcs[j] != NULL && callInst->getCalledFunction() == funcs[j]) 
                break;
        }
        if (j == nFuncs)
            // not a __is_compile_time_constant_* function
            continue;

        // This optimization pass can be disabled with both the (poorly
        // named) disableGatherScatterFlattening option and
        // disableMaskAllOnOptimizations.
        if (g->opt.disableGatherScatterFlattening ||
            g->opt.disableMaskAllOnOptimizations) {
            llvm::ReplaceInstWithValue(i->getParent()->getInstList(), i, LLVMFalse);
            modifiedAny = true;
            goto restart;
        }

        // Is it a constant?  Bingo, turn the call's value into a constant
        // true value.
        llvm::Value *operand = callInst->getArgOperand(0);
        if (llvm::isa<llvm::Constant>(operand)) {
            llvm::ReplaceInstWithValue(i->getParent()->getInstList(), i, LLVMTrue);
            modifiedAny = true;
            goto restart;
        }

        // This pass runs multiple times during optimization.  Up until the
        // very last time, it only replaces the call with a 'true' if the
        // value is known to be constant and otherwise leaves the call
        // alone, in case further optimization passes can help resolve its
        // value.  The last time through, it eventually has to give up, and
        // replaces any remaining ones with 'false' constants.
        if (isLastTry) {
            llvm::ReplaceInstWithValue(i->getParent()->getInstList(), i, LLVMFalse);
            modifiedAny = true;
            goto restart;
        }
    }

    return modifiedAny;
}


static llvm::Pass *
CreateIsCompileTimeConstantPass(bool isLastTry) {
    return new IsCompileTimeConstantPass(isLastTry);
}


///////////////////////////////////////////////////////////////////////////
// MakeInternalFuncsStaticPass

/** There are a number of target-specific functions that we use during
    these optimization passes.  By the time we are done with optimization,
    any uses of these should be inlined and no calls to these functions
    should remain.  This pass marks all of these functions as having
    private linkage so that subsequent passes can eliminate them as dead
    code, thus cleaning up the final code output by the compiler.  We can't
    just declare these as static from the start, however, since then they
    end up being eliminated as dead code during early optimization passes
    even though we may need to generate calls to them during later
    optimization passes.
 */
class MakeInternalFuncsStaticPass : public llvm::ModulePass {
public:
    static char ID;
    MakeInternalFuncsStaticPass(bool last = false) : ModulePass(ID) {
    }

    const char *getPassName() const { return "Make internal funcs \"static\""; }
    bool runOnModule(llvm::Module &m);
};

char MakeInternalFuncsStaticPass::ID = 0;

bool
MakeInternalFuncsStaticPass::runOnModule(llvm::Module &module) {
    const char *names[] = {
        "__fast_masked_vload", 
        "__gather_base_offsets32_i8", "__gather_base_offsets32_i16",
        "__gather_base_offsets32_i32", "__gather_base_offsets32_i64",
        "__gather_base_offsets64_i8", "__gather_base_offsets64_i16",
        "__gather_base_offsets64_i32", "__gather_base_offsets64_i64",
        "__gather32_i8", "__gather32_i16",
        "__gather32_i32", "__gather32_i64",
        "__gather64_i8", "__gather64_i16",
        "__gather64_i32", "__gather64_i64",
        "__gather_elt32_i8", "__gather_elt32_i16", 
        "__gather_elt32_i32", "__gather_elt32_i64", 
        "__gather_elt64_i8", "__gather_elt64_i16", 
        "__gather_elt64_i32", "__gather_elt64_i64", 
        "__load_and_broadcast_8", "__load_and_broadcast_16",
        "__load_and_broadcast_32", "__load_and_broadcast_64",
        "__masked_load_8", "__masked_load_16",
        "__masked_load_32", "__masked_load_64",
        "__masked_store_8", "__masked_store_16",
        "__masked_store_32", "__masked_store_64",
        "__masked_store_blend_8", "__masked_store_blend_16",
        "__masked_store_blend_32", "__masked_store_blend_64",
        "__scatter_base_offsets32_i8", "__scatter_base_offsets32_i16",
        "__scatter_base_offsets32_i32", "__scatter_base_offsets32_i64",
        "__scatter_base_offsets64_i8", "__scatter_base_offsets64_i16",
        "__scatter_base_offsets64_i32", "__scatter_base_offsets64_i64",
        "__scatter_elt32_i8", "__scatter_elt32_i16", 
        "__scatter_elt32_i32", "__scatter_elt32_i64", 
        "__scatter_elt64_i8", "__scatter_elt64_i16", 
        "__scatter_elt64_i32", "__scatter_elt64_i64", 
        "__scatter32_i8", "__scatter32_i16",
        "__scatter32_i32", "__scatter32_i64",
        "__scatter64_i8", "__scatter64_i16",
        "__scatter64_i32", "__scatter64_i64",
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


static llvm::Pass *
CreateMakeInternalFuncsStaticPass() {
    return new MakeInternalFuncsStaticPass;
}
