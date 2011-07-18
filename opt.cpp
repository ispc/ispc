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
#ifndef LLVM_2_8
    #include <llvm/Target/TargetLibraryInfo.h>
    #ifdef LLVM_2_9
        #include <llvm/Support/StandardPasses.h>
    #else
        #include <llvm/Support/PassManagerBuilder.h>
    #endif // LLVM_2_9
#endif // LLVM_2_8
#include <llvm/ADT/Triple.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Target/TargetData.h>
#include <llvm/Analysis/Verifier.h>
#include <llvm/Support/raw_ostream.h>
#ifndef LLVM_2_8
#include <llvm/Analysis/DIBuilder.h>
#endif
#include <llvm/Analysis/DebugInfo.h>
#include <llvm/Support/Dwarf.h>

static llvm::Pass *CreateIntrinsicsOptPass();
static llvm::Pass *CreateGatherScatterFlattenPass();
static llvm::Pass *CreateGatherScatterImprovementsPass();
static llvm::Pass *CreateLowerGatherScatterPass();
static llvm::Pass *CreateLowerMaskedStorePass();
static llvm::Pass *CreateMaskedStoreOptPass();
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
    llvm::MDNode *line = inst->getMetadata("line");
    llvm::MDNode *column = inst->getMetadata("column");
    if (!filename || !line || !column)
        return false;

    // All of these asserts are things that FunctionEmitContext::addGSMetadata() is
    // expected to have done in its operation
    assert(filename->getNumOperands() == 1 && line->getNumOperands() == 1);
    llvm::MDString *str = llvm::dyn_cast<llvm::MDString>(filename->getOperand(0));
    assert(str);
    llvm::ConstantInt *lnum = llvm::dyn_cast<llvm::ConstantInt>(line->getOperand(0));
    assert(lnum);
    llvm::ConstantInt *colnum = llvm::dyn_cast<llvm::ConstantInt>(column->getOperand(0));
    assert(column);

    *pos = SourcePos(str->getString().data(), (int)lnum->getZExtValue(),
                   (int)colnum->getZExtValue());
    return true;
}


/** Utility routine that prints out the LLVM IR for everything in the
    module.  (Used for debugging).
 */
static void
lPrintModuleCode(llvm::Module *module) {
    llvm::PassManager ppm;
    ppm.add(llvm::createPrintModulePass(&llvm::outs()));
    ppm.run(*module);
}


void
Optimize(llvm::Module *module, int optLevel) {
    if (g->debugPrint) {
        printf("*** Code going into optimization ***\n");
        lPrintModuleCode(module);
    }

    llvm::PassManager optPM;
    llvm::FunctionPassManager funcPM(module);

#ifndef LLVM_2_8
    llvm::TargetLibraryInfo *targetLibraryInfo =
        new llvm::TargetLibraryInfo(llvm::Triple(module->getTargetTriple()));
    optPM.add(targetLibraryInfo);
#endif
    optPM.add(new llvm::TargetData(module));

    if (optLevel == 0) {
        // This is more or less the minimum set of optimizations that we
        // need to do to generate code that will actually run.  (We can't
        // run absolutely no optimizations, since the front-end needs us to
        // take the various __pseudo_* functions it has emitted and turn
        // them into something that can actually execute.
        optPM.add(CreateGatherScatterFlattenPass());
        optPM.add(CreateLowerGatherScatterPass());
        optPM.add(CreateLowerMaskedStorePass());
        optPM.add(CreateIsCompileTimeConstantPass(true));
        optPM.add(llvm::createFunctionInliningPass());
        optPM.add(CreateMakeInternalFuncsStaticPass());
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
#ifndef LLVM_2_8
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
#endif
        // Early optimizations to try to reduce the total amount of code to
        // work with if we can
        optPM.add(CreateGatherScatterFlattenPass());
        optPM.add(llvm::createReassociatePass());
        optPM.add(llvm::createConstantPropagationPass());

        if (!g->opt.disableMaskedStoreOptimizations) {
            optPM.add(CreateIntrinsicsOptPass());
            optPM.add(CreateMaskedStoreOptPass());
        }
        optPM.add(llvm::createDeadInstEliminationPass());

        optPM.add(llvm::createConstantPropagationPass());
        optPM.add(llvm::createDeadInstEliminationPass());
        
        // On to more serious optimizations
        optPM.add(llvm::createCFGSimplificationPass());
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
        optPM.add(llvm::createScalarReplAggregatesPass());
        optPM.add(llvm::createInstructionCombiningPass());
        optPM.add(llvm::createTailCallEliminationPass());

        if (!g->opt.disableMaskedStoreOptimizations) {
            optPM.add(CreateIntrinsicsOptPass());
            optPM.add(CreateMaskedStoreOptPass());
        }
        optPM.add(CreateLowerMaskedStorePass());
        if (!g->opt.disableGatherScatterOptimizations)
            optPM.add(CreateGatherScatterImprovementsPass());
        optPM.add(CreateLowerMaskedStorePass());
        optPM.add(CreateLowerGatherScatterPass());
        optPM.add(llvm::createFunctionInliningPass());
        optPM.add(llvm::createConstantPropagationPass());
        optPM.add(CreateIntrinsicsOptPass());

#if defined(LLVM_2_8)
        optPM.add(CreateIsCompileTimeConstantPass(true));
#elif defined(LLVM_2_9)
        llvm::createStandardModulePasses(&optPM, 3, 
                                         false /* opt size */,
                                         true /* unit at a time */, 
                                         false /* unroll loops */,
                                         true /* simplify lib calls */,
                                         false /* may have exceptions */,
                                         llvm::createFunctionInliningPass());
        llvm::createStandardLTOPasses(&optPM, true /* internalize pass */,
                                      true /* inline once again */,
                                      false /* verify after each pass */);
        llvm::createStandardFunctionPasses(&optPM, 3);

        optPM.add(CreateIsCompileTimeConstantPass(true));
        optPM.add(CreateIntrinsicsOptPass());

        llvm::createStandardModulePasses(&optPM, 3, 
                                         false /* opt size */,
                                         true /* unit at a time */, 
                                         false /* unroll loops */,
                                         true /* simplify lib calls */,
                                         false /* may have exceptions */,
                                         llvm::createFunctionInliningPass());
#else
        llvm::PassManagerBuilder builder;
        builder.OptLevel = 3;
        builder.Inliner = llvm::createFunctionInliningPass();
        builder.populateFunctionPassManager(funcPM);
        builder.populateModulePassManager(optPM);
        optPM.add(CreateIsCompileTimeConstantPass(true));
        optPM.add(CreateIntrinsicsOptPass());
        builder.populateLTOPassManager(optPM, true /* internalize */,
                                       true /* inline once again */);
        optPM.add(CreateIsCompileTimeConstantPass(true));
        optPM.add(CreateIntrinsicsOptPass());
        builder.populateModulePassManager(optPM);
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
        lPrintModuleCode(module);
    }
}



///////////////////////////////////////////////////////////////////////////
// IntrinsicsOpt

/** This is a relatively simple optimization pass that does a few small
    optimizations that LLVM's x86 optimizer doesn't currently handle.
    (Specifically, MOVMSK of a constant can be replaced with the
    corresponding constant value, and a BLENDVPS with either an 'all on' or
    'all off' blend factor can be replaced with the corredponding value of
    one of the two operands.

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
llvm::RegisterPass<IntrinsicsOpt> sse("sse-constants", "Intrinsics Cleanup Pass");


IntrinsicsOpt::IntrinsicsOpt() 
    : BasicBlockPass(ID) {

    // All of the mask instructions we may encounter.  Note that even if
    // compiling for AVX, we may still encounter the regular 4-wide SSE
    // MOVMSK instruction.
    llvm::Function *sseMovmsk = 
        llvm::Intrinsic::getDeclaration(m->module, llvm::Intrinsic::x86_sse_movmsk_ps);
    maskInstructions.push_back(sseMovmsk);
    maskInstructions.push_back(m->module->getFunction("llvm.x86.avx.movmsk.ps"));
    maskInstructions.push_back(m->module->getFunction("llvm.x86.mic.mask16.to.int"));
    maskInstructions.push_back(m->module->getFunction("__movmsk"));

    // And all of the blend instructions
    blendInstructions.push_back(BlendInstruction(
        llvm::Intrinsic::getDeclaration(m->module, llvm::Intrinsic::x86_sse41_blendvps),
        0xf, 0, 1, 2));
    blendInstructions.push_back(BlendInstruction(
        m->module->getFunction("llvm.x86.avx.blendvps"), 0xff, 0, 1, 2));
    blendInstructions.push_back(BlendInstruction(
        m->module->getFunction("llvm.x86.mic.blend.ps"), 0xffff, 1, 2, 0));
}


/** Given an llvm::Value represinting a vector mask, see if the value is a
    constant.  If so, return the integer mask found by taking the high bits
    of the mask values in turn and concatenating them into a single integer.
    In other words, given the 4-wide mask: < 0xffffffff, 0, 0, 0xffffffff >, 
    we have 0b1001 = 9.

    @todo This will break if we ever do 32-wide compilation, in which case
    it don't be possible to distinguish between -1 for "don't know" and
    "known and all bits on".
 */
static int
lGetMask(llvm::Value *factor) {
    llvm::ConstantVector *cv = llvm::dyn_cast<llvm::ConstantVector>(factor);
    if (cv) {
        int mask = 0;
        llvm::SmallVector<llvm::Constant *, ISPC_MAX_NVEC> elements;
        cv->getVectorElements(elements);

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
                assert(ci != NULL);  // vs return -1 if NULL?
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
        // else we should be able to handle it above...
        assert(!llvm::isa<llvm::Constant>(factor));
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
    bool modifiedAny = false;
 restart:
    for (llvm::BasicBlock::iterator i = bb.begin(), e = bb.end(); i != e; ++i) {
        llvm::CallInst *callInst = llvm::dyn_cast<llvm::CallInst>(&*i);
        if (!callInst)
            continue;

        BlendInstruction *blend = matchingBlendInstruction(callInst->getCalledFunction());
        if (blend != NULL) {
            llvm::Value *v[2] = { callInst->getArgOperand(blend->op0), 
                                  callInst->getArgOperand(blend->op1) };
            llvm::Value *factor = callInst->getArgOperand(blend->opFactor);

            // If the values are the same, then no need to blend..
            if (v[0] == v[1]) {
                llvm::ReplaceInstWithValue(i->getParent()->getInstList(), i, v[0]);
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
                llvm::ReplaceInstWithValue(i->getParent()->getInstList(), i, v[1]);
                modifiedAny = true;
                goto restart;
            }
            if (lIsUndef(v[1])) {
                llvm::ReplaceInstWithValue(i->getParent()->getInstList(), i, v[0]);
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
                llvm::ReplaceInstWithValue(i->getParent()->getInstList(), i, value);
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
                llvm::ReplaceInstWithValue(i->getParent()->getInstList(), i, value);
                modifiedAny = true;
                goto restart;
            }
        }
    }
    return modifiedAny;
}


bool
IntrinsicsOpt::matchesMaskInstruction(llvm::Function *function) {
    for (unsigned int i = 0; i < maskInstructions.size(); ++i)
        if (function == maskInstructions[i].function)
            return true;
    return false;
}


IntrinsicsOpt::BlendInstruction *
IntrinsicsOpt::matchingBlendInstruction(llvm::Function *function) {
    for (unsigned int i = 0; i < blendInstructions.size(); ++i)
        if (function == blendInstructions[i].function)
            return &blendInstructions[i];
    return NULL;
}


static llvm::Pass *
CreateIntrinsicsOptPass() {
    return new IntrinsicsOpt;
}


///////////////////////////////////////////////////////////////////////////
// GatherScatterFlattenOpt

/** When the front-end emits gathers and scatters, it generates an array of
    vector-width pointers to represent the set of addresses to read from or
    write to.  However, because ispc doesn't support pointers, it turns
    out to be the case that scatters and gathers always end up indexing
    into an array with a common base pointer.  Therefore, this optimization
    transforms the original arrays of general pointers into a single base
    pointer and an array of offsets.

    (Implementation seems to be easier with this approach versus having the
    front-end try to emit base pointer + offset stuff from the start,
    though arguably the latter approach would be a little more elegant.)

    See for example the comments discussing the __pseudo_gather functions
    in builtins.cpp for more information about this.

    @todo The implementation of this is pretty messy, and it sure would be
    nice to not have all the complexity of built-in assumptions of the
    structure of how the front end will have generated code, all of the
    instruction dyn_casts, etc.  Can we do something simpler, e.g. an early
    pass to flatten out GEPs when the size is known, then do LLVM's
    constant folding, then flatten into an array, etc.?
 */
class GatherScatterFlattenOpt : public llvm::BasicBlockPass {
public:
    static char ID;
    GatherScatterFlattenOpt() : BasicBlockPass(ID) { }

    const char *getPassName() const { return "Gather/Scatter Flattening"; }
    bool runOnBasicBlock(llvm::BasicBlock &BB);
};

char GatherScatterFlattenOpt::ID = 0;

llvm::RegisterPass<GatherScatterFlattenOpt> gsf("gs-flatten", "Gather/Scatter Flatten Pass");


/** Given an llvm::Value known to be an unsigned integer, return its value as
    an int64_t.
*/
static uint64_t
lGetIntValue(llvm::Value *offset) {
    llvm::ConstantInt *intOffset = llvm::dyn_cast<llvm::ConstantInt>(offset);
    assert(intOffset && (intOffset->getBitWidth() == 32 ||
                         intOffset->getBitWidth() == 64));
    return intOffset->getZExtValue();
}


/** Returns the size of the given llvm::Type as an llvm::Value, if the size
    can be easily determined at compile type.  If it's not easy to figure
    out the size, this just returns NULL and we handle finding its size
    differently.
 */
static bool
lSizeOfIfKnown(const llvm::Type *type, uint64_t *size) {
    if (type == LLVMTypes::Int8Type) {
        *size = 1;
        return true;
    }
    else if (type == LLVMTypes::Int16Type) {
        *size = 2;
        return true;
    }
    else if (type == LLVMTypes::FloatType || type == LLVMTypes::Int32Type) {
        *size = 4;
        return true;
    }
    else if (type == LLVMTypes::FloatVectorType || type == LLVMTypes::Int32VectorType) {
        *size = g->target.vectorWidth * 4;
        return true;
    }
    else if (type == LLVMTypes::DoubleType || type == LLVMTypes::Int64Type) {
        *size = 8;
        return true;
    }
    else if (type == LLVMTypes::DoubleVectorType || type == LLVMTypes::Int64VectorType) {
        *size = g->target.vectorWidth * 8;
        return true;
    }
    else if (llvm::isa<const llvm::ArrayType>(type)) {
        const llvm::ArrayType *at = llvm::dyn_cast<const llvm::ArrayType>(type);
        uint64_t eltSize;
        if (lSizeOfIfKnown(at->getElementType(), &eltSize)) {
            *size = eltSize * at->getNumElements();
            return true;
        }
        else
            return false;
    }
    return false;
}


/** This function returns an llvm::Value giving the size of the given type.
    If any instructions need to be generated to compute the size, they are
    inserted before insertBefore.
 */
static llvm::Value *
lSizeOf(LLVM_TYPE_CONST llvm::Type *type, llvm::Instruction *insertBefore) {
    // First try the easy case and see if we can get it as a simple
    // constant..
    uint64_t size;
    if (lSizeOfIfKnown(type, &size))
        return LLVMInt64(size);

    // Otherwise use the trick of doing a GEP with a NULL pointer to get
    // the pointer to the second element of an array of items of this type.
    // Then convert that pointer to an int and we've got the offset to the
    // second element in the array, a.k.a. the size of the type.
    LLVM_TYPE_CONST llvm::Type *ptrType = llvm::PointerType::get(type, 0);
    llvm::Value *nullPtr = llvm::Constant::getNullValue(ptrType);
    llvm::Value *index[1] = { LLVMInt32(1) };
    llvm::Value *poffset = llvm::GetElementPtrInst::Create(nullPtr, &index[0], &index[1],
                                                           "offset_ptr", insertBefore);
    lCopyMetadata(poffset, insertBefore);
    llvm::Instruction *inst = new llvm::PtrToIntInst(poffset, LLVMTypes::Int64Type, 
                                                     "offset_int", insertBefore);
    lCopyMetadata(inst, insertBefore);
    return inst;
}


/** This function returns a value that gives the offset in bytes from the
    start of the given structure type to the given struct member.  The
    instructions that compute this value are inserted before insertBefore.
 */
static llvm::Value *
lStructOffset(LLVM_TYPE_CONST llvm::Type *type, uint64_t member, 
              llvm::Instruction *insertBefore) {
    // Do a similar trick to the one done in lSizeOf above; compute the
    // pointer to the member starting from a NULL base pointer and then
    // cast that 'pointer' to an int...
    assert(llvm::isa<const llvm::StructType>(type));
    LLVM_TYPE_CONST llvm::Type *ptrType = llvm::PointerType::get(type, 0);
    llvm::Value *nullPtr = llvm::Constant::getNullValue(ptrType);
    llvm::Value *index[2] = { LLVMInt32(0), LLVMInt32((int32_t)member) };
    llvm::Value *poffset = llvm::GetElementPtrInst::Create(nullPtr, &index[0], &index[2],
                                                           "member_ptr", insertBefore);
    lCopyMetadata(poffset, insertBefore);
    llvm::Instruction *inst = new llvm::PtrToIntInst(poffset, LLVMTypes::Int64Type, 
                                                     "member_int", insertBefore);
    lCopyMetadata(inst, insertBefore);
    return inst;
}


static llvm::Value *
lGetTypeSize(LLVM_TYPE_CONST llvm::Type *type, llvm::Instruction *insertBefore) {
    LLVM_TYPE_CONST llvm::ArrayType *arrayType =  
        llvm::dyn_cast<LLVM_TYPE_CONST llvm::ArrayType>(type);
    if (arrayType != NULL)
        type = arrayType->getElementType();

    llvm::Value *scale = lSizeOf(type, insertBefore);
    llvm::Instruction *inst = new llvm::TruncInst(scale, LLVMTypes::Int32Type, "sizeof32", 
                                                  insertBefore);
    lCopyMetadata(inst, insertBefore);
    return inst;
}


static llvm::Value *
lGetOffsetForLane(int lane, llvm::Value *value, llvm::Value **offset, 
                  LLVM_TYPE_CONST llvm::Type **scaleType, bool *leafIsVarying,
                  llvm::Instruction *insertBefore) {
    if (!llvm::isa<llvm::GetElementPtrInst>(value)) {
        assert(llvm::isa<llvm::BitCastInst>(value));
        value = llvm::dyn_cast<llvm::BitCastInst>(value)->getOperand(0);

        llvm::ExtractValueInst *ev = llvm::dyn_cast<llvm::ExtractValueInst>(value);
        assert(ev->hasIndices() && ev->getNumIndices() == 1);
        assert(int(*(ev->idx_begin())) == lane);

        llvm::InsertValueInst *iv = llvm::dyn_cast<llvm::InsertValueInst>(ev->getOperand(0));
        assert(iv->hasIndices() && iv->getNumIndices() == 1);
        while (int(*(iv->idx_begin())) != lane) {
            iv = llvm::dyn_cast<llvm::InsertValueInst>(iv->getOperand(0));
            assert(iv && iv->hasIndices() && iv->getNumIndices() == 1);
        }
    
        value = iv->getOperand(1);
    }

    if (leafIsVarying != NULL) {
        LLVM_TYPE_CONST llvm::Type *pt = value->getType();
        LLVM_TYPE_CONST llvm::PointerType *ptrType = 
            llvm::dyn_cast<LLVM_TYPE_CONST llvm::PointerType>(pt);
        assert(ptrType);
        LLVM_TYPE_CONST llvm::Type *eltType = ptrType->getElementType();
        *leafIsVarying = llvm::isa<LLVM_TYPE_CONST llvm::VectorType>(eltType);
    }

    llvm::GetElementPtrInst *gep = llvm::dyn_cast<llvm::GetElementPtrInst>(value);
    assert(gep);

    assert(lGetIntValue(gep->getOperand(1)) == 0);
    LLVM_TYPE_CONST llvm::PointerType *targetPtrType = 
        llvm::dyn_cast<LLVM_TYPE_CONST llvm::PointerType>(gep->getOperand(0)->getType());
    assert(targetPtrType);
    LLVM_TYPE_CONST llvm::Type *targetType = targetPtrType->getElementType();

    if (llvm::isa<const llvm::StructType>(targetType)) {
        *offset = lStructOffset(targetType, lGetIntValue(gep->getOperand(2)),
                                insertBefore);
        *offset = new llvm::TruncInst(*offset, LLVMTypes::Int32Type, "member32", 
                                      insertBefore);
        lCopyMetadata(*offset, insertBefore);
        *scaleType = LLVMTypes::Int8Type; // aka char aka sizeof(1)
    }
    else {
        *offset = gep->getOperand(2);
        assert(*scaleType == NULL || *scaleType == targetType);
        *scaleType = targetType;
    }

    llvm::ExtractValueInst *ee = 
        llvm::dyn_cast<llvm::ExtractValueInst>(gep->getOperand(0));
    if (ee == NULL) {
        // found the base pointer, here it is...
        return gep->getOperand(0);
    }
    else {
        assert(ee->hasIndices() && ee->getNumIndices() == 1 &&
               int(*(ee->idx_begin())) == lane);
        llvm::InsertValueInst *iv =
            llvm::dyn_cast<llvm::InsertValueInst>(ee->getOperand(0));
        assert(iv != NULL);
        // do this chain of inserts for the next dimension...
        return iv;
    }
}


/** We have an LLVM array of pointer values, where each pointer has been
    computed with a GEP from some common base pointer value.  This function
    deconstructs the LLVM array, storing the offset from the base pointer
    as an llvm::Value for the i'th element into the i'th element of the
    offsets[] array passed in to the function.  It returns a scale factor
    for the offsets via *scaleType, and sets *leafIsVarying to true if the
    leaf data type being indexed into is a 'varying' ispc type.  The
    return value is either the base pointer or the an array of pointers for
    the next dimension of indexing (that we'll in turn deconstruct with
    this function).

    @todo All of the additional indexing magic for varying stuff should
    happen in the front end.
 */
static llvm::Value *
lTraverseInsertChain(llvm::Value *ptrs, llvm::Value *offsets[ISPC_MAX_NVEC],
                     LLVM_TYPE_CONST llvm::Type **scaleType, bool *leafIsVarying,
                     llvm::Instruction *insertBefore) {
    // This depends on the front-end constructing the arrays of pointers
    // via InsertValue instructions.  (Which it does do in
    // FunctionEmitContext::GetElementPtrInst()).
    llvm::InsertValueInst *ivInst = llvm::dyn_cast<llvm::InsertValueInst>(ptrs);
    assert(ivInst != NULL);

    // We have a chain of insert value instructions where each instruction
    // sets one of the elements of the array and where the input array is
    // either the base pointer or another insert value instruction.  Here
    // we talk through all of the insert value instructions until we hit
    // the end.
    llvm::Value *nextChain = NULL;
    while (ivInst != NULL) {
        // Figure out which array index this current instruction is setting
        // the value of.
        assert(ivInst->hasIndices() && ivInst->getNumIndices() == 1);
        int elementIndex = *(ivInst->idx_begin());
        assert(elementIndex >= 0 && elementIndex < g->target.vectorWidth);
        // We shouldn't have already seen something setting the value for
        // this index.
        assert(offsets[elementIndex] == NULL);

        // Set offsets[elementIndex] here.  This returns the value from
        // which the GEP operation was done; this should either be the base
        // pointer or an insert value chain for another dimension of the
        // array being indexed into.
        llvm::Value *myNext = lGetOffsetForLane(elementIndex, ivInst->getOperand(1), 
                                                &offsets[elementIndex], scaleType,
                                                leafIsVarying, insertBefore);
        if (nextChain == NULL)
            nextChain = myNext;
        else
            // All of these insert value instructions should have the same
            // base value
            assert(nextChain == myNext);

        // Do we have another element of the array to process?
        llvm::Value *nextInsert = ivInst->getOperand(0);
        ivInst = llvm::dyn_cast<llvm::InsertValueInst>(nextInsert);
        if (!ivInst)
            assert(llvm::isa<llvm::UndefValue>(nextInsert));
    }
    return nextChain;
}


/** Given a scalar value, return a vector of width g->target.vectorWidth
    that has the scalar replicated across each of its elements.

    @todo Using shufflevector to do this seems more idiomatic (and would be
    just a single instruction).  Switch to that?
 */
static llvm::Value *
lSmearScalar(llvm::Value *scalar, llvm::Instruction *insertBefore) {
    LLVM_TYPE_CONST llvm::Type *vectorType = llvm::VectorType::get(scalar->getType(), 
                                                                   g->target.vectorWidth);
    llvm::Value *result = llvm::UndefValue::get(vectorType);
    for (int i = 0; i < g->target.vectorWidth; ++i) {
        result = llvm::InsertElementInst::Create(result, scalar, LLVMInt32(i),
                                                 "smearinsert", insertBefore);
        lCopyMetadata(result, insertBefore);
    }
    return result;
}


static llvm::Value *
lGetPtrAndOffsets(llvm::Value *ptrs, llvm::Value **basePtr, 
                  llvm::Instruction *insertBefore, int eltSize) {
    llvm::Value *offset = LLVMInt32Vector(0);
    bool firstLoop = true, leafIsVarying;

    while (ptrs != NULL) {
        llvm::Value *offsets[ISPC_MAX_NVEC];
        for (int i = 0; i < g->target.vectorWidth; ++i)
            offsets[i] = NULL;
        LLVM_TYPE_CONST llvm::Type *scaleType = NULL;

        llvm::Value *nextChain = 
            lTraverseInsertChain(ptrs, offsets, &scaleType,
                                 firstLoop ? &leafIsVarying : NULL, insertBefore);

        for (int i = 0; i < g->target.vectorWidth; ++i)
            assert(offsets[i] != NULL);
        llvm::Value *delta = llvm::UndefValue::get(LLVMTypes::Int32VectorType);
        for (int i = 0; i < g->target.vectorWidth; ++i) {
            delta = llvm::InsertElementInst::Create(delta, offsets[i],
                                                    LLVMInt32(i), "dim",
                                                    insertBefore);
            lCopyMetadata(delta, insertBefore);
        }

        llvm::Value *size = lGetTypeSize(scaleType, insertBefore);

        llvm::Value *scale = lSmearScalar(size, insertBefore);
        delta = llvm::BinaryOperator::Create(llvm::Instruction::Mul, delta, 
                                             scale, "delta_scale", insertBefore);
        lCopyMetadata(delta, insertBefore);
        offset = llvm::BinaryOperator::Create(llvm::Instruction::Add, offset, 
                                              delta, "offset_delta", 
                                              insertBefore);
        lCopyMetadata(offset, insertBefore);

        if (llvm::dyn_cast<llvm::InsertValueInst>(nextChain))
            ptrs = nextChain;
        else {
            // else we don't have a unique starting pointer....
            assert(*basePtr == NULL || *basePtr == nextChain);
            *basePtr = nextChain;
            break;
        }
        firstLoop = false;
    }

    // handle varying stuff...
    if (leafIsVarying) {
        llvm::Value *deltaVector = llvm::UndefValue::get(LLVMTypes::Int32VectorType);
        for (int i = 0; i < g->target.vectorWidth; ++i) {
            deltaVector = 
                llvm::InsertElementInst::Create(deltaVector, LLVMInt32(eltSize*i),
                                                LLVMInt32(i), "delta", insertBefore);
            lCopyMetadata(deltaVector, insertBefore);
        }
        offset = llvm::BinaryOperator::Create(llvm::Instruction::Add, offset, 
                                              deltaVector, "offset_varying_delta", 
                                              insertBefore);
        lCopyMetadata(offset, insertBefore);
    }

    return offset;
}


bool
GatherScatterFlattenOpt::runOnBasicBlock(llvm::BasicBlock &bb) {
    llvm::Function *gather32Func = m->module->getFunction("__pseudo_gather_32");
    llvm::Function *gather64Func = m->module->getFunction("__pseudo_gather_64");
    llvm::Function *scatter32Func = m->module->getFunction("__pseudo_scatter_32");
    llvm::Function *scatter64Func = m->module->getFunction("__pseudo_scatter_64");
    assert(gather32Func && gather64Func && scatter32Func && scatter64Func);

    bool modifiedAny = false;
 restart:
    // Iterate through all of the instructions in the basic block.
    for (llvm::BasicBlock::iterator i = bb.begin(), e = bb.end(); i != e; ++i) {
        llvm::CallInst *callInst = llvm::dyn_cast<llvm::CallInst>(&*i);
        // If we don't have a call to one of the
        // __pseudo_{gather,scatter}_* functions, then just go on to the
        // next instruction.
        if (!callInst ||
            (callInst->getCalledFunction() != gather32Func &&
             callInst->getCalledFunction() != gather64Func &&
             callInst->getCalledFunction() != scatter32Func &&
             callInst->getCalledFunction() != scatter64Func))
            continue;

        bool isGather = (callInst->getCalledFunction() == gather32Func ||
                         callInst->getCalledFunction() == gather64Func);
        bool is32 = (callInst->getCalledFunction() == gather32Func ||
                     callInst->getCalledFunction() == scatter32Func);

        // Transform the array of pointers to a single base pointer and an
        // array of int32 offsets.  (All the hard work is done by
        // lGetPtrAndOffsets).
        llvm::Value *ptrs = callInst->getArgOperand(0);
        llvm::Value *basePtr = NULL;
        llvm::Value *offsetVector = lGetPtrAndOffsets(ptrs, &basePtr, callInst, 
                                                      is32 ? 4 : 8);
        // Cast the base pointer to a void *, since that's what the
        // __pseudo_*_base_offsets_* functions want.
        basePtr = new llvm::BitCastInst(basePtr, LLVMTypes::VoidPointerType, "base2void", 
                                        callInst);
        lCopyMetadata(basePtr, callInst);

        if (isGather) {
            llvm::Value *mask = callInst->getArgOperand(1);
            llvm::Function *gFunc = 
                m->module->getFunction(is32 ? "__pseudo_gather_base_offsets_32" :
                                              "__pseudo_gather_base_offsets_64");
            assert(gFunc != NULL);

            // Generate a new function call to the next pseudo gather
            // base+offsets instruction.  Note that we're passing a NULL
            // llvm::Instruction to llvm::CallInst::Create; this means that
            // the instruction isn't inserted into a basic block and that
            // way we can then call ReplaceInstWithInst().
            llvm::Value *newArgs[3] = { basePtr, offsetVector, mask };
#if defined(LLVM_3_0) || defined(LLVM_3_0svn)
            llvm::ArrayRef<llvm::Value *> newArgArray(&newArgs[0], &newArgs[3]);
            llvm::Instruction *newCall = 
                llvm::CallInst::Create(gFunc, newArgArray, "newgather", 
                                       (llvm::Instruction *)NULL);
#else
            llvm::Instruction *newCall = 
                llvm::CallInst::Create(gFunc, &newArgs[0], &newArgs[3], "newgather");
#endif
            lCopyMetadata(newCall, callInst);
            llvm::ReplaceInstWithInst(callInst, newCall);
        }
        else {
            llvm::Value *mask = callInst->getArgOperand(2);
            llvm::Value *rvalue = callInst->getArgOperand(1);
            llvm::Function *gFunc = 
                m->module->getFunction(is32 ? "__pseudo_scatter_base_offsets_32" :
                                              "__pseudo_scatter_base_offsets_64");
            assert(gFunc);

            // Generate a new function call to the next pseudo scatter
            // base+offsets instruction.  See above for why passing NULL
            // for the Instruction * is intended.
            llvm::Value *newArgs[4] = { basePtr, offsetVector, rvalue, mask };
#if defined(LLVM_3_0) || defined(LLVM_3_0svn)
            llvm::ArrayRef<llvm::Value *> newArgArray(&newArgs[0], &newArgs[4]);
            llvm::Instruction *newCall = 
                llvm::CallInst::Create(gFunc, newArgArray, "", 
                                       (llvm::Instruction *)NULL);
#else
            llvm::Instruction *newCall = 
                llvm::CallInst::Create(gFunc, &newArgs[0], &newArgs[4]);
#endif
            lCopyMetadata(newCall, callInst);
            llvm::ReplaceInstWithInst(callInst, newCall);
        }
        modifiedAny = true;
        goto restart;
    }
    return modifiedAny;
}


static llvm::Pass *
CreateGatherScatterFlattenPass() {
    return new GatherScatterFlattenOpt;
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

llvm::RegisterPass<MaskedStoreOptPass> mss("masked-store-scalarize",
                                           "Masked Store Scalarize Pass");

bool
MaskedStoreOptPass::runOnBasicBlock(llvm::BasicBlock &bb) {
    llvm::Function *pms32Func = m->module->getFunction("__pseudo_masked_store_32");
    llvm::Function *pms64Func = m->module->getFunction("__pseudo_masked_store_64");
    llvm::Function *msb32Func = m->module->getFunction("__masked_store_blend_32");
    llvm::Function *msb64Func = m->module->getFunction("__masked_store_blend_64");
    llvm::Function *ms32Func = m->module->getFunction("__masked_store_32");
    llvm::Function *ms64Func = m->module->getFunction("__masked_store_64");

    bool modifiedAny = false;
 restart:
    // Iterate over all of the instructions to look for one of the various
    // masked store functions
    for (llvm::BasicBlock::iterator i = bb.begin(), e = bb.end(); i != e; ++i) {
        llvm::CallInst *callInst = llvm::dyn_cast<llvm::CallInst>(&*i);
        if (!callInst)
            continue;

        llvm::Function *called = callInst->getCalledFunction();
        if (called != pms32Func && called != pms64Func &&
            called != msb32Func && called != msb64Func &&
            called != ms32Func  && called != ms64Func)
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
            // Need to update this when int8/int16 are added
            int align = (called == pms32Func || called == pms64Func ||
                         called == msb32Func) ? 4 : 8;

            lvalue = new llvm::BitCastInst(lvalue, ptrType, "lvalue_to_ptr_type", callInst);
            lCopyMetadata(lvalue, callInst);
            llvm::Instruction *store = 
                new llvm::StoreInst(rvalue, lvalue, false /* not volatile */,
                                    align);
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
// LowerMaskedStorePass

/** When the front-end needs to do a masked store, it emits a
    __pseudo_masked_store_{32,64} call as a placeholder.  This pass lowers
    these calls to either __masked_store_{32,64} or
    __masked_store_blend_{32,64} calls.
  */
class LowerMaskedStorePass : public llvm::BasicBlockPass {
public:
    static char ID;
    LowerMaskedStorePass() : BasicBlockPass(ID) { }

    const char *getPassName() const { return "Lower Masked Stores"; }
    bool runOnBasicBlock(llvm::BasicBlock &BB);
};


char LowerMaskedStorePass::ID = 0;

llvm::RegisterPass<LowerMaskedStorePass> lms("masked-store-lower",
                                             "Lower Masked Store Pass");


/** This routine attempts to determine if the given pointer in lvalue is
    pointing to stack-allocated memory.  It's conservative in that it
    should never return true for non-stack allocated memory, but may return
    false for memory that actually is stack allocated.  The basic strategy
    is to traverse through the operands and see if the pointer originally
    comes from an AllocaInst.
 */
static bool
lIsStackVariablePointer(llvm::Value *lvalue) {
    llvm::BitCastInst *bc = llvm::dyn_cast<llvm::BitCastInst>(lvalue);
    if (bc)
        return lIsStackVariablePointer(bc->getOperand(0));
    else {
        llvm::AllocaInst *ai = llvm::dyn_cast<llvm::AllocaInst>(lvalue);
        if (ai)
            return true;
        else {
            llvm::GetElementPtrInst *gep = llvm::dyn_cast<llvm::GetElementPtrInst>(lvalue);
            if (gep)
                return lIsStackVariablePointer(gep->getOperand(0));
            else
                return false;
        }
    }
}


/** Utilty routine to figure out which masked store function to use.  The
    blend parameter indicates if we want the blending version, is32
    indicates if the element size is 32 bits.
 */
static const char *
lMaskedStoreName(bool blend, bool is32) {
    if (blend) {
        if (is32)
            return "__masked_store_blend_32";
        else
            return "__masked_store_blend_64";
    }
    else {
        if (is32)
            return "__masked_store_32";
        else
            return "__masked_store_64";
    }
}


bool
LowerMaskedStorePass::runOnBasicBlock(llvm::BasicBlock &bb) {
    llvm::Function *maskedStore32Func = m->module->getFunction("__pseudo_masked_store_32");
    llvm::Function *maskedStore64Func = m->module->getFunction("__pseudo_masked_store_64");
    assert(maskedStore32Func && maskedStore64Func);

    bool modifiedAny = false;
 restart:
    for (llvm::BasicBlock::iterator i = bb.begin(), e = bb.end(); i != e; ++i) {
        // Iterate through all of the instructions and look for
        // __pseudo_masked_store_* calls.
        llvm::CallInst *callInst = llvm::dyn_cast<llvm::CallInst>(&*i);
        if (!callInst ||
            (callInst->getCalledFunction() != maskedStore32Func &&
             callInst->getCalledFunction() != maskedStore64Func))
            continue;

        bool is32 = (callInst->getCalledFunction() == maskedStore32Func);
        llvm::Value *lvalue = callInst->getArgOperand(0);
        llvm::Value *rvalue  = callInst->getArgOperand(1);
        llvm::Value *mask = callInst->getArgOperand(2);

        // On SSE, we need to choose between doing the load + blend + store
        // trick, or serializing the masked store.  On targets with a
        // native masked store instruction, the implementations of
        // __masked_store_blend_* should be the same as __masked_store_*,
        // so this doesn't matter.  On SSE, blending is generally more
        // efficient and is always safe to do on stack-allocated values.(?)
        bool doBlend = lIsStackVariablePointer(lvalue);
        if (g->target.isa == Target::SSE4 || g->target.isa == Target::SSE2)
            doBlend |= !g->opt.disableBlendedMaskedStores;

        // Generate the call to the appropriate masked store function and
        // replace the __pseudo_* one with it.
        llvm::Function *fms = m->module->getFunction(lMaskedStoreName(doBlend, is32));
        assert(fms);
        llvm::Value *args[3] = { lvalue, rvalue, mask };
#if defined(LLVM_3_0) || defined(LLVM_3_0svn)
        llvm::ArrayRef<llvm::Value *> newArgArray(&args[0], &args[3]);
        llvm::Instruction *inst = llvm::CallInst::Create(fms, newArgArray, "", 
                                                         callInst);
#else
        llvm::Instruction *inst = llvm::CallInst::Create(fms, &args[0], &args[3], "", 
                                                         callInst);
#endif
        lCopyMetadata(inst, callInst);

        callInst->eraseFromParent();
        modifiedAny = true;
        goto restart;
    }
    return modifiedAny;
}


static llvm::Pass *
CreateLowerMaskedStorePass() {
    return new LowerMaskedStorePass;
}

///////////////////////////////////////////////////////////////////////////
// GSImprovementsPass

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
class GSImprovementsPass : public llvm::BasicBlockPass {
public:
    static char ID;
    GSImprovementsPass() : BasicBlockPass(ID) { }

    const char *getPassName() const { return "Gather/Scatter Improvements"; }
    bool runOnBasicBlock(llvm::BasicBlock &BB);
};


char GSImprovementsPass::ID = 0;

llvm::RegisterPass<GSImprovementsPass> gsi("gs-improvements",
                                           "Gather/Scatter Improvements Pass");


#if 0
// Debugging routine: dump the values of all of the elmenets in a
// flattened-out vector
static void lPrintVector(const char *info, llvm::Value *elements[ISPC_MAX_NVEC]) {
    fprintf(stderr, "--- %s ---\n", info);
    for (int i = 0; i < g->target.vectorWidth; ++i) {
        fprintf(stderr, "%d: ", i);
        elements[i]->dump();
    }
    fprintf(stderr, "-----\n");
}
#endif


/** Given an LLVM vector in vec, return a 'scalarized' version of the
    vector in the provided offsets[] array.  For example, if the vector
    value passed in is:  

    add <4 x i32> %a_smear, <4 x i32> <4, 8, 12, 16>,

    and if %a_smear was computed by replicating a scalar value i32 %a
    across all of the elements of %a_smear, then the values returned will
    be:

    offsets[0] = add i32 %a, i32 4
    offsets[1] = add i32 %a, i32 8
    offsets[2] = add i32 %a, i32 12
    offsets[3] = add i32 %a, i32 16
    
    This function isn't fully general, but it seems to be able to handle
    all of the patterns that currently arise in practice.  If it can't
    scalarize a vector value, then it just returns false and the calling
    code proceeds as best it can without this information.

    @param vec               Vector to be scalarized
    @param scalarizedVector  Array in which to store the individual vector 
                             elements
    @returns                 True if the vector was successfully scalarized and
                             the values in offsets[] are valid; false otherwise
 */
static bool
lScalarizeVector(llvm::Value *vec, llvm::Value *scalarizedVector[ISPC_MAX_NVEC]) {
    // First initialize the values of scalarizedVector[] to NULL.
    for (int i = 0; i < g->target.vectorWidth; ++i)
        scalarizedVector[i] = NULL;

    // ConstantVectors are easy; just pull out the individual constant
    // element values
    llvm::ConstantVector *cv = llvm::dyn_cast<llvm::ConstantVector>(vec);
    if (cv != NULL) {
        for (int i = 0; i < g->target.vectorWidth; ++i)
            scalarizedVector[i] = cv->getOperand(i);
        return true;
    }

    // It's also easy if it's just a vector of all zeros
    llvm::ConstantAggregateZero *caz = llvm::dyn_cast<llvm::ConstantAggregateZero>(vec);
    if (caz) {
        for (int i = 0; i < g->target.vectorWidth; ++i)
            scalarizedVector[i] = LLVMInt32(0);
        return true;
    }

    llvm::BinaryOperator *bo = llvm::dyn_cast<llvm::BinaryOperator>(vec);
    if (bo) {
        // BinaryOperators are handled by attempting to scalarize both of
        // the operands.  If we're successful at this, then the vector of
        // scalar values we return from here are synthesized with scalar
        // versions of the original vector binary operator
        llvm::Instruction::BinaryOps opcode = bo->getOpcode();
        llvm::Value *v0[ISPC_MAX_NVEC], *v1[ISPC_MAX_NVEC];

        if (!lScalarizeVector(bo->getOperand(0), v0) || 
            !lScalarizeVector(bo->getOperand(1), v1))
            return false;

        for (int i = 0; i < g->target.vectorWidth; ++i) {
            scalarizedVector[i] = 
                llvm::BinaryOperator::Create(opcode, v0[i], v1[i], "flat_bop", bo);
            lCopyMetadata(scalarizedVector[i], bo);
        }

        return true;
    }

    llvm::InsertElementInst *ie = llvm::dyn_cast<llvm::InsertElementInst>(vec);
    if (ie != NULL) {
        // If we have an InsertElement instruction, we generally have a
        // chain along the lines of:
        //
        // %v0 = insertelement undef, value_0, i32 index_0
        // %v1 = insertelement %v1,   value_1, i32 index_1
        // ...
        // %vn = insertelement %vn-1, value_n-1, i32 index_n-1
        //
        // We start here witn %vn and work backwards through the chain of
        // insertelement instructions until we get to the undef value that
        // started it all.  At each instruction, we set the appropriate
        // vaue in scalarizedVector[] based on the value being inserted.
        while (ie != NULL) {
            uint64_t iOffset = lGetIntValue(ie->getOperand(2));
            assert((int)iOffset < g->target.vectorWidth);
            assert(scalarizedVector[iOffset] == NULL);

            scalarizedVector[iOffset] = ie->getOperand(1);

            llvm::Value *insertBase = ie->getOperand(0);
            ie = llvm::dyn_cast<llvm::InsertElementInst>(insertBase);
            if (!ie)
                assert(llvm::isa<llvm::UndefValue>(insertBase));
        }
        return true;
    }

    llvm::CastInst *ci = llvm::dyn_cast<llvm::CastInst>(vec);
    if (ci) {
        // Casts are similar to BinaryOperators in that we attempt to
        // scalarize the vector being cast and if successful, we apply
        // equivalent scalar cast operators to each of the values in the
        // scalarized vector.
        llvm::Instruction::CastOps op = ci->getOpcode();

        llvm::Value *scalarizedTarget[ISPC_MAX_NVEC];
        if (!lScalarizeVector(ci->getOperand(0), scalarizedTarget))
            return false;

        LLVM_TYPE_CONST llvm::Type *destType = ci->getDestTy();
        LLVM_TYPE_CONST llvm::VectorType *vectorDestType =
            llvm::dyn_cast<LLVM_TYPE_CONST llvm::VectorType>(destType);
        assert(vectorDestType != NULL);
        LLVM_TYPE_CONST llvm::Type *elementType = vectorDestType->getElementType();

        for (int i = 0; i < g->target.vectorWidth; ++i) {
            scalarizedVector[i] = 
                llvm::CastInst::Create(op, scalarizedTarget[i], elementType,
                                       "cast", ci);
            lCopyMetadata(scalarizedVector[i], ci);
        }
        return true;
    }

    llvm::ShuffleVectorInst *svi = llvm::dyn_cast<llvm::ShuffleVectorInst>(vec);
    if (svi) {
        // Note that the code for shufflevector instructions is untested.
        // (We haven't yet had a case where it needs to run).  Therefore,
        // an assert at the bottom of this routien will hit the first time
        // it runs as a reminder that this needs to be tested further.

        LLVM_TYPE_CONST llvm::VectorType *svInstType = 
            llvm::dyn_cast<LLVM_TYPE_CONST llvm::VectorType>(svi->getType());
        assert(svInstType != NULL);
        assert((int)svInstType->getNumElements() == g->target.vectorWidth);

        // Scalarize the two vectors being shuffled.  First figure out how
        // big they are.
        LLVM_TYPE_CONST llvm::Type *type0 = svi->getOperand(0)->getType();
        LLVM_TYPE_CONST llvm::Type *type1 = svi->getOperand(1)->getType();
        LLVM_TYPE_CONST llvm::VectorType *vectorType0 = 
            llvm::dyn_cast<LLVM_TYPE_CONST llvm::VectorType>(type0);
        LLVM_TYPE_CONST llvm::VectorType *vectorType1 = 
            llvm::dyn_cast<LLVM_TYPE_CONST llvm::VectorType>(type1);
        assert(vectorType0 != NULL && vectorType1 != NULL);

        int n0 = vectorType0->getNumElements();
        int n1 = vectorType1->getNumElements();

        // FIXME: It's actually totally legitimate for these two to have
        // different sizes; the final result just needs to have the native
        // vector width.  To handle this, not only do we need to
        // potentially dynamically allocate space for the arrays passed
        // into lScalarizeVector, but we need to change the rest of its
        // implementation to not key off g->target.vectorWidth everywhere
        // to get the sizes of the arrays to iterate over, etc.
        assert(n0 == g->target.vectorWidth && n1 == g->target.vectorWidth);

        // Go ahead and scalarize the two input vectors now.
        // FIXME: it's ok if some or all of the values of these two vectors
        // have undef values, so long as we don't try to access undef
        // values with the vector indices provided to the instruction.
        // Should fix lScalarizeVector so that it doesn't return false in
        // this case and just leaves the elements of the arrays with undef
        // values as NULL.
        llvm::Value *v0[ISPC_MAX_NVEC], *v1[ISPC_MAX_NVEC];
        if (!lScalarizeVector(svi->getOperand(0), v0) ||
            !lScalarizeVector(svi->getOperand(1), v1))
            return false;

        llvm::ConstantVector *shuffleIndicesVector = 
            llvm::dyn_cast<llvm::ConstantVector>(svi->getOperand(2));
        // I think this has to be a ConstantVector.  If this ever hits,
        // we'll dig into what we got instead and figure out how to handle
        // that...
        assert(shuffleIndicesVector != NULL);

        // Get the integer indices for each element of the returned vector
        llvm::SmallVector<llvm::Constant *, ISPC_MAX_NVEC> shuffleIndices;
        shuffleIndicesVector->getVectorElements(shuffleIndices);
        assert((int)shuffleIndices.size() == g->target.vectorWidth);

        // And loop over the indices, setting the i'th element of the
        // result vector with the source vector element that corresponds to
        // the i'th shuffle index value.
        for (unsigned int i = 0; i < shuffleIndices.size(); ++i) {
            if (!llvm::isa<llvm::ConstantInt>(shuffleIndices[i]))
                // I'm not sure when this case would ever happen, though..
                return false;
            int offset = (int)lGetIntValue(shuffleIndices[i]);
            assert(offset >= 0 && offset < n0+n1);

            if (offset < n0)
                // Offsets from 0 to n0-1 index into the first vector
                scalarizedVector[i] = v0[offset];
            else
                // And offsets from n0 to (n0+n1-1) index into the second
                // vector
                scalarizedVector[i] = v1[offset - n0];
        }
        FATAL("the above code is untested so far; check now that it's actually running");
        return true;
    }

#if 0
    fprintf(stderr, "flatten vector fixme\n");
    vec->dump();
    assert(0);
#endif

    return false;
}


/** Conservative test to see if two values are equal.  There are
    (potentially many) cases where the two values actually are equal but
    this will return false.  However, if it does return true, the two
    vectors definitely are equal.  

    @todo This seems to catch all of the cases we currently need it for in
    practice, but it's be nice to make it a little more robust/general.  In
    general, though, a little something called the halting problem means we
    won't get all of them.
 */
static bool
lValuesAreEqual(llvm::Value *v0, llvm::Value *v1) {
    // Thanks to the fact that LLVM hashes and returns the same pointer for
    // constants (of all sorts, even constant expressions), this first test
    // actually catches a lot of cases.  LLVM's SSA form also helps a lot
    // with this..
    if (v0 == v1)
        return true;

    llvm::BinaryOperator *bo0 = llvm::dyn_cast<llvm::BinaryOperator>(v0);
    llvm::BinaryOperator *bo1 = llvm::dyn_cast<llvm::BinaryOperator>(v1);
    if (bo0 && bo1) {
        if (bo0->getOpcode() != bo1->getOpcode())
            return false;
        return (lValuesAreEqual(bo0->getOperand(0), bo1->getOperand(0)) &&
                lValuesAreEqual(bo0->getOperand(1), bo1->getOperand(1)));
    }

    return false;
}


/** Tests to see if all of the llvm::Values in the array are equal.  Like
    lValuesAreEqual, this is a conservative test and may return false for
    arrays where the values are actually all equal.
 */
static bool
lVectorValuesAllEqual(llvm::Value *v[ISPC_MAX_NVEC]) {
    for (int i = 0; i < g->target.vectorWidth-1; ++i)
        if (!lValuesAreEqual(v[i], v[i+1]))
            return false;
    return true;
}


/** Given an array of scalar integer values, test to see if they are a
    linear sequence of compile-time constant integers starting from an
    arbirary value but then having a step of value "stride" between
    elements.
 */
static bool
lVectorIsLinearConstantInts(llvm::Value *v[ISPC_MAX_NVEC], int stride) {
    llvm::ConstantInt *prev = llvm::dyn_cast<llvm::ConstantInt>(v[0]);
    if (!prev)
        return false;
    int prevVal = (int)prev->getZExtValue();

    // For each element in the array, see if it is both a ConstantInt and
    // if the difference between it and the value of the previous element
    // is stride.  If not, fail.
    for (int i = 1; i < g->target.vectorWidth; ++i) {
        llvm::ConstantInt *next = llvm::dyn_cast<llvm::ConstantInt>(v[i]);
        if (!next) 
            return false;

        int nextVal = (int)next->getZExtValue();
        if (prevVal + stride != nextVal)
            return false;

        prevVal = nextVal;
    }
    return true;
}


/** Given an array of integer-typed values, see if the elements of the
    array have a step of 'stride' between their values.  This function
    tries to handle as many possibilities as possible, including things
    like all elements equal to some non-constant value plus an integer
    offset, etc.

    @todo FIXME Crazy thought: can we just build up expressions that
    subtract the constants [v[0], v[0]+stride, v[0]+2*stride, ...] from the
    given values, throw the LLVM optimizer at those, and then see if we get
    back an array of all zeros?
 */
static bool
lVectorIsLinear(llvm::Value *v[ISPC_MAX_NVEC], int stride) {
#if 0
    lPrintVector("called lVectorIsLinear", v);
#endif

    // First try the easy case: if the values are all just constant
    // integers and have the expected stride between them, then we're done.
    if (lVectorIsLinearConstantInts(v, stride))
        return true;

    // ConstantExprs need a bit of deconstruction to figure out

    // FIXME: do we need to handle cases where e.g. v[0] is an
    // llvm::ConstantInt and then the rest are ConstExprs??
    if (llvm::dyn_cast<llvm::ConstantExpr>(v[0])) {
        // First, see if all of the array elements are ConstantExprs of
        // some sort.  If not, give up.
        // FIXME: are we potentially missing cases here, e.g. a mixture of
        // ConstantExprs and ConstantInts?
        for (int i = 0; i < g->target.vectorWidth; ++i) {
            if (!llvm::isa<llvm::ConstantExpr>(v[i]))
                return false;
        }

        // See if any of the array elements are adds of constant
        // expressions.  As it turns out, LLVM's constant expression
        // optimizer is very thorough about converting "add(0, foo)" to
        // "foo", so we need to deal with cases where element 0 is "foo",
        // element 1 is add(4, foo), etc...
        bool anyAdds = false, allAdds = true;
        for (int i = 0; i < g->target.vectorWidth; ++i) {
            llvm::ConstantExpr *ce = llvm::dyn_cast<llvm::ConstantExpr>(v[i]);
            if (ce->getOpcode() == llvm::Instruction::Add)
                anyAdds = true;
            else 
                allAdds = false;
        }

        if (anyAdds && !allAdds) {
            // In v[], we should have an array of elements that are all
            // either ConstExprs with add operators, where one of the
            // operads is a constant int, or other non-add ConstExpr
            // values.  
            // 
            // Now we through each element and:
            // 1. For ones that aren't add ConstExprs, treat them as if they 
            //    are an add with 0 as the other operand.
            // 2. Extract the ConstInt operand of the add into the intBit[]
            //    array and put the other operand in the otherBit[] array.
            llvm::Value *intBit[ISPC_MAX_NVEC], *otherBit[ISPC_MAX_NVEC];
            for (int i = 0; i < g->target.vectorWidth; ++i) {
                llvm::ConstantExpr *ce = llvm::dyn_cast<llvm::ConstantExpr>(v[i]);
                if (ce->getOpcode() == llvm::Instruction::Add) {
                    // The ConstantInt may be either of the two operands of
                    // the add.  Put the operands in the right arrays.
                    if (llvm::isa<llvm::ConstantInt>(ce->getOperand(0))) {
                        intBit[i] = ce->getOperand(0);
                        otherBit[i] = ce->getOperand(1);
                    }
                    else {
                        intBit[i] = ce->getOperand(1);
                        otherBit[i] = ce->getOperand(0);
                    }
                }
                else {
                    // We don't have an Add, so pretend we have an add with
                    // zero.
                    intBit[i] = LLVMInt32(0);
                    otherBit[i] = v[i];
                }
            }

            // Now that everything is lined up, see if we have a case where
            // we're adding constant values with the desired stride to the
            // same base value.  If so, we know we have a linear set of
            // locations.
            return (lVectorIsLinear(intBit, stride) &&
                    lVectorValuesAllEqual(otherBit));
        }

        // If this ever hits, the assertion can just be commented out and
        // false returned below.  However, it's worth figuring out how the
        // analysis needs to be generalized rather than necessarily giving
        // up and possibly hurting performance of the final code.
        FATAL("Unexpected case with a ConstantExpr in lVectorIsLinear");
#if 0
        for (int i = 0; i < g->target.vectorWidth; ++i)
            v[i]->dump();
        FATAL("FIXME");
#endif
        return false;
    }


    llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(v[0]);
    if (bop) {
        // We also need to deal with non-constant binary operators that
        // represent linear accesses here..
        // FIXME: here, too, what about cases with v[0] being a load or something
        // and then everything after element 0 being a binary operator with an add.
        // That won't get caught by this case??
        bool anyAdd = false;
        for (int i = 0; i < g->target.vectorWidth; ++i) {
            llvm::BinaryOperator *bopi = llvm::dyn_cast<llvm::BinaryOperator>(v[i]);
            if (bopi && bopi->getOpcode() == llvm::Instruction::Add)
                anyAdd = true;
        }

        if (anyAdd) {
            // is one of the operands the same for all elements?  if so, then just
            // need to check this case for the other operand...

            // FIXME: do we need a more general check that starts with both
            // the first and second operand of v[0]'s add and then checks
            // the remainder of the elements to see if either one of their
            // two operands matches the one we started with?  That would be
            // more robust to switching the ordering of operands, in case
            // that ever happens...
            for (int operand = 0; operand <= 1; ++operand) {
                llvm::Value *addOperandValues[ISPC_MAX_NVEC];
                // Go through the vector elements and grab the operand'th
                // one if this is an add or the v
                for (int i = 0; i < g->target.vectorWidth; ++i) {
                    llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(v[i]);
                    if (bop->getOpcode() == llvm::Instruction::Add)
                        addOperandValues[i] = bop->getOperand(operand);
                    else
                        // The other guys are adds, so we'll treat this as
                        // an "add 0" in the below, so just grab the value
                        // v[i] itself
                        addOperandValues[i] = v[i];
                }

                if (lVectorValuesAllEqual(addOperandValues)) {
                    // If this operand's values are all equal, then the
                    // overall result is a linear sequence if the second
                    // operand's values are themselves a linear sequence...
                    int otherOperand = operand ^ 1;
                    for (int i = 0; i < g->target.vectorWidth; ++i) {
                        llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(v[i]);
                        if (bop->getOpcode() == llvm::Instruction::Add)
                            addOperandValues[i] = bop->getOperand(otherOperand);
                        else
                            addOperandValues[i] = LLVMInt32(0);
                    }
                    return lVectorIsLinear(addOperandValues, stride);
                }
            }
        }

        if (bop->getOpcode() == llvm::Instruction::Mul) {
            // Finally, if we have a multiply, then if one of the operands
            // has the same value for all elements and if the other operand
            // is a linear sequence such that the scale times the sequence
            // values is a linear sequence with the desired stride, then
            // we're good.
            llvm::ConstantInt *op0 = llvm::dyn_cast<llvm::ConstantInt>(bop->getOperand(0));
            llvm::ConstantInt *op1 = llvm::dyn_cast<llvm::ConstantInt>(bop->getOperand(1));

            // We need one of them to be a constant for us to be able to proceed...
            if (!op0 && !op1)
                return false;
            // But if they're both constants, then the LLVM constant folder
            // should have simplified them down to their product!
            assert(!(op0 && op1));

            // Figure out which operand number is the constant scale and
            // which is the varying one
            int scaleOperand, otherOperand;
            llvm::ConstantInt *scaleValue;
            if (op0 != NULL) {
                scaleOperand = 0;
                otherOperand = 1;
                scaleValue = op0;
            }
            else {
                scaleOperand = 1;
                otherOperand = 0;
                scaleValue = op1;
            }

            // Find the scale value; make sure it evenly divides the
            // stride.  Otherwise there's no chance that the scale times a
            // set of integer values will give a sequence with the desired
            // stride.
            int mulScale = (int)scaleValue->getZExtValue();
            if ((stride % mulScale) != 0)
                return false;

            llvm::Value *otherValue[ISPC_MAX_NVEC];
            for (int i = 0; i < g->target.vectorWidth; ++i) {
                llvm::BinaryOperator *eltBop = llvm::dyn_cast<llvm::BinaryOperator>(v[i]);
                // Give up if it's not matching the desired pattern of "all
                // mul ops with the scaleOperand being a constant with the
                // same value".
                if (!eltBop || eltBop->getOpcode() != llvm::Instruction::Mul)
                    return false;
                if (eltBop->getOperand(scaleOperand) != scaleValue)
                    return false;

                otherValue[i] = eltBop->getOperand(otherOperand);
            }
            // Now see if the sequence of values being scaled gives us
            // something with the desired stride.
            return lVectorIsLinear(otherValue, stride / mulScale);
        }
    }

    return false;
}


bool
GSImprovementsPass::runOnBasicBlock(llvm::BasicBlock &bb) {
    llvm::Function *gather32Func = m->module->getFunction("__pseudo_gather_base_offsets_32");
    llvm::Function *gather64Func = m->module->getFunction("__pseudo_gather_base_offsets_64");
    llvm::Function *scatter32Func = m->module->getFunction("__pseudo_scatter_base_offsets_32");
    llvm::Function *scatter64Func = m->module->getFunction("__pseudo_scatter_base_offsets_64");
    assert(gather32Func && gather64Func && scatter32Func && scatter64Func);

    bool modifiedAny = false;

 restart:
    for (llvm::BasicBlock::iterator i = bb.begin(), e = bb.end(); i != e; ++i) {
        // Iterate over all of the instructions and look for calls to
        // __pseudo_*_base_offsets_* calls.
        llvm::CallInst *callInst = llvm::dyn_cast<llvm::CallInst>(&*i);
        if (!callInst || 
            (callInst->getCalledFunction() != gather32Func &&
             callInst->getCalledFunction() != gather64Func &&
             callInst->getCalledFunction() != scatter32Func &&
             callInst->getCalledFunction() != scatter64Func))
            continue;

        SourcePos pos;
        bool ok = lGetSourcePosFromMetadata(callInst, &pos);
        assert(ok);     

        bool isGather = (callInst->getCalledFunction() == gather32Func ||
                         callInst->getCalledFunction() == gather64Func);
        bool is32 = (callInst->getCalledFunction() == gather32Func ||
                     callInst->getCalledFunction() == scatter32Func);

        // Get the actual base pointer; note that it comes into the gather
        // or scatter function bitcast to an i8 *, so we need to work back
        // to get the pointer as the original type.
        llvm::Value *base = callInst->getArgOperand(0);
        llvm::BitCastInst *bci = llvm::dyn_cast<llvm::BitCastInst>(base);
        if (bci)
            base = bci->getOperand(0);
        llvm::ConstantExpr *ce = llvm::dyn_cast<llvm::ConstantExpr>(base);
        if (ce && ce->getOpcode() == llvm::Instruction::BitCast)
            base = ce->getOperand(0);

        // Try to out the offsets; the i'th element of the offsetElements
        // array should be an i32 with the value of the offset for the i'th
        // vector lane.  This may fail; if so, just give up.
        llvm::Value *offsetElements[ISPC_MAX_NVEC];
        if (!lScalarizeVector(callInst->getArgOperand(1), offsetElements))
            continue;

        llvm::Value *mask = callInst->getArgOperand(isGather ? 2 : 3);

        if (lVectorValuesAllEqual(offsetElements)) {
            // If all the offsets are equal, then compute the single
            // pointer they all represent based on the first one of them
            // (arbitrarily).
            llvm::Value *indices[1] = { offsetElements[0] };
            llvm::Value *basei8 =
                new llvm::BitCastInst(base, LLVMTypes::VoidPointerType, "base2i8", callInst);
            lCopyMetadata(basei8, callInst);
            llvm::Value *ptr = 
                llvm::GetElementPtrInst::Create(basei8, &indices[0], &indices[1],
                                                "ptr", callInst);
            lCopyMetadata(ptr, callInst);

            if (isGather) {
                // A gather with everyone going to the same location is
                // handled as a scalar load and broadcast across the lanes.
                // Note that we do still have to pass the mask to the
                // __load_and_broadcast_* functions, since they shouldn't
                // access memory if the mask is all off (the location may
                // be invalid in that case).
                Debug(pos, "Transformed gather to scalar load and broadcast!");
                llvm::Function *loadBroadcast = 
                    m->module->getFunction(is32 ? "__load_and_broadcast_32" :
                                                  "__load_and_broadcast_64");
                assert(loadBroadcast);
                llvm::Value *args[2] = { ptr, mask };
#if defined(LLVM_3_0) || defined(LLVM_3_0svn)
                llvm::ArrayRef<llvm::Value *> newArgArray(&args[0], &args[2]);
                llvm::Instruction *newCall = 
                    llvm::CallInst::Create(loadBroadcast, newArgArray,
                                           "load_broadcast", (llvm::Instruction *)NULL);
#else
                llvm::Instruction *newCall = 
                    llvm::CallInst::Create(loadBroadcast, &args[0], &args[2],
                                           "load_broadcast");
#endif
                lCopyMetadata(newCall, callInst);
                llvm::ReplaceInstWithInst(callInst, newCall);
            }
            else {
                // A scatter with everyone going to the same location is
                // undefined.  Issue a warning and arbitrarily let the
                // first guy win.
                Warning(pos, "Undefined behavior: all program instances are "
                        "writing to the same location!");

                llvm::Value *rvalue = callInst->getArgOperand(2);
                llvm::Value *first = 
                    llvm::ExtractElementInst::Create(rvalue, LLVMInt32(0), "rvalue_first",
                                                     callInst);
                lCopyMetadata(first, callInst);
                ptr = new llvm::BitCastInst(ptr, llvm::PointerType::get(first->getType(), 0),
                                            "ptr2rvalue_type", callInst);
                lCopyMetadata(ptr, callInst);
                llvm::Instruction *sinst = 
                    new llvm::StoreInst(first, ptr, false, is32 ? 4 : 8 /* align */);
                lCopyMetadata(sinst, callInst);
                llvm::ReplaceInstWithInst(callInst, sinst);
            }

            modifiedAny = true;
            goto restart;
        }

        if (lVectorIsLinear(offsetElements, is32 ? 4 : 8)) {
            // We have a linear sequence of memory locations being accessed
            // starting with the location given by the offset from
            // offsetElements[0], with stride of 4 or 8 bytes (for 32 bit
            // and 64 bit gather/scatters, respectively.)

            // Get the base pointer using the first guy's offset.
            llvm::Value *indices[2] = { offsetElements[0] };
            llvm::Value *basei8 =
                new llvm::BitCastInst(base, LLVMTypes::VoidPointerType, "base2i8", callInst);
            lCopyMetadata(basei8, callInst);
            llvm::Value *ptr = 
                llvm::GetElementPtrInst::Create(basei8, &indices[0], &indices[1],
                                                "ptr", callInst);
            lCopyMetadata(ptr, callInst);

            if (isGather) {
                Debug(pos, "Transformed gather to unaligned vector load!");
                // FIXME: make this an aligned load when possible..
                // FIXME: are there lurking potential bugs when e.g. the
                // last few entries of the mask are off and the load ends
                // up straddling a page boundary?
                llvm::Function *loadMasked = 
                    m->module->getFunction(is32 ? "__load_masked_32" : "__load_masked_64");
                assert(loadMasked);

                llvm::Value *args[2] = { ptr, mask };
#if defined(LLVM_3_0) || defined(LLVM_3_0svn)
                llvm::ArrayRef<llvm::Value *> argArray(&args[0], &args[2]);
                llvm::Instruction *newCall = 
                    llvm::CallInst::Create(loadMasked, argArray, "load_masked",
                                           (llvm::Instruction *)NULL);
#else
                llvm::Instruction *newCall = 
                    llvm::CallInst::Create(loadMasked, &args[0], &args[2], "load_masked");
#endif
                lCopyMetadata(newCall, callInst);
                llvm::ReplaceInstWithInst(callInst, newCall);
            }
            else {
                Debug(pos, "Transformed scatter to unaligned vector store!");
                // FIXME: make this an aligned store when possible.  Need
                // to work through the messiness of issuing a pseudo store
                // here.
                llvm::Value *rvalue = callInst->getArgOperand(2);

                llvm::Function *storeMasked = 
                    m->module->getFunction(is32 ? "__pseudo_masked_store_32" :
                                                  "__pseudo_masked_store_64");
                assert(storeMasked);
                LLVM_TYPE_CONST llvm::Type *vecPtrType = is32 ?
                    LLVMTypes::Int32VectorPointerType : LLVMTypes::Int64VectorPointerType;
                ptr = new llvm::BitCastInst(ptr, vecPtrType, "ptrcast", callInst);

                llvm::Value *args[3] = { ptr, rvalue, mask };
#if defined(LLVM_3_0) || defined(LLVM_3_0svn)
                llvm::ArrayRef<llvm::Value *> argArray(&args[0], &args[3]);
                llvm::Instruction *newCall = 
                    llvm::CallInst::Create(storeMasked, argArray, "",
                                           (llvm::Instruction *)NULL);
#else
                llvm::Instruction *newCall = 
                    llvm::CallInst::Create(storeMasked, &args[0], &args[3], "");
#endif
                lCopyMetadata(newCall, callInst);
                llvm::ReplaceInstWithInst(callInst, newCall);
            }

            modifiedAny = true;
            goto restart;
        }

#if 0
        lPrintVector("scatter/gather no love: flattened", offsetElements);
        bb.dump();
#endif
    }

    return modifiedAny;
}


static llvm::Pass *
CreateGatherScatterImprovementsPass() {
    return new GSImprovementsPass;
}


///////////////////////////////////////////////////////////////////////////
// LowerGSPass

/** For any gathers and scatters remaining after the GSImprovementsPass
    runs, we need to turn them into actual native gathers and scatters.
    This task is handled by the LowerGSPass here.
 */
class LowerGSPass : public llvm::BasicBlockPass {
public:
    static char ID;
    LowerGSPass() : BasicBlockPass(ID) { }

    const char *getPassName() const { return "Gather/Scatter Improvements"; }
    bool runOnBasicBlock(llvm::BasicBlock &BB);
};


char LowerGSPass::ID = 0;

llvm::RegisterPass<LowerGSPass> lgs("lower-gs",
                                    "Lower Gather/Scatter Pass");

bool
LowerGSPass::runOnBasicBlock(llvm::BasicBlock &bb) {
    llvm::Function *gather32Func = m->module->getFunction("__pseudo_gather_base_offsets_32");
    llvm::Function *gather64Func = m->module->getFunction("__pseudo_gather_base_offsets_64");
    llvm::Function *scatter32Func = m->module->getFunction("__pseudo_scatter_base_offsets_32");
    llvm::Function *scatter64Func = m->module->getFunction("__pseudo_scatter_base_offsets_64");
    assert(gather32Func && gather64Func && scatter32Func && scatter64Func);

    bool modifiedAny = false;
 restart:
    for (llvm::BasicBlock::iterator i = bb.begin(), e = bb.end(); i != e; ++i) {
        // Loop over the instructions and find calls to the
        // __pseudo_*_base_offsets_* functions.
        llvm::CallInst *callInst = llvm::dyn_cast<llvm::CallInst>(&*i);
        if (!callInst || 
            (callInst->getCalledFunction() != gather32Func &&
             callInst->getCalledFunction() != gather64Func &&
             callInst->getCalledFunction() != scatter32Func &&
             callInst->getCalledFunction() != scatter64Func))
            continue;

        bool isGather = (callInst->getCalledFunction() == gather32Func ||
                         callInst->getCalledFunction() == gather64Func);
        bool is32 = (callInst->getCalledFunction() == gather32Func ||
                     callInst->getCalledFunction() == scatter32Func);

        // Get the source position from the metadata attached to the call
        // instruction so that we can issue PerformanceWarning()s below.
        SourcePos pos;
        bool ok = lGetSourcePosFromMetadata(callInst, &pos);
        assert(ok);     

        if (isGather) {
            llvm::Function *gFunc = m->module->getFunction(is32 ? "__gather_base_offsets_i32" :
                                                                  "__gather_base_offsets_i64");
            assert(gFunc);
            callInst->setCalledFunction(gFunc);
            PerformanceWarning(pos, "Gather required to compute value in expression.");
        }
        else {
            llvm::Function *sFunc = m->module->getFunction(is32 ? "__scatter_base_offsets_i32" :
                                                                  "__scatter_base_offsets_i64");
            assert(sFunc);
            callInst->setCalledFunction(sFunc);
            PerformanceWarning(pos, "Scatter required for storing value.");
        }
        modifiedAny = true;
        goto restart;
    }
    return modifiedAny;
}


static llvm::Pass *
CreateLowerGatherScatterPass() {
    return new LowerGSPass;
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

llvm::RegisterPass<IsCompileTimeConstantPass> 
    ctcrp("compile-time-constant", "Compile-Time Constant Resolve Pass");

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
            if (callInst->getCalledFunction() == funcs[j]) 
                break;
        }
        if (j == nFuncs)
            // not a __is_compile_time_constant_* function
            continue;

        // This optimization pass can be disabled with the (poorly named)
        // disableGatherScatterFlattening option.
        if (g->opt.disableGatherScatterFlattening) {
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

llvm::RegisterPass<MakeInternalFuncsStaticPass> 
  mifsp("make-internal-funcs-static", "Make Internal Funcs Static Pass");

bool
MakeInternalFuncsStaticPass::runOnModule(llvm::Module &module) {
    const char *names[] = {
        "__do_print", "__gather_base_offsets_i32", "__gather_base_offsets_i64",
        "__gather_elt_32", "__gather_elt_64", "__load_and_broadcast_32", 
        "__load_and_broadcast_64", "__load_masked_32", "__load_masked_64",
        "__masked_store_32", "__masked_store_64", "__masked_store_blend_32",
        "__masked_store_blend_64", "__packed_load_active", "__packed_store_active",
        "__scatter_base_offsets_i32", "__scatter_base_offsets_i64", "__scatter_elt_32",
        "__scatter_elt_64", };

    int count = sizeof(names) / sizeof(names[0]);
    for (int i = 0; i < count; ++i) {
        llvm::Function *f = m->module->getFunction(names[i]);
        if (f != NULL)
            f->setLinkage(llvm::GlobalValue::PrivateLinkage);
    }

    return true;
}


static llvm::Pass *
CreateMakeInternalFuncsStaticPass() {
    return new MakeInternalFuncsStaticPass;
}
