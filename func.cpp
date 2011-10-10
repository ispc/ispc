/*
  Copyright (c) 2011, Intel Corporation
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

/** @file func.cpp
    @brief 
*/

#include "func.h"
#include "ctx.h"
#include "decl.h"
#include "expr.h"
#include "llvmutil.h"
#include "module.h"
#include "type.h"
#include "stmt.h"
#include "sym.h"
#include "util.h"
#include <stdio.h>

#include <llvm/LLVMContext.h>
#include <llvm/Module.h>
#include <llvm/Type.h>
#include <llvm/DerivedTypes.h>
#include <llvm/Instructions.h>
#include <llvm/Intrinsics.h>
#include <llvm/PassManager.h>
#include <llvm/PassRegistry.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Support/FormattedStream.h>
#include <llvm/Support/FileUtilities.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Target/TargetData.h>
#include <llvm/PassManager.h>
#include <llvm/Analysis/Verifier.h>
#include <llvm/Support/CFG.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Assembly/PrintModulePass.h>

Function::Function(DeclSpecs *ds, Declarator *decl, Stmt *c) {
    code = c;

    maskSymbol = m->symbolTable->LookupVariable("__mask");
    assert(maskSymbol != NULL);

    if (code) {
        code = code->TypeCheck();
        if (code)
            code = code->Optimize();
    }

    if (g->debugPrint) {
        printf("Add Function\n");
        ds->Print();
        printf("\n");
        decl->Print();
        printf("\n");
        code->Print(0);
        printf("\n\n\n");
    }

    // Get the symbol for the function from the symbol table.  (It should
    // already have been added to the symbol table by AddGlobal() by the
    // time we get here.)
    type = dynamic_cast<const FunctionType *>(decl->GetType(ds));
    assert(type != NULL);
    sym = m->symbolTable->LookupFunction(decl->sym->name.c_str(), type);
    assert(sym != NULL);
    sym->pos = decl->pos;

    isExported = (ds->storageClass == SC_EXPORT);

    if (decl->functionArgs != NULL) {
        for (unsigned int i = 0; i < decl->functionArgs->size(); ++i) {
            Declaration *pdecl = (*decl->functionArgs)[i];
            assert(pdecl->declarators.size() == 1);
            Symbol *sym = pdecl->declarators[0]->sym;
            if (dynamic_cast<const ReferenceType *>(sym->type) == NULL)
                sym->parentFunction = this;
            args.push_back(sym);
        }
    }

    if (type->isTask) {
        threadIndexSym = m->symbolTable->LookupVariable("threadIndex");
        assert(threadIndexSym);
        threadCountSym = m->symbolTable->LookupVariable("threadCount");
        assert(threadCountSym);
        taskIndexSym = m->symbolTable->LookupVariable("taskIndex");
        assert(taskIndexSym);
        taskCountSym = m->symbolTable->LookupVariable("taskCount");
        assert(taskCountSym);
    }
    else
        threadIndexSym = threadCountSym = taskIndexSym = taskCountSym = NULL;
}


/** Given an arbitrary type, see if it or any of the types contained in it
    are varying.  Returns true if so, false otherwise. 
*/
static bool
lRecursiveCheckVarying(const Type *t) {
    t = t->GetBaseType();
    if (t->IsVaryingType()) return true;

    const StructType *st = dynamic_cast<const StructType *>(t);
    if (st) {
        for (int i = 0; i < st->GetElementCount(); ++i)
            if (lRecursiveCheckVarying(st->GetElementType(i)))
                return true;
    }
    return false;
}


/** Given a Symbol representing a function parameter, see if it or any
    contained types are varying.  If so, issue an error.  (This function
    should only be called for parameters to 'export'ed functions, where
    varying parameters is illegal.
 */
static void
lCheckForVaryingParameter(Symbol *sym) {
    if (lRecursiveCheckVarying(sym->type)) {
        const Type *t = sym->type->GetBaseType();
        if (dynamic_cast<const StructType *>(t))
            Error(sym->pos, "Struct parameter \"%s\" with varying member(s) is illegal "
                  "in an exported function.",
                  sym->name.c_str());
        else
            Error(sym->pos, "Varying parameter \"%s\" is illegal in an exported function.",
                  sym->name.c_str());
    }
}


/** Given a function type, loop through the function parameters and see if
    any are StructTypes.  If so, issue an error (this seems to be broken
    currently).

    @todo Fix passing structs from C/C++ to ispc functions.
 */
static void
lCheckForStructParameters(const FunctionType *ftype, SourcePos pos) {
    const std::vector<const Type *> &argTypes = ftype->GetArgumentTypes();
    for (unsigned int i = 0; i < argTypes.size(); ++i) {
        const Type *type = argTypes[i];
        if (dynamic_cast<const StructType *>(type) != NULL) {
            Error(pos, "Passing structs to/from application functions is currently broken. "
                  "Use a reference or const reference instead for now.");
            return;
        }
    }
}


/** We've got a declaration for a function to process.  This function does
    all the work of creating the corresponding llvm::Function instance,
    adding the symbol for the function to the symbol table and doing
    various sanity checks.  This function returns true upon success and
    false if any errors were encountered.
 */
Symbol *
Function::InitFunctionSymbol(DeclSpecs *ds, Declarator *decl) {
    // Make sure that we've got what we expect here
    Symbol *funSym = decl->sym;
    assert(decl->isFunction);
    assert(decl->arraySize.size() == 0);

    // So far, so good.  Go ahead and set the type of the function symbol
    funSym->type = decl->GetType(ds);

    // If a global variable with the same name has already been declared
    // issue an error.
    if (m->symbolTable->LookupVariable(funSym->name.c_str()) != NULL) {
        Error(decl->pos, "Function \"%s\" shadows previously-declared global variable. "
              "Ignoring this definition.",
              funSym->name.c_str());
        return NULL;
    }

    if (ds->storageClass == SC_EXTERN_C) {
        // Make sure the user hasn't supplied both an 'extern "C"' and a
        // 'task' qualifier with the function
        if (ds->typeQualifier & TYPEQUAL_TASK) {
            Error(funSym->pos, "\"task\" qualifier is illegal with C-linkage extern "
                  "function \"%s\".  Ignoring this function.", funSym->name.c_str());
            return NULL;
        }
        std::vector<Symbol *> *funcs;
        funcs = m->symbolTable->LookupFunction(decl->sym->name.c_str());
        if (funcs != NULL) {
            if (funcs->size() > 1) {
                // Multiple functions with this name have already been declared; 
                // can't overload here
                Error(funSym->pos, "Can't overload extern \"C\" function \"%s\"; "
                      "%d functions with the same name have already been declared.",
                      funSym->name.c_str(), (int)funcs->size());
                return NULL;
            }

            // One function with the same name has been declared; see if it
            // has the same type as this one, in which case it's ok.
            if (Type::Equal((*funcs)[0]->type, funSym->type))
                return (*funcs)[0];
            else {
                Error(funSym->pos, "Can't overload extern \"C\" function \"%s\".",
                      funSym->name.c_str());
                return NULL;
            }
        }
    }

    // We should have gotten a FunctionType back from the GetType() call above.
    const FunctionType *functionType = 
        dynamic_cast<const FunctionType *>(funSym->type);
    assert(functionType != NULL);

    // Get the LLVM FunctionType
    bool includeMask = (ds->storageClass != SC_EXTERN_C);
    LLVM_TYPE_CONST llvm::FunctionType *llvmFunctionType = 
        functionType->LLVMFunctionType(g->ctx, includeMask);
    if (llvmFunctionType == NULL)
        return NULL;

    // And create the llvm::Function
    llvm::GlobalValue::LinkageTypes linkage = (ds->storageClass == SC_STATIC ||
                                               (ds->typeQualifier & TYPEQUAL_INLINE)) ?
        llvm::GlobalValue::InternalLinkage : llvm::GlobalValue::ExternalLinkage;
    std::string functionName = ((ds->storageClass == SC_EXTERN_C) ?
                                funSym->name : funSym->MangledName());
    if (g->mangleFunctionsWithTarget)
        functionName += g->target.GetISAString();
    llvm::Function *function = 
        llvm::Function::Create(llvmFunctionType, linkage, functionName.c_str(), m->module);

    // Set function attributes: we never throw exceptions, and want to
    // inline everything we can
    function->setDoesNotThrow(true);
    if (!(ds->storageClass == SC_EXTERN_C) && !g->generateDebuggingSymbols &&
        (ds->typeQualifier & TYPEQUAL_INLINE))
        function->addFnAttr(llvm::Attribute::AlwaysInline);
    if (functionType->isTask)
        // This also applies transitively to members I think? 
        function->setDoesNotAlias(1, true);

    // Make sure that the return type isn't 'varying' if the function is
    // 'export'ed.
    if (ds->storageClass == SC_EXPORT && 
        lRecursiveCheckVarying(functionType->GetReturnType()))
        Error(decl->pos, "Illegal to return a \"varying\" type from exported function \"%s\"",
              funSym->name.c_str());

    if (functionType->isTask && (functionType->GetReturnType() != AtomicType::Void))
        Error(funSym->pos, "Task-qualified functions must have void return type.");

    if (functionType->isExported || functionType->isExternC)
        lCheckForStructParameters(functionType, funSym->pos);

    // Loop over all of the arguments; process default values if present
    // and do other checks and parameter attribute setting.
    bool seenDefaultArg = false;
    std::vector<ConstExpr *> argDefaults;
    int nArgs = decl->functionArgs ? decl->functionArgs->size() : 0;
    for (int i = 0; i < nArgs; ++i) {
        Declaration *pdecl = (*decl->functionArgs)[i];
        assert(pdecl->declarators.size() == 1);
        Symbol *sym = pdecl->declarators[0]->sym;

        // If the function is exported, make sure that the parameter
        // doesn't have any varying stuff going on in it.
        if (ds->storageClass == SC_EXPORT)
            lCheckForVaryingParameter(sym);

        // ISPC assumes that all memory passed in is aligned to the native
        // width and that no pointers alias.  (It should be possible to
        // specify when this is not the case, but this should be the
        // default.)  Set parameter attributes accordingly.
        if (!functionType->isTask && dynamic_cast<const ReferenceType *>(sym->type) != NULL) {
            // NOTE: LLVM indexes function parameters starting from 1.
            // This is unintuitive.
            function->setDoesNotAlias(i+1, true);
            int align = 4 * RoundUpPow2(g->target.nativeVectorWidth);
            function->addAttribute(i+1, llvm::Attribute::constructAlignmentFromInt(align));
        }

        if (m->symbolTable->LookupFunction(sym->name.c_str()) != NULL)
            Warning(sym->pos, "Function parameter \"%s\" shadows a function "
                    "declared in global scope.", sym->name.c_str());
        
        // See if a default argument value was provided with the parameter
        Expr *defaultValue = pdecl->declarators[0]->initExpr;
        if (defaultValue != NULL) {
            // If we have one, make sure it's a compile-time constant
            seenDefaultArg = true;
            defaultValue = defaultValue->TypeCheck();
            defaultValue = defaultValue->Optimize();
            defaultValue = dynamic_cast<ConstExpr *>(defaultValue);
            if (!defaultValue) {
                Error(sym->pos, "Default value for parameter \"%s\" must be "
                      "a compile-time constant.", sym->name.c_str());
                return NULL;
            }
        }
        else if (seenDefaultArg) {
            // Once one parameter has provided a default value, then all of
            // the following ones must have them as well.
            Error(sym->pos, "Parameter \"%s\" is missing default: all parameters after "
                  "the first parameter with a default value must have default values "
                  "as well.", sym->name.c_str());
        }

        // Add the default value to argDefaults.  Note that we make this
        // call for all parameters, even those where no default value was
        // provided.  In that case, a NULL value is stored here.  This
        // approach means that we can always just look at the i'th entry of
        // argDefaults to find the default value for the i'th parameter.
        argDefaults.push_back(dynamic_cast<ConstExpr *>(defaultValue));
    }

    // And only now can we set the default values in the FunctionType
    functionType->SetArgumentDefaults(argDefaults);

    // If llvm gave us back a Function * with a different name than the one
    // we asked for, then there's already a function with that same
    // (mangled) name in the llvm::Module.  In that case, erase the one we
    // tried to add and just work with the one it already had.
    if (function->getName() != functionName) {
        function->eraseFromParent();
        function = m->module->getFunction(functionName);
    }
    funSym->function = function;

    // But if that function has a definition, we don't want to redefine it.
    if (!function->empty()) {
        Warning(funSym->pos, "Ignoring redefinition of function \"%s\".", 
                funSym->name.c_str());
        return NULL;
    }

    // Finally, we know all is good and we can add the function to the
    // symbol table
    bool ok = m->symbolTable->AddFunction(funSym);
    assert(ok);
    return funSym;
}


const Type *
Function::GetReturnType() const {
    return type->GetReturnType();
}


const FunctionType *
Function::GetType() const {
    return type;
}


/** Parameters for tasks are stored in a big structure; this utility
    function emits code to copy those values out of the task structure into
    local stack-allocated variables.  (Which we expect that LLVM's
    'mem2reg' pass will in turn promote to SSA registers..
 */
static void
lCopyInTaskParameter(int i, llvm::Value *structArgPtr, const std::vector<Symbol *> &args,
                     FunctionEmitContext *ctx) {
    // We expect the argument structure to come in as a poitner to a
    // structure.  Confirm and figure out its type here.
    const llvm::Type *structArgType = structArgPtr->getType();
    assert(llvm::isa<llvm::PointerType>(structArgType));
    const llvm::PointerType *pt = llvm::dyn_cast<const llvm::PointerType>(structArgType);
    assert(llvm::isa<llvm::StructType>(pt->getElementType()));
    const llvm::StructType *argStructType = 
        llvm::dyn_cast<const llvm::StructType>(pt->getElementType());

    // Get the type of the argument we're copying in and its Symbol pointer
    LLVM_TYPE_CONST llvm::Type *argType = argStructType->getElementType(i);
    Symbol *sym = args[i];

    // allocate space to copy the parameter in to
    sym->storagePtr = ctx->AllocaInst(argType, sym->name.c_str());

    // get a pointer to the value in the struct
    llvm::Value *ptr = ctx->GetElementPtrInst(structArgPtr, 0, i, sym->name.c_str());

    // and copy the value from the struct and into the local alloca'ed
    // memory
    llvm::Value *ptrval = ctx->LoadInst(ptr, NULL, sym->name.c_str());
    ctx->StoreInst(ptrval, sym->storagePtr);
    ctx->EmitFunctionParameterDebugInfo(sym);
}


/** Given the statements implementing a function, emit the code that
    implements the function.  Most of the work do be done here just
    involves wiring up the function parameter values to be available in the
    function body code.
 */
void 
Function::emitCode(FunctionEmitContext *ctx, llvm::Function *function, 
                   SourcePos firstStmtPos) {
    llvm::Value *maskPtr = ctx->AllocaInst(LLVMTypes::MaskType, "mask_memory");
    ctx->StoreInst(LLVMMaskAllOn, maskPtr);
    maskSymbol->storagePtr = maskPtr;
    ctx->SetMaskPointer(maskPtr);

    // add debugging info for __mask, programIndex, ...
    maskSymbol->pos = firstStmtPos;
    ctx->EmitVariableDebugInfo(maskSymbol);

#if 0
    llvm::BasicBlock *entryBBlock = ctx->GetCurrentBasicBlock();
#endif
    if (type->isTask == true) {
        // For tasks, we there should always be three parmeters: the
        // pointer to the structure that holds all of the arguments, the
        // thread index, and the thread count variables.
        llvm::Function::arg_iterator argIter = function->arg_begin();
        llvm::Value *structParamPtr = argIter++;
        llvm::Value *threadIndex = argIter++;
        llvm::Value *threadCount = argIter++;
        llvm::Value *taskIndex = argIter++;
        llvm::Value *taskCount = argIter++;

        // Copy the function parameter values from the structure into local
        // storage
        for (unsigned int i = 0; i < args.size(); ++i)
            lCopyInTaskParameter(i, structParamPtr, args, ctx);

        // Copy in the mask as well.
        int nArgs = (int)args.size();
        // The mask is the last parameter in the argument structure
        llvm::Value *ptr = ctx->GetElementPtrInst(structParamPtr, 0, nArgs,
                                                  "task_struct_mask");
        llvm::Value *ptrval = ctx->LoadInst(ptr, NULL, "mask");
        ctx->SetFunctionMask(ptrval);

        // Copy threadIndex and threadCount into stack-allocated storage so
        // that their symbols point to something reasonable.
        threadIndexSym->storagePtr = ctx->AllocaInst(LLVMTypes::Int32Type, "threadIndex");
        ctx->StoreInst(threadIndex, threadIndexSym->storagePtr);

        threadCountSym->storagePtr = ctx->AllocaInst(LLVMTypes::Int32Type, "threadCount");
        ctx->StoreInst(threadCount, threadCountSym->storagePtr);

        // Copy taskIndex and taskCount into stack-allocated storage so
        // that their symbols point to something reasonable.
        taskIndexSym->storagePtr = ctx->AllocaInst(LLVMTypes::Int32Type, "taskIndex");
        ctx->StoreInst(taskIndex, taskIndexSym->storagePtr);

        taskCountSym->storagePtr = ctx->AllocaInst(LLVMTypes::Int32Type, "taskCount");
        ctx->StoreInst(taskCount, taskCountSym->storagePtr);
    }
    else {
        // Regular, non-task function
        llvm::Function::arg_iterator argIter = function->arg_begin(); 
        for (unsigned int i = 0; i < args.size(); ++i, ++argIter) {
            Symbol *sym = args[i];
            argIter->setName(sym->name.c_str());

            // Allocate stack storage for the parameter and emit code
            // to store the its value there.
            sym->storagePtr = ctx->AllocaInst(argIter->getType(), sym->name.c_str());
            ctx->StoreInst(argIter, sym->storagePtr);
            ctx->EmitFunctionParameterDebugInfo(sym);
        }

        // If the number of actual function arguments is equal to the
        // number of declared arguments in decl->functionArgs, then we
        // don't have a mask parameter, so set it to be all on.  This
        // happens for exmaple with 'export'ed functions that the app
        // calls.
        if (argIter == function->arg_end())
            ctx->SetFunctionMask(LLVMMaskAllOn);
        else {
            // Otherwise use the mask to set the entry mask value
            argIter->setName("__mask");
            assert(argIter->getType() == LLVMTypes::MaskType);
            ctx->SetFunctionMask(argIter);
            assert(++argIter == function->arg_end());
        }
    }

    // Finally, we can generate code for the function
    if (code != NULL) {
        int costEstimate = code->EstimateCost();
        bool checkMask = (type->isTask == true) || 
            ((function->hasFnAttr(llvm::Attribute::AlwaysInline) == false) &&
             costEstimate > CHECK_MASK_AT_FUNCTION_START_COST);
        Debug(code->pos, "Estimated cost for function \"%s\" = %d\n", 
              sym->name.c_str(), costEstimate);
        // If the body of the function is non-trivial, then we wrap the
        // entire thing around a varying "cif (true)" test in order to reap
        // the side-effect benefit of checking to see if the execution mask
        // is all on and thence having a specialized code path for that
        // case.  If this is a simple function, then this isn't worth the
        // code bloat / overhead.
        if (checkMask) {
            bool allTrue[ISPC_MAX_NVEC];
            for (int i = 0; i < g->target.vectorWidth; ++i)
                allTrue[i] = true;
            Expr *trueExpr = new ConstExpr(AtomicType::VaryingBool, allTrue, 
                                           code->pos);
            code = new IfStmt(trueExpr, code, NULL, true, code->pos);
        }

        ctx->SetDebugPos(code->pos);
        ctx->AddInstrumentationPoint("function entry");
        code->EmitCode(ctx);
    }

    if (ctx->GetCurrentBasicBlock()) {
        // FIXME: We'd like to issue a warning if we've reached the end of
        // the function without a return statement (for non-void
        // functions).  But the test below isn't right, since we can have
        // (with 'x' a varying test) "if (x) return a; else return b;", in
        // which case we have a valid basic block but its unreachable so ok
        // to not have return statement.
#if 0
        // If the bblock has no predecessors, then it doesn't matter if it
        // doesn't have a return; it'll never be reached.  If it does,
        // issue a warning.  Also need to warn if it's the entry block for
        // the function (in which case it will not have predeccesors but is
        // still reachable.)
        if (type->GetReturnType() != AtomicType::Void &&
            (pred_begin(ec.bblock) != pred_end(ec.bblock) || (ec.bblock == entryBBlock)))
            Warning(sym->pos, "Missing return statement in function returning \"%s\".",
                    type->rType->GetString().c_str());
#endif

        // FIXME: would like to set the context's current position to
        // e.g. the end of the function code

        // if bblock is non-NULL, it hasn't been terminated by e.g. a
        // return instruction.  Need to add a return instruction.
        ctx->ReturnInst();
    }
}


void
Function::GenerateIR() {
    llvm::Function *function = sym->function;
    assert(function != NULL);

    // Figure out a reasonable source file position for the start of the
    // function body.  If possible, get the position of the first actual
    // non-StmtList statment...
    SourcePos firstStmtPos = sym->pos;
    if (code) {
        StmtList *sl = dynamic_cast<StmtList *>(code);
        if (sl && sl->GetStatements().size() > 0 && 
            sl->GetStatements()[0] != NULL)
            firstStmtPos = sl->GetStatements()[0]->pos;
        else
            firstStmtPos = code->pos;
    }

    // And we can now go ahead and emit the code 
    {
        FunctionEmitContext ec(this, sym, function, firstStmtPos);
        emitCode(&ec, function, firstStmtPos);
    }

    if (m->errorCount == 0) {
        if (llvm::verifyFunction(*function, llvm::ReturnStatusAction) == true) {
            if (g->debugPrint) {
                llvm::PassManager ppm;
                ppm.add(llvm::createPrintModulePass(&llvm::outs()));
                ppm.run(*m->module);
            }
            FATAL("Function verificication failed");
        }

        // If the function is 'export'-qualified, emit a second version of
        // it without a mask parameter and without name mangling so that
        // the application can call it
        if (isExported) {
            if (!type->isTask) {
                LLVM_TYPE_CONST llvm::FunctionType *ftype = 
                    type->LLVMFunctionType(g->ctx);
                llvm::GlobalValue::LinkageTypes linkage = llvm::GlobalValue::ExternalLinkage;
                std::string functionName = sym->name;
                if (g->mangleFunctionsWithTarget)
                    functionName += std::string("_") + g->target.GetISAString();
                llvm::Function *appFunction = 
                    llvm::Function::Create(ftype, linkage, functionName.c_str(), m->module);
                appFunction->setDoesNotThrow(true);

                if (appFunction->getName() != functionName) {
                    // this was a redefinition for which we already emitted an
                    // error, so don't worry about this one...
                    appFunction->eraseFromParent();
                }
                else {
                    // And emit the code again
                    FunctionEmitContext ec(this, sym, appFunction, firstStmtPos);
                    emitCode(&ec, appFunction, firstStmtPos);
                    if (m->errorCount == 0) {
                        sym->exportedFunction = appFunction;
                        if (llvm::verifyFunction(*appFunction, 
                                                 llvm::ReturnStatusAction) == true) {
                            if (g->debugPrint) {
                                llvm::PassManager ppm;
                                ppm.add(llvm::createPrintModulePass(&llvm::outs()));
                                ppm.run(*m->module);
                            }
                            FATAL("Function verificication failed");
                        }
                    }
                }
            }
        }
    }
}
