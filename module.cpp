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

/** @file module.cpp
    @brief Impementation of the Module class, which collects the result of compiling
           a source file and then generates output (object files, etc.)
*/

#include "module.h"
#include "util.h"
#include "ctx.h"
#include "builtins.h"
#include "decl.h"
#include "type.h"
#include "expr.h"
#include "sym.h"
#include "stmt.h"
#include "opt.h"
#include "llvmutil.h"

#include <stdio.h>
#include <assert.h>
#include <stdarg.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <algorithm>
#include <set>
#ifdef ISPC_IS_WINDOWS
#include <windows.h>
#include <io.h>
#define strcasecmp stricmp
#endif

#include <llvm/LLVMContext.h>
#include <llvm/Module.h>
#include <llvm/Type.h>
#include <llvm/DerivedTypes.h>
#include <llvm/Instructions.h>
#include <llvm/Intrinsics.h>
#include <llvm/Support/FormattedStream.h>
#include <llvm/Support/FileUtilities.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetRegistry.h>
#include <llvm/Target/TargetSelect.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Target/TargetData.h>
#if !defined(LLVM_3_0) && !defined(LLVM_3_0svn)
#include <llvm/Target/SubtargetFeature.h>
#endif // !LLVM_3_0
#include <llvm/PassManager.h>
#include <llvm/Analysis/Verifier.h>
#include <llvm/Support/CFG.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/Utils.h>
#include <clang/Basic/TargetInfo.h>
#ifndef LLVM_2_8
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Support/Host.h>
#else // !LLVM_2_8
#include <llvm/System/Host.h>
#endif // LLVM_2_8
#include <llvm/Assembly/PrintModulePass.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Bitcode/ReaderWriter.h>

///////////////////////////////////////////////////////////////////////////
// Module

Module::Module(const char *fn) {
    // FIXME: It's a hack to do this here, but it must be done after the
    // target information has been set (so e.g. the vector width is
    // known...)
    InitLLVMUtil(g->ctx, g->target);

    filename = fn;
    errorCount = 0;
    symbolTable = new SymbolTable;
    module = new llvm::Module(filename ? filename : "<stdin>", *g->ctx);

    // initialize target in module
    llvm::InitializeAllTargets();

    llvm::Triple triple;
    // Start with the host triple as the default
    triple.setTriple(llvm::sys::getHostTriple());
    if (g->target.arch != "") {
        // If the user specified a target architecture, see if it's a known
        // one; print an error with the valid ones otherwise.
        const llvm::Target *target = NULL;
        for (llvm::TargetRegistry::iterator iter = llvm::TargetRegistry::begin();
             iter != llvm::TargetRegistry::end(); ++iter) {
            if (g->target.arch == iter->getName()) {
                target = &*iter;
                break;
            }
        }
        if (!target) {
            fprintf(stderr, "Invalid target \"%s\"\nOptions: ", 
                    g->target.arch.c_str());
            llvm::TargetRegistry::iterator iter;
            for (iter = llvm::TargetRegistry::begin();
                 iter != llvm::TargetRegistry::end(); ++iter)
                fprintf(stderr, "%s ", iter->getName());
            fprintf(stderr, "\n");
            exit(1);
        }

        // And override the arch in the host triple
        llvm::Triple::ArchType archType = 
            llvm::Triple::getArchTypeForLLVMName(g->target.arch);
        if (archType != llvm::Triple::UnknownArch)
            triple.setArch(archType);
    }
    module->setTargetTriple(triple.str());


#ifndef LLVM_2_8
    if (g->generateDebuggingSymbols)
        diBuilder = new llvm::DIBuilder(*module);
    else
        diBuilder = NULL;
#endif // LLVM_2_8

#ifndef LLVM_2_8
    // If we're generating debugging symbols, let the DIBuilder know that
    // we're starting a new compilation unit.
    if (diBuilder != NULL) {
        if (filename == NULL) {
            // Unfortunately we can't yet call Error() since the global 'm'
            // variable hasn't been initialized yet.
            fprintf(stderr, "Can't emit debugging information with no "
                    "source file on disk.\n");
            ++errorCount;
            delete diBuilder;
            diBuilder = NULL;
        }
        else {
            std::string directory, name;
            GetDirectoryAndFileName(g->currentDirectory, filename, &directory,
                                    &name);
            diBuilder->createCompileUnit(llvm::dwarf::DW_LANG_C99,  /* lang */
                                         name,  /* filename */
                                         directory, /* directory */
                                         "ispc", /* producer */
                                         g->opt.level > 0 /* is optimized */,
                                         "-g", /* command line args */
                                         0 /* run time version */);
        }
    }
#endif // LLVM_2_8
}


extern FILE *yyin;
extern int yyparse();
typedef struct yy_buffer_state *YY_BUFFER_STATE;
extern void yy_switch_to_buffer(YY_BUFFER_STATE);
extern YY_BUFFER_STATE yy_scan_string(const char *);
extern YY_BUFFER_STATE yy_create_buffer(FILE *, int);
extern void yy_delete_buffer(YY_BUFFER_STATE);

int
Module::CompileFile() {
    // FIXME: it'd be nice to do this in the Module constructor, but this
    // function ends up calling into routines that expect the global
    // variable 'm' to be initialized and available (which it isn't until
    // the Module constructor returns...)
    DefineStdlib(symbolTable, g->ctx, module, g->includeStdlib);

    bool runPreprocessor = g->runCPP;

    if (runPreprocessor) {
        if (filename != NULL) {
            // Try to open the file first, since otherwise we crash in the
            // preprocessor if the file doesn't exist.
            FILE *f = fopen(filename, "r");
            if (!f) {
                perror(filename);
                return 1;
            }
            fclose(f);
        }

        std::string buffer;
        llvm::raw_string_ostream os(buffer);
        execPreprocessor((filename != NULL) ? filename : "-", &os);
        YY_BUFFER_STATE strbuf = yy_scan_string(os.str().c_str());
        yyparse();
        yy_delete_buffer(strbuf);
    }
    else {
        // No preprocessor, just open up the file if it's not stdin..
        FILE* f = NULL;
        if (filename == NULL) 
            f = stdin;
        else {
            f = fopen(filename, "r");
            if (f == NULL) {
                perror(filename);
                return 1;
            }
        }
        yyin = f;
        yy_switch_to_buffer(yy_create_buffer(yyin, 4096));
        yyparse();
        fclose(f);
    }

    if (errorCount == 0)
        Optimize(module, g->opt.level);

    return errorCount;
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
static bool
lInitFunSymDecl(DeclSpecs *ds, Declarator *decl) {
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
        return false;
    }

    if (ds->storageClass == SC_EXTERN_C) {
        // Make sure the user hasn't supplied both an 'extern "C"' and a
        // 'task' qualifier with the function
        if (ds->typeQualifier & TYPEQUAL_TASK) {
            Error(funSym->pos, "\"task\" qualifier is illegal with C-linkage extern "
                  "function \"%s\".  Ignoring this function.", funSym->name.c_str());
            return false;
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
                return false;
            }

            // One function with the same name has been declared; see if it
            // has the same type as this one, in which case it's ok.
            if (Type::Equal((*funcs)[0]->type, funSym->type))
                return true;
            else {
                Error(funSym->pos, "Can't overload extern \"C\" function \"%s\".",
                      funSym->name.c_str());
                return false;
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
        return false;

    // And create the llvm::Function
    llvm::GlobalValue::LinkageTypes linkage = ds->storageClass == SC_STATIC ?
        llvm::GlobalValue::InternalLinkage : llvm::GlobalValue::ExternalLinkage;
    std::string functionName = ((ds->storageClass == SC_EXTERN_C) ?
                                funSym->name : funSym->MangledName());
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
    if (ds->storageClass == SC_EXPORT && lRecursiveCheckVarying(functionType->GetReturnType()))
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
                return false;
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
        return false;
    }

    // Finally, we know all is good and we can add the function to the
    // symbol table
    bool ok = m->symbolTable->AddFunction(funSym);
    assert(ok);
    return true;
}


void
Module::AddGlobal(DeclSpecs *ds, Declarator *decl) {
    // This function is called for a number of cases: function
    // declarations, typedefs, and global variables declarations /
    // definitions.  Figure out what we've got and take care of it.

    if (decl->isFunction) {
        // function declaration
        const Type *t = decl->GetType(ds);
        const FunctionType *ft = dynamic_cast<const FunctionType *>(t);
        assert(ft != NULL);
        if (m->symbolTable->LookupFunction(decl->sym->name.c_str(), ft) != NULL)
            // Ignore redeclaration of a function with the same name and type
            return;
        // Otherwise do all of the llvm Module and SymbolTable work..
        lInitFunSymDecl(ds, decl);
    }
    else if (ds->storageClass == SC_TYPEDEF) {
        // Typedefs are easy; just add the mapping between the given name
        // and the given type.
        m->symbolTable->AddType(decl->sym->name.c_str(), decl->sym->type,
                                decl->sym->pos);
    }
    else {
        // global variable
        if (m->symbolTable->LookupFunction(decl->sym->name.c_str()) != NULL) {
            Error(decl->pos, "Global variable \"%s\" shadows previously-declared function.",
                  decl->sym->name.c_str());
            return;
        }

        // These may be NULL due to errors in parsing; just gracefully
        // return here if so.
        if (!decl->sym || !decl->sym->type) {
            // But if these are NULL and there haven't been any previous
            // errors, something surprising is going on
            assert(errorCount > 0);
            return;
        }

        if (ds->storageClass == SC_EXTERN_C) {
            Error(decl->pos, "extern \"C\" qualifier can only be used for functions.");
            return;
        }

        LLVM_TYPE_CONST llvm::Type *llvmType = decl->sym->type->LLVMType(g->ctx);
        llvm::GlobalValue::LinkageTypes linkage =
            (ds->storageClass == SC_STATIC) ? llvm::GlobalValue::InternalLinkage :
                                              llvm::GlobalValue::ExternalLinkage;

        // See if we have an initializer expression for the global.  If so,
        // make sure it's a compile-time constant!
        llvm::Constant *llvmInitializer = NULL;
        if (ds->storageClass == SC_EXTERN || ds->storageClass == SC_EXTERN_C) {
            externGlobals.push_back(decl->sym);
            if (decl->initExpr != NULL)
                Error(decl->pos, "Initializer can't be provided with \"extern\" "
                      "global variable \"%s\".", decl->sym->name.c_str());
        }
        else {
            if (decl->initExpr != NULL) {
                decl->initExpr = decl->initExpr->TypeCheck();
                if (decl->initExpr != NULL) {
                    // We need to make sure the initializer expression is
                    // the same type as the global.  (But not if it's an
                    // ExprList; they don't have types per se / can't type
                    // convert themselves anyway.)
                    if (dynamic_cast<ExprList *>(decl->initExpr) == NULL)
                        decl->initExpr = 
                            decl->initExpr->TypeConv(decl->sym->type, "initializer");

                    if (decl->initExpr != NULL) {
                        decl->initExpr = decl->initExpr->Optimize();
                        // Fingers crossed, now let's see if we've got a
                        // constant value..
                        llvmInitializer = decl->initExpr->GetConstant(decl->sym->type);

                        if (llvmInitializer != NULL) {
                            if (decl->sym->type->IsConstType())
                                // Try to get a ConstExpr associated with
                                // the symbol.  This dynamic_cast can
                                // validly fail, for example for types like
                                // StructTypes where a ConstExpr can't
                                // represent their values.
                                decl->sym->constValue = 
                                    dynamic_cast<ConstExpr *>(decl->initExpr);
                        }
                        else
                            Error(decl->pos, "Initializer for global variable \"%s\" "
                                  "must be a constant.", decl->sym->name.c_str());
                    }
                }
            }

            // If no initializer was provided or if we couldn't get a value
            // above, initialize it with zeros..
            if (llvmInitializer == NULL)
                llvmInitializer = llvm::Constant::getNullValue(llvmType);
        }

        bool isConst = (ds->typeQualifier & TYPEQUAL_CONST) != 0;
        decl->sym->storagePtr = new llvm::GlobalVariable(*module, llvmType, isConst, 
                                                         linkage, llvmInitializer, 
                                                         decl->sym->name.c_str());
        m->symbolTable->AddVariable(decl->sym);

#ifndef LLVM_2_8
        if (diBuilder && (ds->storageClass != SC_EXTERN)) {
            llvm::DIFile file = decl->pos.GetDIFile();
            diBuilder->createGlobalVariable(decl->sym->name, 
                                            file,
                                            decl->pos.first_line,
                                            decl->sym->type->GetDIType(file),
                                            (ds->storageClass == SC_STATIC),
                                            decl->sym->storagePtr);
        }
#endif // LLVM_2_8
    }
}


/** Parameters for tasks are stored in a big structure; this utility
    function emits code to copy those values out of the task structure into
    local stack-allocated variables.  (Which we expect that LLVM's
    'mem2reg' pass will in turn promote to SSA registers..
 */
static void
lCopyInTaskParameter(int i, llvm::Value *structArgPtr, Declarator *decl, 
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
    Declaration *pdecl = (*decl->functionArgs)[i];
    assert(pdecl->declarators.size() == 1);
    Symbol *sym = pdecl->declarators[0]->sym;

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
static void 
lEmitFunctionCode(FunctionEmitContext *ctx, llvm::Function *function, 
                  const FunctionType *ft, Symbol *funSym,
                  Declarator *decl, Stmt *code) {
#if 0
    llvm::BasicBlock *entryBBlock = ctx->GetCurrentBasicBlock();
#endif
    if (ft->isTask == true) {
        // For tasks, we there should always be three parmeters: the
        // pointer to the structure that holds all of the arguments, the
        // thread index, and the thread count variables.
        llvm::Function::arg_iterator argIter = function->arg_begin();
        llvm::Value *structParamPtr = argIter++;
        llvm::Value *threadIndex = argIter++;
        llvm::Value *threadCount = argIter++;

        // Copy the function parameter values from the structure into local
        // storage
        if (decl->functionArgs)
            for (unsigned int i = 0; i < decl->functionArgs->size(); ++i)
                lCopyInTaskParameter(i, structParamPtr, decl, ctx);

        // Copy in the mask as well.
        int nArgs = decl->functionArgs ? decl->functionArgs->size() : 0;
        // The mask is the last parameter in the argument structure
        llvm::Value *ptr = ctx->GetElementPtrInst(structParamPtr, 0, nArgs,
                                                  "task_struct_mask");
        llvm::Value *ptrval = ctx->LoadInst(ptr, NULL, "mask");
        ctx->SetEntryMask(ptrval);

        // Copy threadIndex and threadCount into stack-allocated storage so
        // that their symbols point to something reasonable.
        Symbol *threadIndexSym = m->symbolTable->LookupVariable("threadIndex");
        assert(threadIndexSym);
        threadIndexSym->storagePtr = ctx->AllocaInst(LLVMTypes::Int32Type, "threadIndex");
        ctx->StoreInst(threadIndex, threadIndexSym->storagePtr);

        Symbol *threadCountSym = m->symbolTable->LookupVariable("threadCount");
        assert(threadCountSym);
        threadCountSym->storagePtr = ctx->AllocaInst(LLVMTypes::Int32Type, "threadCount");
        ctx->StoreInst(threadCount, threadCountSym->storagePtr);

#ifdef ISPC_IS_WINDOWS
        // On Windows, we dynamically-allocate space for the task arguments
        // (see FunctionEmitContext::LaunchInst().)  Here is where we emit
        // the code to free that memory, now that we've copied the
        // parameter values out of the structure.
        ctx->EmitFree(structParamPtr);
#endif // ISPC_IS_WINDOWS
    }
    else {
        // Regular, non-task function
        llvm::Function::arg_iterator argIter = function->arg_begin(); 
        if (decl->functionArgs) {
            for (unsigned int i = 0; i < decl->functionArgs->size(); ++i, ++argIter) {
                Declaration *pdecl = (*decl->functionArgs)[i];
                assert(pdecl->declarators.size() == 1);
                Symbol *sym = pdecl->declarators[0]->sym;
                argIter->setName(sym->name.c_str());

                // Allocate stack storage for the parameter and emit code
                // to store the its value there.
                sym->storagePtr = ctx->AllocaInst(argIter->getType(), sym->name.c_str());
                ctx->StoreInst(argIter, sym->storagePtr);
                ctx->EmitFunctionParameterDebugInfo(sym);
            }
        }

        // If the number of actual function arguments is equal to the
        // number of declared arguments in decl->functionArgs, then we
        // don't have a mask parameter, so set it to be all on.  This
        // happens for exmaple with 'export'ed functions that the app
        // calls.
        if (argIter == function->arg_end())
            ctx->SetEntryMask(LLVMMaskAllOn);
        else {
            // Otherwise use the mask to set the entry mask value
            argIter->setName("__mask");
            assert(argIter->getType() == LLVMTypes::MaskType);
            ctx->SetEntryMask(argIter);
            assert(++argIter == function->arg_end());
        }
    }

    // Finally, we can generate code for the function
    if (code != NULL) {
        bool checkMask = (ft->isTask == true) || 
            (function->hasFnAttr(llvm::Attribute::AlwaysInline) == false);
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
        if (ft->GetReturnType() != AtomicType::Void &&
            (pred_begin(ec.bblock) != pred_end(ec.bblock) || (ec.bblock == entryBBlock)))
            Warning(funSym->pos, "Missing return statement in function returning \"%s\".",
                    ft->rType->GetString().c_str());
#endif

        // FIXME: would like to set the context's current position to
        // e.g. the end of the function code

        // if bblock is non-NULL, it hasn't been terminated by e.g. a
        // return instruction.  Need to add a return instruction.
        ctx->ReturnInst();
    }
}


void
Module::AddFunction(DeclSpecs *ds, Declarator *decl, Stmt *code) {
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
    const FunctionType *functionType =
        dynamic_cast<const FunctionType *>(decl->GetType(ds));
    assert(functionType != NULL);
    Symbol *funSym = symbolTable->LookupFunction(decl->sym->name.c_str(), 
                                                 functionType);
    assert(funSym != NULL);
    funSym->pos = decl->pos;

    llvm::Function *function = funSym->function;
    assert(function != NULL);

    // Figure out a reasonable source file position for the start of the
    // function body.  If possible, get the position of the first actual
    // non-StmtList statment...
    SourcePos firstStmtPos = funSym->pos;
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
        FunctionEmitContext ec(functionType->GetReturnType(), function, funSym,
                               firstStmtPos);
        lEmitFunctionCode(&ec, function, functionType, funSym, decl, code);
    }

    if (errorCount == 0) {
        if (llvm::verifyFunction(*function, llvm::ReturnStatusAction) == true) {
            if (g->debugPrint) {
                llvm::PassManager ppm;
                ppm.add(llvm::createPrintModulePass(&llvm::outs()));
                ppm.run(*module);
            }
            FATAL("Function verificication failed");
        }

        // If the function is 'export'-qualified, emit a second version of
        // it without a mask parameter and without name mangling so that
        // the application can call it
        if (ds->storageClass == SC_EXPORT) {
            if (!functionType->isTask) {
                LLVM_TYPE_CONST llvm::FunctionType *ftype = 
                    functionType->LLVMFunctionType(g->ctx);
                llvm::GlobalValue::LinkageTypes linkage = llvm::GlobalValue::ExternalLinkage;
                llvm::Function *appFunction = 
                    llvm::Function::Create(ftype, linkage, funSym->name.c_str(), module);
                appFunction->setDoesNotThrow(true);

                if (appFunction->getName() != funSym->name) {
                    // this was a redefinition for which we already emitted an
                    // error, so don't worry about this one...
                    appFunction->eraseFromParent();
                }
                else {
                    // And emit the code again
                    FunctionEmitContext ec(functionType->GetReturnType(), appFunction, funSym,
                                           firstStmtPos);
                    lEmitFunctionCode(&ec, appFunction, functionType, funSym, decl, code);
                    if (errorCount == 0) {
                        if (llvm::verifyFunction(*appFunction, 
                                                 llvm::ReturnStatusAction) == true) {
                            if (g->debugPrint) {
                                llvm::PassManager ppm;
                                ppm.add(llvm::createPrintModulePass(&llvm::outs()));
                                ppm.run(*module);
                            }
                            FATAL("Function verificication failed");
                        }
                    }
                }
            }
        }
    }
}


bool
Module::WriteOutput(OutputType outputType, const char *outFileName) {
#if defined(LLVM_3_0) || defined(LLVM_3_0svn)
    if (diBuilder != NULL && outputType != Header)
        diBuilder->finalize();
#endif // LLVM_3_0

    // First, issue a warning if the output file suffix and the type of
    // file being created seem to mismatch.  This can help catch missing
    // command-line arguments specifying the output file type.
    const char *suffix = strrchr(outFileName, '.');
    if (suffix != NULL) {
        ++suffix;
        const char *fileType = NULL;
        switch (outputType) {
        case Asm:
            if (strcasecmp(suffix, "s"))
                fileType = "assembly";
            break;
        case Bitcode:
            if (strcasecmp(suffix, "bc"))
                fileType = "LLVM bitcode";
            break;
        case Object:
            if (strcasecmp(suffix, "o") && strcasecmp(suffix, "obj"))
                fileType = "object";
            break;
        case Header:
            if (strcasecmp(suffix, "h") && strcasecmp(suffix, "hh") &&
                strcasecmp(suffix, "hpp"))
                fileType = "header";
            break;
        }
        if (fileType != NULL)
            fprintf(stderr, "Warning: emitting %s file, but filename \"%s\" "
                    "has suffix \"%s\"?\n", fileType, outFileName, suffix);
    }

    if (outputType == Header)
        return writeHeader(outFileName);
    else {
        if (outputType == Bitcode) {
            // Get a file descriptor corresponding to where we want the output
            // to go.  If we open it, it'll be closed by the
            // llvm::raw_fd_ostream destructor.
            int fd;
            if (!strcmp(outFileName, "-"))
                fd = 1; // stdout
            else {
                int flags = O_CREAT|O_WRONLY|O_TRUNC;
#ifdef ISPC_IS_WINDOWS
                flags |= O_BINARY;
                fd = _open(outFileName, flags, 0644);
#else
                fd = open(outFileName, flags, 0644);
#endif // ISPC_IS_WINDOWS
                if (fd == -1) {
                    perror(outFileName);
                    return false;
                }
            }

            llvm::raw_fd_ostream fos(fd, (fd != 1), false);
            llvm::WriteBitcodeToFile(module, fos);
            return true;
        }
        else {
#ifdef LLVM_2_8
            fprintf(stderr, "Direct object file emission not supported in this build.\n");
            return false;
#else
            return writeObjectFileOrAssembly(outputType, outFileName);
#endif // LLVM_2_8
        }
    }
}


bool
Module::writeObjectFileOrAssembly(OutputType outputType, const char *outFileName) {
#if defined(LLVM_3_0) || defined(LLVM_3_0svn)
    llvm::InitializeAllTargetMCs();
#endif
    llvm::InitializeAllAsmPrinters();
    llvm::InitializeAllAsmParsers();

    llvm::Triple triple(module->getTargetTriple());
    assert(triple.getTriple().empty() == false);

    const llvm::Target *target = NULL;
    std::string error;
    target = llvm::TargetRegistry::lookupTarget(triple.getTriple(), error);
    assert(target != NULL);

    std::string featuresString;
    llvm::TargetMachine *targetMachine = NULL;
#if defined LLVM_3_0svn || defined LLVM_3_0
    if (g->target.isa == Target::AVX)
        featuresString = "+avx";
    targetMachine = target->createTargetMachine(triple.getTriple(), g->target.cpu,
                                                featuresString);
#else
    if (g->target.cpu.size()) {
        llvm::SubtargetFeatures features;
        features.setCPU(g->target.cpu);
        featuresString = features.getString();
    }

    targetMachine = target->createTargetMachine(triple.getTriple(), 
                                                featuresString);
#endif
    if (targetMachine == NULL) {
        fprintf(stderr, "Unable to create target machine for target \"%s\"!",
                triple.str().c_str());
        return false;
    }
    targetMachine->setAsmVerbosityDefault(true);

    // Figure out if we're generating object file or assembly output, and
    // set binary output for object files
    llvm::TargetMachine::CodeGenFileType fileType = (outputType == Object) ? 
        llvm::TargetMachine::CGFT_ObjectFile : llvm::TargetMachine::CGFT_AssemblyFile;
    bool binary = (fileType == llvm::TargetMachine::CGFT_ObjectFile);
    unsigned int flags = binary ? llvm::raw_fd_ostream::F_Binary : 0;

    llvm::tool_output_file *of = new llvm::tool_output_file(outFileName, error, flags);
    if (error.size()) {
        fprintf(stderr, "Error opening output file \"%s\".\n", outFileName);
        return false;
    }

    llvm::PassManager pm;
    if (const llvm::TargetData *td = targetMachine->getTargetData())
        pm.add(new llvm::TargetData(*td));
    else
        pm.add(new llvm::TargetData(module));

    llvm::formatted_raw_ostream fos(of->os());
    llvm::CodeGenOpt::Level optLevel = 
        (g->opt.level > 0) ? llvm::CodeGenOpt::Aggressive : llvm::CodeGenOpt::None;

    if (targetMachine->addPassesToEmitFile(pm, fos, fileType, optLevel)) {
        fprintf(stderr, "Fatal error adding passes to emit object file for "
                "target %s!\n", triple.str().c_str());
        return false;
    }

    // Finally, run the passes to emit the object file/assembly
    pm.run(*module);

    // Success; tell tool_output_file to keep the final output file. 
    of->keep();

    return true;
}


/** Small structure used in representing dependency graphs of structures
    (i.e. given a StructType, which other structure types does it have as
    elements).
 */ 
struct StructDAGNode {
    StructDAGNode()
        : visited(false) { }

    bool visited;
    std::vector<const StructType *> dependents;
};


/** Visit a node for the topological sort.
 */
static void
lVisitNode(const StructType *structType, 
           std::map<const StructType *, StructDAGNode *> &structToNode,
           std::vector<const StructType *> &sortedTypes) {
    assert(structToNode.find(structType) != structToNode.end());
    // Get the node that encodes the structs that this one is immediately
    // dependent on.
    StructDAGNode *node = structToNode[structType];
    if (node->visited)
        return;

    node->visited = true;
    // Depth-first traversal: visit all of the dependent nodes...
    for (unsigned int i = 0; i < node->dependents.size(); ++i)
        lVisitNode(node->dependents[i], structToNode, sortedTypes);
    // ...and then add this one to the sorted list
    sortedTypes.push_back(structType);
}
           

/** Given a set of structures that we want to print C declarations of in a
    header file, order them so that any struct that is used as a member
    variable in another struct is printed before the struct that uses it
    and then print them to the given file.
 */
static void
lEmitStructDecls(std::vector<const StructType *> &structTypes, FILE *file) {
    // First, build a DAG among the struct types where there is an edge
    // from node A to node B if struct type A depends on struct type B

    // Records the struct types that have incoming edges in the
    // DAG--i.e. the ones that one or more other struct types depend on
    std::set<const StructType *> hasIncomingEdges;
    // Records the mapping between struct type pointers and the
    // StructDagNode structures
    std::map<const StructType *, StructDAGNode *> structToNode;
    for (unsigned int i = 0; i < structTypes.size(); ++i) {
        // For each struct type, create its DAG node and record the
        // relationship between it and its node
        const StructType *st = structTypes[i];
        StructDAGNode *node = new StructDAGNode;
        structToNode[st] = node;

        for (int j = 0; j < st->GetElementCount(); ++j) {
            const StructType *elementStructType = 
                dynamic_cast<const StructType *>(st->GetElementType(j));
            // If this element is a struct type and we haven't already
            // processed it for the current struct type, then upate th
            // dependencies and record that this element type has other
            // struct types that depend on it.
            if (elementStructType != NULL &&
                (std::find(node->dependents.begin(), node->dependents.end(), 
                           elementStructType) == node->dependents.end())) {
                node->dependents.push_back(elementStructType);
                hasIncomingEdges.insert(elementStructType);
            }
        }
    }

    // Perform a topological sort of the struct types.  Kick it off by
    // visiting nodes with no incoming edges; i.e. the struct types that no
    // other struct types depend on.
    std::vector<const StructType *> sortedTypes;
    for (unsigned int i = 0; i < structTypes.size(); ++i) {
        const StructType *structType = structTypes[i];
        if (hasIncomingEdges.find(structType) == hasIncomingEdges.end())
            lVisitNode(structType, structToNode, sortedTypes);
    }
    assert(sortedTypes.size() == structTypes.size());

    // And finally we can emit the struct declarations by going through the
    // sorted ones in order.
    for (unsigned int i = 0; i < sortedTypes.size(); ++i) {
        const StructType *st = sortedTypes[i];
        fprintf(file, "struct %s {\n", st->GetStructName().c_str());
        for (int j = 0; j < st->GetElementCount(); ++j) {
            const Type *type = st->GetElementType(j)->GetAsNonConstType();
            std::string d = type->GetCDeclaration(st->GetElementName(j));
            fprintf(file, "    %s;\n", d.c_str());
        }
        fprintf(file, "};\n\n");
    }
}


/** Emit C declarations of enumerator types to the generated header file.
 */
static void
lEmitEnumDecls(const std::vector<const EnumType *> &enumTypes, FILE *file) {
    if (enumTypes.size() == 0)
        return;

    fprintf(file, "///////////////////////////////////////////////////////////////////////////\n");
    fprintf(file, "// Enumerator types with external visibility from ispc code\n");
    fprintf(file, "///////////////////////////////////////////////////////////////////////////\n\n");
    
    for (unsigned int i = 0; i < enumTypes.size(); ++i) {
        std::string declaration = enumTypes[i]->GetCDeclaration("");
        fprintf(file, "%s {\n", declaration.c_str());

        // Print the individual enumerators 
        for (int j = 0; j < enumTypes[i]->GetEnumeratorCount(); ++j) {
            const Symbol *e = enumTypes[i]->GetEnumerator(j);
            assert(e->constValue != NULL);
            unsigned int enumValue;
            int count = e->constValue->AsUInt32(&enumValue);
            assert(count == 1);

            // Always print an initializer to set the value.  We could be
            // 'clever' here and detect whether the implicit value given by
            // one plus the previous enumerator value (or zero, for the
            // first enumerator) is the same as the value stored with the
            // enumerator, though that doesn't seem worth the trouble...
            fprintf(file, "    %s = %d%c\n", e->name.c_str(), enumValue,
                    (j < enumTypes[i]->GetEnumeratorCount() - 1) ? ',' : ' ');
        }
        fprintf(file, "};\n");
    }
}


/** Print declarations of VectorTypes used in 'export'ed parts of the
    program in the header file.
 */
static void
lEmitVectorTypedefs(const std::vector<const VectorType *> &types, FILE *file) {
    if (types.size() == 0)
        return;

    fprintf(file, "///////////////////////////////////////////////////////////////////////////\n");
    fprintf(file, "// Vector types with external visibility from ispc code\n");
    fprintf(file, "///////////////////////////////////////////////////////////////////////////\n\n");

    int align = g->target.nativeVectorWidth * 4;

    for (unsigned int i = 0; i < types.size(); ++i) {
        std::string baseDecl;
        const VectorType *vt = types[i]->GetAsNonConstType();
        int size = vt->GetElementCount();

        baseDecl = vt->GetBaseType()->GetCDeclaration("");
        fprintf(file, "#ifdef _MSC_VER\n__declspec( align(%d) ) ", align);
        fprintf(file, "struct %s%d { %s v[%d]; };\n", baseDecl.c_str(), size,
                baseDecl.c_str(), size);
        fprintf(file, "#else\n");
        fprintf(file, "struct %s%d { %s v[%d]; } __attribute__ ((aligned(%d)));\n", 
                baseDecl.c_str(), size, baseDecl.c_str(), size, align);
        fprintf(file, "#endif\n");
    }
    fprintf(file, "\n");
}


/** Add the given type to the vector, if that type isn't already in there.
 */
template <typename T> static void
lAddTypeIfNew(const Type *type, std::vector<const T *> *exportedTypes) {
    type = type->GetAsNonConstType();

    // Linear search, so this ends up being n^2.  It's unlikely this will
    // matter in practice, though.
    for (unsigned int i = 0; i < exportedTypes->size(); ++i)
        if (Type::Equal((*exportedTypes)[i], type))
            return;

    const T *castType = dynamic_cast<const T *>(type);
    assert(castType != NULL);
    exportedTypes->push_back(castType);
}


/** Given an arbitrary type that appears in the app/ispc interface, add it
    to an appropriate vector if it is a struct, enum, or short vector type.
    Then, if it's a struct, recursively process its members to do the same.
 */
static void
lGetExportedTypes(const Type *type, 
                  std::vector<const StructType *> *exportedStructTypes,
                  std::vector<const EnumType *> *exportedEnumTypes,
                  std::vector<const VectorType *> *exportedVectorTypes) {
    const ArrayType *arrayType = dynamic_cast<const ArrayType *>(type);
    const StructType *structType = dynamic_cast<const StructType *>(type);

    if (dynamic_cast<const ReferenceType *>(type) != NULL)
        lGetExportedTypes(type->GetReferenceTarget(), exportedStructTypes, 
                          exportedEnumTypes, exportedVectorTypes);
    else if (arrayType != NULL)
        lGetExportedTypes(arrayType->GetElementType(), exportedStructTypes, 
                          exportedEnumTypes, exportedVectorTypes);
    else if (structType != NULL) {
        lAddTypeIfNew(type, exportedStructTypes);
        for (int i = 0; i < structType->GetElementCount(); ++i)
            lGetExportedTypes(structType->GetElementType(i), exportedStructTypes,
                              exportedEnumTypes, exportedVectorTypes);
    }
    else if (dynamic_cast<const EnumType *>(type) != NULL)
        lAddTypeIfNew(type, exportedEnumTypes);
    else if (dynamic_cast<const VectorType *>(type) != NULL)
        lAddTypeIfNew(type, exportedVectorTypes);
    else
        assert(dynamic_cast<const AtomicType *>(type) != NULL);
}


/** Given a set of functions, return the set of structure and vector types
    present in the parameters to them.
 */
static void
lGetExportedParamTypes(const std::vector<Symbol *> &funcs, 
                       std::vector<const StructType *> *exportedStructTypes,
                       std::vector<const EnumType *> *exportedEnumTypes,
                       std::vector<const VectorType *> *exportedVectorTypes) {
    for (unsigned int i = 0; i < funcs.size(); ++i) {
        const FunctionType *ftype = dynamic_cast<const FunctionType *>(funcs[i]->type);
        // Handle the return type
        lGetExportedTypes(ftype->GetReturnType(), exportedStructTypes,
                          exportedEnumTypes, exportedVectorTypes);

        // And now the parameter types...
        const std::vector<const Type *> &argTypes = ftype->GetArgumentTypes();
        for (unsigned int j = 0; j < argTypes.size(); ++j)
            lGetExportedTypes(argTypes[j], exportedStructTypes,
                              exportedEnumTypes, exportedVectorTypes);
    }
}


static void
lPrintFunctionDeclarations(FILE *file, const std::vector<Symbol *> &funcs) {
    fprintf(file, "#ifdef __cplusplus\nextern \"C\" {\n#endif // __cplusplus\n");
    for (unsigned int i = 0; i < funcs.size(); ++i) {
        const FunctionType *ftype = dynamic_cast<const FunctionType *>(funcs[i]->type);
        assert(ftype);
        std::string decl = ftype->GetCDeclaration(funcs[i]->name);
        fprintf(file, "    extern %s;\n", decl.c_str());
    }
    fprintf(file, "#ifdef __cplusplus\n}\n#endif // __cplusplus\n");
}


static void
lPrintExternGlobals(FILE *file, const std::vector<Symbol *> &externGlobals) {
    for (unsigned int i = 0; i < externGlobals.size(); ++i) {
        Symbol *sym = externGlobals[i];
        if (lRecursiveCheckVarying(sym->type))
            Warning(sym->pos, "Not emitting declaration for symbol \"%s\" into generated "
                    "header file since it (or some of its members) are varying.",
                    sym->name.c_str());
        else
            fprintf(file, "extern %s;\n", sym->type->GetCDeclaration(sym->name).c_str());
    }
}


static bool
lIsExported(const Symbol *sym) {
    const FunctionType *ft = dynamic_cast<const FunctionType *>(sym->type);
    assert(ft);
    return ft->isExported;
}


static bool
lIsExternC(const Symbol *sym) {
    const FunctionType *ft = dynamic_cast<const FunctionType *>(sym->type);
    assert(ft);
    return ft->isExternC;
}


bool
Module::writeHeader(const char *fn) {
    FILE *f = fopen(fn, "w");
    if (!f) {
        perror("fopen");
        return false;
    }
    fprintf(f, "//\n// %s\n// (Header automatically generated by the ispc compiler.)\n", fn);
    fprintf(f, "// DO NOT EDIT THIS FILE.\n//\n\n");

    // Create a nice guard string from the filename, turning any
    // non-number/letter characters into underbars
    std::string guard = "ISPC_";
    const char *p = fn;
    while (*p) {
        if (isdigit(*p)) 
            guard += *p;
        else if (isalpha(*p)) 
            guard += toupper(*p);
        else
            guard += "_";
        ++p;
    }
    fprintf(f, "#ifndef %s\n#define %s\n\n", guard.c_str(), guard.c_str());

    fprintf(f, "#include <stdint.h>\n\n");

    switch (g->target.isa) {
    case Target::SSE2:
        fprintf(f, "#define ISPC_TARGET_SSE2\n\n");
        break;
    case Target::SSE4:
        fprintf(f, "#define ISPC_TARGET_SSE4\n\n");
        break;
    case Target::AVX:
        fprintf(f, "#define ISPC_TARGET_AVX\n\n");
        break;
    default:
        FATAL("Unhandled target in header emission");
    }

    fprintf(f, "#ifdef __cplusplus\nnamespace ispc {\n#endif // __cplusplus\n\n");

    if (g->emitInstrumentation) {
        fprintf(f, "#define ISPC_INSTRUMENTATION 1\n");
        fprintf(f, "extern \"C\" {\n");
        fprintf(f, "  void ISPCInstrument(const char *fn, const char *note, int line, int mask);\n");
        fprintf(f, "}\n");
    }

    // Collect single linear arrays of the exported and extern "C"
    // functions
    std::vector<Symbol *> exportedFuncs, externCFuncs;
    m->symbolTable->GetMatchingFunctions(lIsExported, &exportedFuncs);
    m->symbolTable->GetMatchingFunctions(lIsExternC, &externCFuncs);
    
    // Get all of the struct, vector, and enumerant types used as function
    // parameters.  These vectors may have repeats.
    std::vector<const StructType *> exportedStructTypes;
    std::vector<const EnumType *> exportedEnumTypes;
    std::vector<const VectorType *> exportedVectorTypes;
    lGetExportedParamTypes(exportedFuncs, &exportedStructTypes,
                           &exportedEnumTypes, &exportedVectorTypes);
    lGetExportedParamTypes(externCFuncs, &exportedStructTypes,
                           &exportedEnumTypes, &exportedVectorTypes);

    // And do the same for the 'extern' globals
    for (unsigned int i = 0; i < externGlobals.size(); ++i)
        lGetExportedTypes(externGlobals[i]->type, &exportedStructTypes,
                          &exportedEnumTypes, &exportedVectorTypes);

    // And print them
    lEmitVectorTypedefs(exportedVectorTypes, f);
    lEmitEnumDecls(exportedEnumTypes, f);
    lEmitStructDecls(exportedStructTypes, f);

    // emit externs for globals
    if (externGlobals.size() > 0) {
        fprintf(f, "///////////////////////////////////////////////////////////////////////////\n");
        fprintf(f, "// Globals declared \"extern\" from ispc code\n");
        fprintf(f, "///////////////////////////////////////////////////////////////////////////\n");
        lPrintExternGlobals(f, externGlobals);
    }

    // emit function declarations for exported stuff...
    if (exportedFuncs.size() > 0) {
        fprintf(f, "\n");
        fprintf(f, "///////////////////////////////////////////////////////////////////////////\n");
        fprintf(f, "// Functions exported from ispc code\n");
        fprintf(f, "///////////////////////////////////////////////////////////////////////////\n");
        lPrintFunctionDeclarations(f, exportedFuncs);
    }
#if 0
    if (externCFuncs.size() > 0) {
        fprintf(f, "\n");
        fprintf(f, "///////////////////////////////////////////////////////////////////////////\n");
        fprintf(f, "// External C functions used by ispc code\n");
        fprintf(f, "///////////////////////////////////////////////////////////////////////////\n");
        lPrintFunctionDeclarations(f, externCFuncs);
    }
#endif

    // end namespace
    fprintf(f, "\n#ifdef __cplusplus\n}\n#endif // __cplusplus\n");

    // end guard
    fprintf(f, "\n#endif // %s\n", guard.c_str());

    fclose(f);
    return true;
}


void
Module::execPreprocessor(const char* infilename, llvm::raw_string_ostream* ostream) const
{
    clang::CompilerInstance inst;
    std::string error;

    inst.createFileManager();
    inst.createDiagnostics(0, NULL);
    clang::TargetOptions& options = inst.getTargetOpts();

    llvm::Triple triple(module->getTargetTriple());
    if (triple.getTriple().empty())
        triple.setTriple(llvm::sys::getHostTriple());
    
    options.Triple = triple.getTriple();

    clang::TargetInfo* target 
        = clang::TargetInfo::CreateTargetInfo(inst.getDiagnostics(), options);

    inst.setTarget(target);
    inst.createSourceManager(inst.getFileManager());
    inst.InitializeSourceManager(infilename);

    clang::PreprocessorOptions& opts = inst.getPreprocessorOpts();

    //Add defs for ISPC and PI
    opts.addMacroDef("ISPC");
    opts.addMacroDef("PI=3.1415926535");

    for (unsigned int i = 0; i < g->cppArgs.size(); ++i) {
        //Sanity Check, should really begin with -D
        if (g->cppArgs[i].substr(0,2) == "-D") {
            opts.addMacroDef(g->cppArgs[i].substr(2));
        }
    }    
    inst.createPreprocessor();
    clang::DoPrintPreprocessedInput(inst.getPreprocessor(),
                                    ostream, inst.getPreprocessorOutputOpts());
}

