/*
  Copyright (c) 2010-2014, Intel Corporation
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
#include "func.h"
#include "builtins.h"
#include "type.h"
#include "expr.h"
#include "sym.h"
#include "stmt.h"
#include "opt.h"
#include "llvmutil.h"

#include <stdio.h>
#include <stdarg.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <algorithm>
#include <set>
#include <sstream>
#include <iostream>
#ifdef ISPC_IS_WINDOWS
#include <windows.h>
#include <io.h>
#define strcasecmp stricmp
#endif

#if defined(LLVM_3_2)
  #include <llvm/LLVMContext.h>
  #include <llvm/Module.h>
  #include <llvm/Type.h>
  #include <llvm/Instructions.h>
  #include <llvm/Intrinsics.h>
  #include <llvm/DerivedTypes.h>
#else
  #include <llvm/IR/LLVMContext.h>
  #include <llvm/IR/Module.h>
  #include <llvm/IR/Type.h>
  #include <llvm/IR/Instructions.h>
  #include <llvm/IR/Intrinsics.h>
  #include <llvm/IR/DerivedTypes.h>
#endif
#include <llvm/PassManager.h>
#include <llvm/PassRegistry.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Support/FormattedStream.h>
#include <llvm/Support/FileUtilities.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#if defined(LLVM_3_2)
  #include <llvm/DataLayout.h>
  #include <llvm/TargetTransformInfo.h>
#else // LLVM 3.3+
  #include <llvm/IR/DataLayout.h>
  #include <llvm/Analysis/TargetTransformInfo.h>
#endif
#if !defined(LLVM_3_2) && !defined(LLVM_3_3) && !defined(LLVM_3_4) // LLVM 3.5+
    #include <llvm/IR/Verifier.h>
    #include <llvm/IR/IRPrintingPasses.h>
    #include <llvm/IR/CFG.h>
#else
    #include <llvm/Analysis/Verifier.h>
    #include <llvm/Assembly/PrintModulePass.h>
    #include <llvm/Support/CFG.h>
#endif
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Frontend/Utils.h>
#include <clang/Basic/TargetInfo.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Bitcode/ReaderWriter.h>

/*! list of files encountered by the parser. this allows emitting of
    the module file's dependencies via the -MMM option */
std::set<std::string> registeredDependencies;

/*! this is where the parser tells us that it has seen the given file
    name in the CPP hash */
void RegisterDependency(const std::string &fileName)
{
  if (fileName[0] != '<' && fileName != "stdlib.ispc")
    registeredDependencies.insert(fileName);
}

static void
lDeclareSizeAndPtrIntTypes(SymbolTable *symbolTable) {
    const Type *ptrIntType = (g->target->is32Bit()) ? AtomicType::VaryingInt32 :
        AtomicType::VaryingInt64;
    ptrIntType = ptrIntType->GetAsUnboundVariabilityType();

    symbolTable->AddType("intptr_t", ptrIntType, SourcePos());
    symbolTable->AddType("uintptr_t", ptrIntType->GetAsUnsignedType(),
                         SourcePos());
    symbolTable->AddType("ptrdiff_t", ptrIntType, SourcePos());

    const Type *sizeType = (g->target->is32Bit() || g->opt.force32BitAddressing) ?
        AtomicType::VaryingUInt32 : AtomicType::VaryingUInt64;
    sizeType = sizeType->GetAsUnboundVariabilityType();
    symbolTable->AddType("size_t", sizeType, SourcePos());
}


/** After compilation completes, there's often a lot of extra debugging
    metadata left around that isn't needed any more--for example, for
    static functions that weren't actually used, function information for
    functions that were inlined, etc.  This function takes a llvm::Module
    and tries to strip out all of this extra stuff.
 */
static void
lStripUnusedDebugInfo(llvm::Module *module) {
    if (g->generateDebuggingSymbols == false)
        return;

    // loop over the compile units that contributed to the final module
    if (llvm::NamedMDNode *cuNodes = module->getNamedMetadata("llvm.dbg.cu")) {
        for (unsigned i = 0, ie = cuNodes->getNumOperands(); i != ie; ++i) {
            llvm::MDNode *cuNode = cuNodes->getOperand(i);
            llvm::DICompileUnit cu(cuNode);
            llvm::DIArray subprograms = cu.getSubprograms();
            std::vector<llvm::Value *> usedSubprograms;

            if (subprograms.getNumElements() == 0)
                continue;

            // And now loop over the subprograms inside each compile unit.
            for (unsigned j = 0, je = subprograms.getNumElements(); j != je; ++j) {
                llvm::MDNode *spNode =
                    llvm::dyn_cast<llvm::MDNode>(subprograms->getOperand(j));
                Assert(spNode != NULL);
                llvm::DISubprogram sp(spNode);

                // Get the name of the subprogram.  Start with the mangled
                // name; if that's empty then we have an export'ed
                // function, so grab the unmangled name in that case.
                std::string name = sp.getLinkageName();
                if (name == "")
                    name = sp.getName();

                // Does the llvm::Function for this function survive in the
                // module?
                if (module->getFunction(name) != NULL)
                    usedSubprograms.push_back(sp);
            }

            Debug(SourcePos(), "%d / %d functions left in module with debug "
                  "info.", (int)usedSubprograms.size(),
                  (int)subprograms.getNumElements());

            // We'd now like to replace the array of subprograms in the
            // compile unit with only the ones that actually have function
            // definitions present.  Unfortunately, llvm::DICompileUnit
            // doesn't provide a method to set the subprograms.  Therefore,
            // we end up needing to directly stuff a new array into the
            // appropriate slot (number 12) in the MDNode for the compile
            // unit.
            //
            // Because this is all so hard-coded and would break if the
            // debugging metadata organization on the LLVM side changed,
            // here is a bunch of asserting to make sure that element 12 of
            // the compile unit's MDNode has the subprograms array....
            //
            // Update: This is not an approved way of working with debug info
            // metadata. It's not supposed to be deleted. But in out use-case
            // it's quite useful thing, as we link in bunch of unnecessary
            // stuff and remove it later on. Removing it is useful, as it
            // reduces size of the binary significantly (manyfold for small
            // programs).
#if defined(LLVM_3_2)
            llvm::MDNode *nodeSPMD =
                llvm::dyn_cast<llvm::MDNode>(cuNode->getOperand(12));
            Assert(nodeSPMD != NULL);
            llvm::MDNode *nodeSPMDArray =
                llvm::dyn_cast<llvm::MDNode>(nodeSPMD->getOperand(0));
            llvm::DIArray nodeSPs(nodeSPMDArray);
            Assert(nodeSPs.getNumElements() == subprograms.getNumElements());
            for (int i = 0; i < (int)nodeSPs.getNumElements(); ++i)
                Assert(nodeSPs.getElement(i) == subprograms.getElement(i));

            // And now we can go and stuff it into the node with some
            // confidence...
            llvm::Value *usedSubprogramsArray =
                m->diBuilder->getOrCreateArray(llvm::ArrayRef<llvm::Value *>(usedSubprograms));
            llvm::MDNode *replNode =
                llvm::MDNode::get(*g->ctx, llvm::ArrayRef<llvm::Value *>(usedSubprogramsArray));
            cuNode->replaceOperandWith(12, replNode);
#else // LLVM 3.3+
            llvm::MDNode *nodeSPMDArray =
                llvm::dyn_cast<llvm::MDNode>(cuNode->getOperand(9));
            Assert(nodeSPMDArray != NULL);
            llvm::DIArray nodeSPs(nodeSPMDArray);
            Assert(nodeSPs.getNumElements() == subprograms.getNumElements());
            for (int i = 0; i < (int)nodeSPs.getNumElements(); ++i)
                Assert(nodeSPs.getElement(i) == subprograms.getElement(i));

            // And now we can go and stuff it into the node with some
            // confidence...
            llvm::MDNode *replNode =
                m->diBuilder->getOrCreateArray(llvm::ArrayRef<llvm::Value *>(usedSubprograms));
            cuNode->replaceOperandWith(9, replNode);
#endif
        }
    }

    // Also, erase a bunch of named metadata detrius; for each function
    // there is sometimes named metadata llvm.dbg.lv.{funcname} that
    // doesn't seem to be otherwise needed.
    std::vector<llvm::NamedMDNode *> toErase;
    llvm::Module::named_metadata_iterator iter = module->named_metadata_begin();
    for (; iter != module->named_metadata_end(); ++iter) {
        if (!strncmp(iter->getName().str().c_str(), "llvm.dbg.lv", 11))
            toErase.push_back(iter);
    }
    for (int i = 0; i < (int)toErase.size(); ++i)
        module->eraseNamedMetadata(toErase[i]);

    // Wrap up by running the LLVM pass to remove anything left that's
    // unused.
    llvm::PassManager pm;
    pm.add(llvm::createStripDeadDebugInfoPass());
    pm.run(*module);
}


///////////////////////////////////////////////////////////////////////////
// Module

Module::Module(const char *fn) {
    // It's a hack to do this here, but it must be done after the target
    // information has been set (so e.g. the vector width is known...)  In
    // particular, if we're compiling to multiple targets with different
    // vector widths, this needs to be redone each time through.
    InitLLVMUtil(g->ctx, *g->target);

    filename = fn;
    errorCount = 0;
    symbolTable = new SymbolTable;
    ast = new AST;

    lDeclareSizeAndPtrIntTypes(symbolTable);

    module = new llvm::Module(filename ? filename : "<stdin>", *g->ctx);
    module->setTargetTriple(g->target->GetTripleString());

    // DataLayout information supposed to be managed in single place in Target class.
    module->setDataLayout(g->target->getDataLayout()->getStringRepresentation());

    if (g->generateDebuggingSymbols) {
        diBuilder = new llvm::DIBuilder(*module);

        // Let the DIBuilder know that we're starting a new compilation
        // unit.
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
            char producerString[512];
#if defined(BUILD_VERSION) && defined (BUILD_DATE)
            sprintf(producerString, "ispc version %s (build %s on %s)",
                    ISPC_VERSION, BUILD_VERSION, BUILD_DATE);
#else
            sprintf(producerString, "ispc version %s (built on %s)",
                    ISPC_VERSION, __DATE__);
#endif
#if !defined(LLVM_3_2) && !defined(LLVM_3_3)
            diCompileUnit = 
#endif // LLVM_3_4+            
            diBuilder->createCompileUnit(llvm::dwarf::DW_LANG_C99,  /* lang */
                                         name,  /* filename */
                                         directory, /* directory */
                                         producerString, /* producer */
                                         g->opt.level > 0 /* is optimized */,
                                         "-g", /* command line args */
                                         0 /* run time version */);
        }
    }
    else
        diBuilder = NULL;
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
    extern void ParserInit();
    ParserInit();

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

    ast->GenerateIR();

    if (errorCount == 0)
        Optimize(module, g->opt.level);

    return errorCount;
}


void
Module::AddTypeDef(const std::string &name, const Type *type,
                   SourcePos pos) {
    // Typedefs are easy; just add the mapping between the given name and
    // the given type.
    symbolTable->AddType(name.c_str(), type, pos);
}


void
Module::AddGlobalVariable(const std::string &name, const Type *type, Expr *initExpr,
                          bool isConst, StorageClass storageClass, SourcePos pos) {
    // These may be NULL due to errors in parsing; just gracefully return
    // here if so.
    if (name == "" || type == NULL) {
        Assert(errorCount > 0);
        return;
    }

    if (symbolTable->LookupFunction(name.c_str())) {
        Error(pos, "Global variable \"%s\" shadows previously-declared "
              "function.", name.c_str());
        return;
    }

    if (storageClass == SC_EXTERN_C) {
        Error(pos, "extern \"C\" qualifier can only be used for "
              "functions.");
        return;
    }

    if (type->IsVoidType()) {
        Error(pos, "\"void\" type global variable is illegal.");
        return;
    }

    type = ArrayType::SizeUnsizedArrays(type, initExpr);
    if (type == NULL)
        return;

    const ArrayType *at = CastType<ArrayType>(type);
    if (at != NULL && at->TotalElementCount() == 0) {
        Error(pos, "Illegal to declare a global variable with unsized "
              "array dimensions that aren't set with an initializer "
              "expression.");
        return;
    }

    llvm::Type *llvmType = type->LLVMType(g->ctx);
    if (llvmType == NULL)
        return;

    // See if we have an initializer expression for the global.  If so,
    // make sure it's a compile-time constant!
    llvm::Constant *llvmInitializer = NULL;
    ConstExpr *constValue = NULL;
    if (storageClass == SC_EXTERN || storageClass == SC_EXTERN_C) {
        if (initExpr != NULL)
            Error(pos, "Initializer can't be provided with \"extern\" "
                  "global variable \"%s\".", name.c_str());
    }
    else {
        if (initExpr != NULL) {
            initExpr = TypeCheck(initExpr);
            if (initExpr != NULL) {
                // We need to make sure the initializer expression is
                // the same type as the global.  (But not if it's an
                // ExprList; they don't have types per se / can't type
                // convert themselves anyway.)
                if (dynamic_cast<ExprList *>(initExpr) == NULL)
                    initExpr = TypeConvertExpr(initExpr, type, "initializer");

                if (initExpr != NULL) {
                    initExpr = Optimize(initExpr);
                    // Fingers crossed, now let's see if we've got a
                    // constant value..
                    llvmInitializer = initExpr->GetConstant(type);

                    if (llvmInitializer != NULL) {
                        if (type->IsConstType())
                            // Try to get a ConstExpr associated with
                            // the symbol.  This dynamic_cast can
                            // validly fail, for example for types like
                            // StructTypes where a ConstExpr can't
                            // represent their values.
                            constValue = dynamic_cast<ConstExpr *>(initExpr);
                    }
                    else
                        Error(initExpr->pos, "Initializer for global variable \"%s\" "
                              "must be a constant.", name.c_str());
                }
            }
        }

        // If no initializer was provided or if we couldn't get a value
        // above, initialize it with zeros..
        if (llvmInitializer == NULL)
            llvmInitializer = llvm::Constant::getNullValue(llvmType);
    }

    Symbol *sym = symbolTable->LookupVariable(name.c_str());
    llvm::GlobalVariable *oldGV = NULL;
    if (sym != NULL) {
        // We've already seen either a declaration or a definition of this
        // global.

        // If the type doesn't match with the previous one, issue an error.
        if (!Type::Equal(sym->type, type) ||
            (sym->storageClass != SC_EXTERN &&
             sym->storageClass != SC_EXTERN_C &&
             sym->storageClass != storageClass)) {
            Error(pos, "Definition of variable \"%s\" conflicts with "
                  "definition at %s:%d.", name.c_str(),
                  sym->pos.name, sym->pos.first_line);
            return;
        }

        llvm::GlobalVariable *gv =
            llvm::dyn_cast<llvm::GlobalVariable>(sym->storagePtr);
        Assert(gv != NULL);

        // And issue an error if this is a redefinition of a variable
        if (gv->hasInitializer() &&
            sym->storageClass != SC_EXTERN && sym->storageClass != SC_EXTERN_C) {
            Error(pos, "Redefinition of variable \"%s\" is illegal. "
                  "(Previous definition at %s:%d.)", sym->name.c_str(),
                  sym->pos.name, sym->pos.first_line);
            return;
        }

        // Now, we either have a redeclaration of a global, or a definition
        // of a previously-declared global.  First, save the pointer to the
        // previous llvm::GlobalVariable
        oldGV = gv;
    }
    else {
        sym = new Symbol(name, pos, type, storageClass);
        symbolTable->AddVariable(sym);
    }
    sym->constValue = constValue;

    llvm::GlobalValue::LinkageTypes linkage =
        (sym->storageClass == SC_STATIC) ? llvm::GlobalValue::InternalLinkage :
        llvm::GlobalValue::ExternalLinkage;

    // Note that the NULL llvmInitializer is what leads to "extern"
    // declarations coming up extern and not defining storage (a bit
    // subtle)...
    sym->storagePtr = new llvm::GlobalVariable(*module, llvmType, isConst,
                                               linkage, llvmInitializer,
                                               sym->name.c_str());

    // Patch up any references to the previous GlobalVariable (e.g. from a
    // declaration of a global that was later defined.)
    if (oldGV != NULL) {
        oldGV->replaceAllUsesWith(sym->storagePtr);
        oldGV->removeFromParent();
        sym->storagePtr->setName(sym->name.c_str());
    }

    if (diBuilder) {
        llvm::DIFile file = pos.GetDIFile();
        llvm::DIGlobalVariable var =
            diBuilder->createGlobalVariable(name,
                                            file,
                                            pos.first_line,
                                            sym->type->GetDIType(file),
                                            (sym->storageClass == SC_STATIC),
                                            sym->storagePtr);
        Assert(var.Verify());
    }
}


/** Given an arbitrary type, see if it or any of the leaf types contained
    in it has a type that's illegal to have exported to C/C++
    code.

    (Note that it's fine for the original struct or a contained struct to
    be varying, so long as all of its members have bound 'uniform'
    variability.)

    This functions returns true and issues an error if are any illegal
    types are found and returns false otherwise.
*/
static bool
lRecursiveCheckValidParamType(const Type *t, bool vectorOk) {
    const StructType *st = CastType<StructType>(t);
    if (st != NULL) {
        for (int i = 0; i < st->GetElementCount(); ++i)
            if (!lRecursiveCheckValidParamType(st->GetElementType(i),
                                               vectorOk))
                return false;
        return true;
    }

    // Vector types are also not supported, pending ispc properly
    // supporting the platform ABI.  (Pointers to vector types are ok,
    // though.)  (https://github.com/ispc/ispc/issues/363)...
    if (vectorOk == false && CastType<VectorType>(t) != NULL)
        return false;

    const SequentialType *seqt = CastType<SequentialType>(t);
    if (seqt != NULL)
        return lRecursiveCheckValidParamType(seqt->GetElementType(), vectorOk);

    const PointerType *pt = CastType<PointerType>(t);
    if (pt != NULL) {
      // Only allow exported uniform pointers
      // Uniform pointers to varying data, however, are ok.
      if (pt->IsVaryingType()) 
        return false;
      else
        return lRecursiveCheckValidParamType(pt->GetBaseType(), true);
    }

    if (t->IsVaryingType() && !vectorOk)
      return false;
    else 
      return true;
}


/** Given a Symbol representing a function parameter, see if it or any
    contained types are varying.  If so, issue an error.  (This function
    should only be called for parameters to 'export'ed functions, where
    varying parameters is illegal.
 */
static void
lCheckExportedParameterTypes(const Type *type, const std::string &name,
                             SourcePos pos) {
    if (lRecursiveCheckValidParamType(type, false) == false) {
        if (CastType<PointerType>(type))
            Error(pos, "Varying pointer type parameter \"%s\" is illegal "
                  "in an exported function.", name.c_str());
        if (CastType<StructType>(type->GetBaseType()))
            Error(pos, "Struct parameter \"%s\" with vector typed "
                  "member(s) is illegal in an exported function.", name.c_str());
        else if (CastType<VectorType>(type))
            Error(pos, "Vector-typed parameter \"%s\" is illegal in an exported "
                  "function.", name.c_str());
        else
            Error(pos, "Varying parameter \"%s\" is illegal in an exported function.",
                  name.c_str());
    }
}


/** Given a function type, loop through the function parameters and see if
    any are StructTypes.  If so, issue an error; this is currently broken
    (https://github.com/ispc/ispc/issues/3).
 */
static void
lCheckForStructParameters(const FunctionType *ftype, SourcePos pos) {
    for (int i = 0; i < ftype->GetNumParameters(); ++i) {
        const Type *type = ftype->GetParameterType(i);
        if (CastType<StructType>(type) != NULL) {
            Error(pos, "Passing structs to/from application functions is "
                  "currently broken.  Use a pointer or const pointer to the "
                  "struct instead for now.");
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
void
Module::AddFunctionDeclaration(const std::string &name,
                               const FunctionType *functionType,
                               StorageClass storageClass, bool isInline,
                               SourcePos pos) {
    Assert(functionType != NULL);

    // If a global variable with the same name has already been declared
    // issue an error.
    if (symbolTable->LookupVariable(name.c_str()) != NULL) {
        Error(pos, "Function \"%s\" shadows previously-declared global variable. "
              "Ignoring this definition.",
              name.c_str());
        return;
    }

    std::vector<Symbol *> overloadFuncs;
    symbolTable->LookupFunction(name.c_str(), &overloadFuncs);
    if (overloadFuncs.size() > 0) {
        for (unsigned int i = 0; i < overloadFuncs.size(); ++i) {
            Symbol *overloadFunc = overloadFuncs[i];

            const FunctionType *overloadType =
                CastType<FunctionType>(overloadFunc->type);
            if (overloadType == NULL) {
                Assert(m->errorCount == 0);
                continue;
            }

            // Check for a redeclaration of a function with the same name
            // and type.  This also hits when we have previously declared
            // the function and are about to define it.
            if (Type::Equal(overloadFunc->type, functionType))
                return;

            if (functionType->isExported || overloadType->isExported)
                Error(pos, "Illegal to provide \"export\" qualifier for "
                      "functions with the same name but different types. "
                      "(Previous function declaration (%s:%d).)",
                      overloadFunc->pos.name, overloadFunc->pos.first_line);

            // If all of the parameter types match but the return type is
            // different, return an error--overloading by return type isn't
            // allowed.
            const FunctionType *ofType =
                CastType<FunctionType>(overloadFunc->type);
            Assert(ofType != NULL);
            if (ofType->GetNumParameters() == functionType->GetNumParameters()) {
                int i;
                for (i = 0; i < functionType->GetNumParameters(); ++i) {
                    if (Type::Equal(ofType->GetParameterType(i),
                                    functionType->GetParameterType(i)) == false)
                        break;
                }
                if (i == functionType->GetNumParameters()) {
                    std::string thisRetType = functionType->GetReturnTypeString();
                    std::string otherRetType = ofType->GetReturnTypeString();
                    Error(pos, "Illegal to overload function by return "
                          "type only.  This function returns \"%s\" while "
                          "previous declaration at %s:%d returns \"%s\".",
                          thisRetType.c_str(), overloadFunc->pos.name,
                          overloadFunc->pos.first_line, otherRetType.c_str());
                    return;
                }
            }
        }
    }

    if (storageClass == SC_EXTERN_C) {
        // Make sure the user hasn't supplied both an 'extern "C"' and a
        // 'task' qualifier with the function
        if (functionType->isTask) {
            Error(pos, "\"task\" qualifier is illegal with C-linkage extern "
                  "function \"%s\".  Ignoring this function.", name.c_str());
            return;
        }

        std::vector<Symbol *> funcs;
        symbolTable->LookupFunction(name.c_str(), &funcs);
        if (funcs.size() > 0) {
            if (funcs.size() > 1) {
                // Multiple functions with this name have already been declared;
                // can't overload here
                Error(pos, "Can't overload extern \"C\" function \"%s\"; "
                      "%d functions with the same name have already been declared.",
                      name.c_str(), (int)funcs.size());
                return;
            }

            // One function with the same name has been declared; see if it
            // has the same type as this one, in which case it's ok.
            if (Type::Equal(funcs[0]->type, functionType))
                return;
            else {
                Error(pos, "Can't overload extern \"C\" function \"%s\".",
                      name.c_str());
                return;
            }
        }
    }

    // Get the LLVM FunctionType
    bool disableMask = (storageClass == SC_EXTERN_C);
    llvm::FunctionType *llvmFunctionType =
        functionType->LLVMFunctionType(g->ctx, disableMask);
    if (llvmFunctionType == NULL)
        return;

    // And create the llvm::Function
    llvm::GlobalValue::LinkageTypes linkage = (storageClass == SC_STATIC ||
                                               isInline) ?
        llvm::GlobalValue::InternalLinkage : llvm::GlobalValue::ExternalLinkage;

    std::string functionName = name;
    if (storageClass != SC_EXTERN_C) {
        functionName += functionType->Mangle();
        if (g->mangleFunctionsWithTarget)
            functionName += g->target->GetISAString();
    }
    llvm::Function *function =
        llvm::Function::Create(llvmFunctionType, linkage, functionName.c_str(),
                               module);

    // Set function attributes: we never throw exceptions
    function->setDoesNotThrow();
    if (storageClass != SC_EXTERN_C &&
        isInline)
#ifdef LLVM_3_2
        function->addFnAttr(llvm::Attributes::AlwaysInline);
#else // LLVM 3.3+
        function->addFnAttr(llvm::Attribute::AlwaysInline);
#endif
    if (functionType->isTask)
        // This also applies transitively to members I think?
        function->setDoesNotAlias(1);

    g->target->markFuncWithTargetAttr(function);

    // Make sure that the return type isn't 'varying' or vector typed if
    // the function is 'export'ed.
    if (functionType->isExported &&
        lRecursiveCheckValidParamType(functionType->GetReturnType(), false) == false)
        Error(pos, "Illegal to return a \"varying\" or vector type from "
              "exported function \"%s\"", name.c_str());

    if (functionType->isTask &&
        functionType->GetReturnType()->IsVoidType() == false)
        Error(pos, "Task-qualified functions must have void return type.");

    if (functionType->isExported || functionType->isExternC)
        lCheckForStructParameters(functionType, pos);

    // Loop over all of the arguments; process default values if present
    // and do other checks and parameter attribute setting.
    bool seenDefaultArg = false;
    int nArgs = functionType->GetNumParameters();
    for (int i = 0; i < nArgs; ++i) {
        const Type *argType = functionType->GetParameterType(i);
        const std::string &argName = functionType->GetParameterName(i);
        Expr *defaultValue = functionType->GetParameterDefault(i);
        const SourcePos &argPos = functionType->GetParameterSourcePos(i);

        // If the function is exported, make sure that the parameter
        // doesn't have any funky stuff going on in it.
        // JCB nomosoa - Varying is now a-ok.
        if (functionType->isExported) {
          lCheckExportedParameterTypes(argType, argName, argPos);
        }

        // ISPC assumes that no pointers alias.  (It should be possible to
        // specify when this is not the case, but this should be the
        // default.)  Set parameter attributes accordingly.  (Only for
        // uniform pointers, since varying pointers are int vectors...)
        if (!functionType->isTask &&
            ((CastType<PointerType>(argType) != NULL &&
              argType->IsUniformType() &&
              // Exclude SOA argument because it is a pair {struct *, int}
              // instead of pointer
              !CastType<PointerType>(argType)->IsSlice())
             ||

             CastType<ReferenceType>(argType) != NULL)) {

            // NOTE: LLVM indexes function parameters starting from 1.
            // This is unintuitive.
            function->setDoesNotAlias(i+1);
#if 0
            int align = 4 * RoundUpPow2(g->target->nativeVectorWidth);
            function->addAttribute(i+1, llvm::Attribute::constructAlignmentFromInt(align));
#endif
        }

        if (symbolTable->LookupFunction(argName.c_str()))
            Warning(argPos, "Function parameter \"%s\" shadows a function "
                    "declared in global scope.", argName.c_str());

        if (defaultValue != NULL)
            seenDefaultArg = true;
        else if (seenDefaultArg) {
            // Once one parameter has provided a default value, then all of
            // the following ones must have them as well.
            Error(argPos, "Parameter \"%s\" is missing default: all "
                  "parameters after the first parameter with a default value "
                  "must have default values as well.", argName.c_str());
        }
    }

    // If llvm gave us back a Function * with a different name than the one
    // we asked for, then there's already a function with that same
    // (mangled) name in the llvm::Module.  In that case, erase the one we
    // tried to add and just work with the one it already had.
    if (function->getName() != functionName) {
        function->eraseFromParent();
        function = module->getFunction(functionName);
    }

    // Finally, we know all is good and we can add the function to the
    // symbol table
    Symbol *funSym = new Symbol(name, pos, functionType, storageClass);
    funSym->function = function;
    bool ok = symbolTable->AddFunction(funSym);
    Assert(ok);
}


void
Module::AddFunctionDefinition(const std::string &name, const FunctionType *type,
                              Stmt *code) {
    Symbol *sym = symbolTable->LookupFunction(name.c_str(), type);
    if (sym == NULL || code == NULL) {
        Assert(m->errorCount > 0);
        return;
    }

    sym->pos = code->pos;

    // FIXME: because we encode the parameter names in the function type,
    // we need to override the function type here in case the function had
    // earlier been declared with anonymous parameter names but is now
    // defined with actual names.  This is yet another reason we shouldn't
    // include the names in FunctionType...
    sym->type = type;

    ast->AddFunction(sym, code);
}


void
Module::AddExportedTypes(const std::vector<std::pair<const Type *,
                                                     SourcePos> > &types) {
    for (int i = 0; i < (int)types.size(); ++i) {
        if (CastType<StructType>(types[i].first) == NULL &&
            CastType<VectorType>(types[i].first) == NULL &&
            CastType<EnumType>(types[i].first) == NULL)
            Error(types[i].second, "Only struct, vector, and enum types, "
                  "not \"%s\", are allowed in type export lists.",
                  types[i].first->GetString().c_str());
        else
            exportedTypes.push_back(types[i]);
    }
}


bool
Module::writeOutput(OutputType outputType, const char *outFileName,
                    const char *includeFileName, DispatchHeaderInfo *DHI) {
    if (diBuilder != NULL && (outputType != Header && outputType != Deps)) {
        diBuilder->finalize();

        lStripUnusedDebugInfo(module);
    }

#if !defined(LLVM_3_2) && !defined(LLVM_3_3) // LLVM 3.4+
    // In LLVM_3_4 after r195494 and r195504 revisions we should pass
    // "Debug Info Version" constant to the module. LLVM will ignore
    // our Debug Info metadata without it.
    if (g->generateDebuggingSymbols == true) {
        module->addModuleFlag(llvm::Module::Error, "Debug Info Version", llvm::DEBUG_METADATA_VERSION);
    }
#endif

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
        case CXX:
            if (strcasecmp(suffix, "c") && strcasecmp(suffix, "cc") &&
                strcasecmp(suffix, "c++") && strcasecmp(suffix, "cxx") &&
                strcasecmp(suffix, "cpp"))
                fileType = "c++";
            break;
        case Header:
            if (strcasecmp(suffix, "h") && strcasecmp(suffix, "hh") &&
                strcasecmp(suffix, "hpp"))
                fileType = "header";
            break;
        case Deps:
          break;
        case DevStub:
          if (strcasecmp(suffix, "c") && strcasecmp(suffix, "cc") &&
              strcasecmp(suffix, "c++") && strcasecmp(suffix, "cxx") &&
              strcasecmp(suffix, "cpp"))
            fileType = "dev-side offload stub";
            break;
        case HostStub:
          if (strcasecmp(suffix, "c") && strcasecmp(suffix, "cc") &&
              strcasecmp(suffix, "c++") && strcasecmp(suffix, "cxx") &&
              strcasecmp(suffix, "cpp"))
            fileType = "host-side offload stub";
            break;
        default:
          Assert(0 /* swtich case not handled */);
          return 1;
        }
        if (fileType != NULL)
            Warning(SourcePos(), "Emitting %s file, but filename \"%s\" "
                    "has suffix \"%s\"?", fileType, outFileName, suffix);
    }

    if (outputType == Header) {
      if (DHI)
        return writeDispatchHeader(DHI);
      else
        return writeHeader(outFileName);
    }
    else if (outputType == Deps)
      return writeDeps(outFileName);
    else if (outputType == HostStub)
      return writeHostStub(outFileName);
    else if (outputType == DevStub)
      return writeDevStub(outFileName);
    else if (outputType == Bitcode)
        return writeBitcode(module, outFileName);
    else if (outputType == CXX) {
        if (g->target->getISA() != Target::GENERIC) {
            Error(SourcePos(), "Only \"generic-*\" targets can be used with "
                  "C++ emission.");
            return false;
        }
        extern bool WriteCXXFile(llvm::Module *module, const char *fn,
                                 int vectorWidth, const char *includeName);
        return WriteCXXFile(module, outFileName, g->target->getVectorWidth(),
                            includeFileName);
    }
    else
        return writeObjectFileOrAssembly(outputType, outFileName);
}


bool
Module::writeBitcode(llvm::Module *module, const char *outFileName) {
    // Get a file descriptor corresponding to where we want the output to
    // go.  If we open it, it'll be closed by the llvm::raw_fd_ostream
    // destructor.
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


bool
Module::writeObjectFileOrAssembly(OutputType outputType, const char *outFileName) {
    llvm::TargetMachine *targetMachine = g->target->GetTargetMachine();
    return writeObjectFileOrAssembly(targetMachine, module, outputType,
                                     outFileName);
}


bool
Module::writeObjectFileOrAssembly(llvm::TargetMachine *targetMachine,
                                  llvm::Module *module, OutputType outputType,
                                  const char *outFileName) {
    // Figure out if we're generating object file or assembly output, and
    // set binary output for object files
    llvm::TargetMachine::CodeGenFileType fileType = (outputType == Object) ?
        llvm::TargetMachine::CGFT_ObjectFile : llvm::TargetMachine::CGFT_AssemblyFile;
    bool binary = (fileType == llvm::TargetMachine::CGFT_ObjectFile);
#if defined(LLVM_3_2) || defined(LLVM_3_3)
    unsigned int flags = binary ? llvm::raw_fd_ostream::F_Binary : 0;
#elif defined(LLVM_3_4)
    llvm::sys::fs::OpenFlags flags = binary ? llvm::sys::fs::F_Binary :
        llvm::sys::fs::F_None;
#else
    llvm::sys::fs::OpenFlags flags = binary ? llvm::sys::fs::F_None :
        llvm::sys::fs::F_Text;

#endif

    std::string error;
    llvm::tool_output_file *of = new llvm::tool_output_file(outFileName, error, flags);
    if (error.size()) {
        fprintf(stderr, "Error opening output file \"%s\".\n", outFileName);
        return false;
    }

    llvm::PassManager pm;
#if !defined(LLVM_3_2) && !defined(LLVM_3_3) && !defined(LLVM_3_4) // LLVM 3.5+
    pm.add(new llvm::DataLayoutPass(*g->target->getDataLayout()));
#else
    pm.add(new llvm::DataLayout(*g->target->getDataLayout()));
#endif

    llvm::formatted_raw_ostream fos(of->os());

    if (targetMachine->addPassesToEmitFile(pm, fos, fileType)) {
        fprintf(stderr, "Fatal error adding passes to emit object file!");
        exit(1);
    }

    // Finally, run the passes to emit the object file/assembly
    pm.run(*module);

    // Success; tell tool_output_file to keep the final output file.
    of->keep();

    return true;
}


/** Given a pointer to an element of a structure, see if it is a struct
    type or an array of a struct type.  If so, return a pointer to the
    underlying struct type. */
static const StructType *
lGetElementStructType(const Type *t) {
    const StructType *st = CastType<StructType>(t);
    if (st != NULL)
        return st;

    const ArrayType *at = CastType<ArrayType>(t);
    if (at != NULL)
        return lGetElementStructType(at->GetElementType());

    return NULL;
}

static bool
lContainsPtrToVarying(const StructType *st) {
  int numElts = st->GetElementCount();

  for (int j = 0; j < numElts; ++j) {
    const Type *t = st->GetElementType(j);

    if (t->IsVaryingType()) return true;
  }

  return false;
}


/** Emits a declaration for the given struct to the given file.  This
    function first makes sure that declarations for any structs that are
    (recursively) members of this struct are emitted first.
 */
static void
lEmitStructDecl(const StructType *st, std::vector<const StructType *> *emittedStructs,
                FILE *file, bool emitUnifs=true) {

    // if we're emitting this for a generic dispatch header file and it's 
    // struct that only contains uniforms, don't bother if we're emitting uniforms
    if (!emitUnifs && !lContainsPtrToVarying(st)) {
      return;
    }

    // Has this struct type already been declared?  (This happens if it's a
    // member of another struct for which we emitted a declaration
    // previously.)
    for (int i = 0; i < (int)emittedStructs->size(); ++i)
        if (Type::EqualIgnoringConst(st, (*emittedStructs)[i]))
            return;

    // Otherwise first make sure any contained structs have been declared.
    for (int i = 0; i < st->GetElementCount(); ++i) {
        const StructType *elementStructType =
            lGetElementStructType(st->GetElementType(i));
        if (elementStructType != NULL)
          lEmitStructDecl(elementStructType, emittedStructs, file, emitUnifs);
    }

    // And now it's safe to declare this one
    emittedStructs->push_back(st);
    
    fprintf(file, "#ifndef __ISPC_STRUCT_%s__\n",st->GetCStructName().c_str());
    fprintf(file, "#define __ISPC_STRUCT_%s__\n",st->GetCStructName().c_str());

    fprintf(file, "struct %s", st->GetCStructName().c_str());
    if (st->GetSOAWidth() > 0)
        // This has to match the naming scheme in
        // StructType::GetCDeclaration().
        fprintf(file, "_SOA%d", st->GetSOAWidth());
    fprintf(file, " {\n");

    for (int i = 0; i < st->GetElementCount(); ++i) {
        const Type *type = st->GetElementType(i)->GetAsNonConstType();
        std::string d = type->GetCDeclaration(st->GetElementName(i));
        if (type->IsVaryingType()) {
          fprintf(file, "    %s[%d];\n", d.c_str(), g->target->getVectorWidth());
        }
        else {
          fprintf(file, "    %s;\n", d.c_str());
        }
    }
    fprintf(file, "};\n");
    fprintf(file, "#endif\n\n");
}


/** Given a set of structures that we want to print C declarations of in a
    header file, emit their declarations.
 */
static void
lEmitStructDecls(std::vector<const StructType *> &structTypes, FILE *file, bool emitUnifs=true) {
    std::vector<const StructType *> emittedStructs;
    for (unsigned int i = 0; i < structTypes.size(); ++i)
      lEmitStructDecl(structTypes[i], &emittedStructs, file, emitUnifs);
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
        fprintf(file, "#ifndef __ISPC_ENUM_%s__\n",enumTypes[i]->GetEnumName().c_str());
        fprintf(file, "#define __ISPC_ENUM_%s__\n",enumTypes[i]->GetEnumName().c_str());
        std::string declaration = enumTypes[i]->GetCDeclaration("");
        fprintf(file, "%s {\n", declaration.c_str());

        // Print the individual enumerators
        for (int j = 0; j < enumTypes[i]->GetEnumeratorCount(); ++j) {
            const Symbol *e = enumTypes[i]->GetEnumerator(j);
            Assert(e->constValue != NULL);
            unsigned int enumValue;
            int count = e->constValue->GetValues(&enumValue);
            Assert(count == 1);

            // Always print an initializer to set the value.  We could be
            // 'clever' here and detect whether the implicit value given by
            // one plus the previous enumerator value (or zero, for the
            // first enumerator) is the same as the value stored with the
            // enumerator, though that doesn't seem worth the trouble...
            fprintf(file, "    %s = %d%c\n", e->name.c_str(), enumValue,
                    (j < enumTypes[i]->GetEnumeratorCount() - 1) ? ',' : ' ');
        }
        fprintf(file, "};\n");
        fprintf(file, "#endif\n\n");
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

    int align = g->target->getNativeVectorWidth() * 4;

    for (unsigned int i = 0; i < types.size(); ++i) {
        std::string baseDecl;
        const VectorType *vt = types[i]->GetAsNonConstType();
        if (!vt->IsUniformType())
            // Varying stuff shouldn't be visibile to / used by the
            // application, so at least make it not simple to access it by
            // not declaring the type here...
            continue;

        int size = vt->GetElementCount();

        baseDecl = vt->GetBaseType()->GetCDeclaration("");
        fprintf(file, "#ifndef __ISPC_VECTOR_%s%d__\n",baseDecl.c_str(), size);
        fprintf(file, "#define __ISPC_VECTOR_%s%d__\n",baseDecl.c_str(), size);
        fprintf(file, "#ifdef _MSC_VER\n__declspec( align(%d) ) ", align);
        fprintf(file, "struct %s%d { %s v[%d]; };\n", baseDecl.c_str(), size,
                baseDecl.c_str(), size);
        fprintf(file, "#else\n");
        fprintf(file, "struct %s%d { %s v[%d]; } __attribute__ ((aligned(%d)));\n",
                baseDecl.c_str(), size, baseDecl.c_str(), size, align);
        fprintf(file, "#endif\n");
        fprintf(file, "#endif\n\n");
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

    const T *castType = CastType<T>(type);
    Assert(castType != NULL);
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
    const ArrayType *arrayType = CastType<ArrayType>(type);
    const StructType *structType = CastType<StructType>(type);

    if (CastType<ReferenceType>(type) != NULL)
        lGetExportedTypes(type->GetReferenceTarget(), exportedStructTypes,
                          exportedEnumTypes, exportedVectorTypes);
    else if (CastType<PointerType>(type) != NULL)
        lGetExportedTypes(type->GetBaseType(), exportedStructTypes,
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
    else if (CastType<UndefinedStructType>(type) != NULL)
        // do nothing
        ;
    else if (CastType<EnumType>(type) != NULL)
        lAddTypeIfNew(type, exportedEnumTypes);
    else if (CastType<VectorType>(type) != NULL)
        lAddTypeIfNew(type, exportedVectorTypes);
    else
        Assert(CastType<AtomicType>(type) != NULL);
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
        const FunctionType *ftype = CastType<FunctionType>(funcs[i]->type);
        // Handle the return type
        lGetExportedTypes(ftype->GetReturnType(), exportedStructTypes,
                          exportedEnumTypes, exportedVectorTypes);

        // And now the parameter types...
        for (int j = 0; j < ftype->GetNumParameters(); ++j)
            lGetExportedTypes(ftype->GetParameterType(j), exportedStructTypes,
                              exportedEnumTypes, exportedVectorTypes);
    }
}


static void
lPrintFunctionDeclarations(FILE *file, const std::vector<Symbol *> &funcs,
                           bool useExternC=1, bool rewriteForDispatch=false) {
  if (useExternC)
    fprintf(file, "#if defined(__cplusplus) && !defined(__ISPC_NO_EXTERN_C)\nextern \"C\" {\n#endif // __cplusplus\n");
    // fprintf(file, "#ifdef __cplusplus\nextern \"C\" {\n#endif // __cplusplus\n");
  for (unsigned int i = 0; i < funcs.size(); ++i) {
    const FunctionType *ftype = CastType<FunctionType>(funcs[i]->type);
    Assert(ftype);
    std::string decl;
    if (rewriteForDispatch) {
      decl = ftype->GetCDeclarationForDispatch(funcs[i]->name);
    }
    else {
      decl = ftype->GetCDeclaration(funcs[i]->name);
    }
    fprintf(file, "    extern %s;\n", decl.c_str());
  }
  if (useExternC)

    fprintf(file, "#if defined(__cplusplus) && !defined(__ISPC_NO_EXTERN_C)\n} /* end extern C */\n#endif // __cplusplus\n");
    // fprintf(file, "#ifdef __cplusplus\n} /* end extern C */\n#endif // __cplusplus\n");
}






static bool
lIsExported(const Symbol *sym) {
    const FunctionType *ft = CastType<FunctionType>(sym->type);
    Assert(ft);
    return ft->isExported;
}


static bool
lIsExternC(const Symbol *sym) {
    const FunctionType *ft = CastType<FunctionType>(sym->type);
    Assert(ft);
    return ft->isExternC;
}


bool
Module::writeDeps(const char *fn) {
  std::cout << "writing dependencies to file " << fn << std::endl;
  FILE *file = fopen(fn,"w");
  if (!file) {
    perror("fopen");
    return false;
  }

  for (std::set<std::string>::const_iterator it=registeredDependencies.begin();
       it != registeredDependencies.end();
       ++it)
    fprintf(file,"%s\n",it->c_str());
  return true;
}


std::string emitOffloadParamStruct(const std::string &paramStructName,
                                   const Symbol *sym,
                                   const FunctionType *fct)
{
  std::stringstream out;
  out << "struct " << paramStructName << " {" << std::endl;

  for (int i=0;i<fct->GetNumParameters();i++) {
    const Type *orgParamType = fct->GetParameterType(i);
    if (orgParamType->IsPointerType() || orgParamType->IsArrayType()) {
      /* we're passing pointers separately -- no pointers in that struct... */
      continue;
    }

    // const reference parameters can be passed as copies.
    const Type *paramType;
    if (orgParamType->IsReferenceType()) {
      if (!orgParamType->IsConstType()) {
        Error(sym->pos,"When emitting offload-stubs, \"export\"ed functions cannot have non-const reference-type parameters.\n");
      }
      const ReferenceType *refType
        = dynamic_cast<const ReferenceType*>(orgParamType);
      paramType = refType->GetReferenceTarget()->GetAsNonConstType();
    } else {
      paramType = orgParamType->GetAsNonConstType();
    }
    std::string paramName = fct->GetParameterName(i);
    std::string paramTypeName = paramType->GetString();

    std::string tmpArgDecl = paramType->GetCDeclaration(paramName);
    out << "   " << tmpArgDecl << ";" << std::endl;
  }

  out << "};" << std::endl;
  return out.str();
}

bool
Module::writeDevStub(const char *fn)
{
  FILE *file = fopen(fn, "w");
  if (!file) {
    perror("fopen");
    return false;
  }
  fprintf(file, "//\n// %s\n// (device stubs automatically generated by the ispc compiler.)\n", fn);
  fprintf(file, "// DO NOT EDIT THIS FILE.\n//\n\n");
  fprintf(file,"#include \"ispc/dev/offload.h\"\n\n");

  fprintf(file, "#include <stdint.h>\n\n");

  // Collect single linear arrays of the *exported* functions (we'll
  // treat those as "__kernel"s in IVL -- "extern" functions will only
  // be used for dev-dev function calls; only "export" functions will
  // get exported to the host
  std::vector<Symbol *> exportedFuncs;
  m->symbolTable->GetMatchingFunctions(lIsExported, &exportedFuncs);

  // Get all of the struct, vector, and enumerant types used as function
  // parameters.  These vectors may have repeats.
  std::vector<const StructType *> exportedStructTypes;
  std::vector<const EnumType *> exportedEnumTypes;
  std::vector<const VectorType *> exportedVectorTypes;
  lGetExportedParamTypes(exportedFuncs, &exportedStructTypes,
                         &exportedEnumTypes, &exportedVectorTypes);

  // And print them
  lEmitVectorTypedefs(exportedVectorTypes, file);
  lEmitEnumDecls(exportedEnumTypes, file);
  lEmitStructDecls(exportedStructTypes, file);

  fprintf(file, "#ifdef __cplusplus\n");
  fprintf(file, "namespace ispc {\n");
  fprintf(file, "#endif // __cplusplus\n");

  fprintf(file, "\n");
  fprintf(file, "///////////////////////////////////////////////////////////////////////////\n");
  fprintf(file, "// Functions exported from ispc code\n");
  fprintf(file, "// (so the dev stub knows what to call)\n");
  fprintf(file, "///////////////////////////////////////////////////////////////////////////\n");
  lPrintFunctionDeclarations(file, exportedFuncs, true);

  fprintf(file, "#ifdef __cplusplus\n");
  fprintf(file, "}/* end namespace */\n");
  fprintf(file, "#endif // __cplusplus\n");

  fprintf(file, "\n");
  fprintf(file, "///////////////////////////////////////////////////////////////////////////\n");
  fprintf(file, "// actual dev stubs\n");
  fprintf(file, "///////////////////////////////////////////////////////////////////////////\n");

  fprintf(file, "// note(iw): due to some linking issues offload stubs *only* work under C++\n");
  fprintf(file, "extern \"C\" {\n\n");
  for (unsigned int i = 0; i < exportedFuncs.size(); ++i) {
    const Symbol *sym = exportedFuncs[i];
    Assert(sym);
    const FunctionType *fct = CastType<FunctionType>(sym->type);
    Assert(fct);

    if (!fct->GetReturnType()->IsVoidType()) {
      //Error(sym->pos,"When emitting offload-stubs, \"export\"ed functions cannot have non-void return types.\n");
      Warning(sym->pos,"When emitting offload-stubs, ignoring \"export\"ed function with non-void return types.\n");
      continue;
    }

    // -------------------------------------------------------
    // first, emit a struct that holds the parameters
    // -------------------------------------------------------
    std::string paramStructName = std::string("__ispc_dev_stub_")+sym->name;
    std::string paramStruct = emitOffloadParamStruct(paramStructName,sym,fct);
    fprintf(file,"%s\n",paramStruct.c_str());
    // -------------------------------------------------------
    // then, emit a fct stub that unpacks the parameters and pointers
    // -------------------------------------------------------
    fprintf(file,"void __ispc_dev_stub_%s(\n"
            "            uint32_t         in_BufferCount,\n"
            "            void**           in_ppBufferPointers,\n"
            "            uint64_t*        in_pBufferLengths,\n"
            "            void*            in_pMiscData,\n"
            "            uint16_t         in_MiscDataLength,\n"
            "            void*            in_pReturnValue,\n"
            "            uint16_t         in_ReturnValueLength)\n",
            sym->name.c_str()
            );
    fprintf(file,"{\n");
    fprintf(file,"  struct %s args;\n  memcpy(&args,in_pMiscData,sizeof(args));\n",
            paramStructName.c_str());
    std::stringstream funcall;

    funcall << "ispc::" << sym->name << "(";
    for (int i=0;i<fct->GetNumParameters();i++) {
      // get param type and make it non-const, so we can write while unpacking
      // const Type *paramType = fct->GetParameterType(i)->GetAsNonConstType();
      const Type *paramType;// = fct->GetParameterType(i)->GetAsNonConstType();
      const Type *orgParamType = fct->GetParameterType(i);
      if (orgParamType->IsReferenceType()) {
        if (!orgParamType->IsConstType()) {
          Error(sym->pos,"When emitting offload-stubs, \"export\"ed functions cannot have non-const reference-type parameters.\n");
        }
        const ReferenceType *refType
          = dynamic_cast<const ReferenceType*>(orgParamType);
        paramType = refType->GetReferenceTarget()->GetAsNonConstType();
      } else {
        paramType = orgParamType->GetAsNonConstType();
      }

      std::string paramName = fct->GetParameterName(i);
      std::string paramTypeName = paramType->GetString();

      if (i) funcall << ", ";
      std::string tmpArgName = std::string("_")+paramName;
      if (paramType->IsPointerType() || paramType->IsArrayType()) {
        std::string tmpArgDecl = paramType->GetCDeclaration(tmpArgName);
        fprintf(file,"  %s;\n",
                tmpArgDecl.c_str());
        fprintf(file,"  (void *&)%s = ispc_dev_translate_pointer(*in_ppBufferPointers++);\n",
                tmpArgName.c_str());
        funcall << tmpArgName;
      } else {
        funcall << "args." << paramName;
      }
    }
    funcall << ");";
    fprintf(file,"  %s\n",funcall.str().c_str());
    fprintf(file,"}\n\n");
  }

  // end extern "C"
  fprintf(file, "}/* end extern C */\n");

  fclose(file);
  return true;
}



bool
Module::writeHostStub(const char *fn)
{
  FILE *file = fopen(fn, "w");
  if (!file) {
    perror("fopen");
    return false;
  }
  fprintf(file, "//\n// %s\n// (device stubs automatically generated by the ispc compiler.)\n", fn);
  fprintf(file, "// DO NOT EDIT THIS FILE.\n//\n\n");
  fprintf(file,"#include \"ispc/host/offload.h\"\n\n");
  fprintf(file,"// note(iw): Host stubs do not get extern C linkage -- dev-side already uses that for the same symbols.\n\n");
  //fprintf(file,"#ifdef __cplusplus\nextern \"C\" {\n#endif // __cplusplus\n");

  fprintf(file, "#ifdef __cplusplus\nnamespace ispc {\n#endif // __cplusplus\n\n");

  // Collect single linear arrays of the *exported* functions (we'll
  // treat those as "__kernel"s in IVL -- "extern" functions will only
  // be used for dev-dev function calls; only "export" functions will
  // get exported to the host
  std::vector<Symbol *> exportedFuncs;
  m->symbolTable->GetMatchingFunctions(lIsExported, &exportedFuncs);

  // Get all of the struct, vector, and enumerant types used as function
  // parameters.  These vectors may have repeats.
  std::vector<const StructType *> exportedStructTypes;
  std::vector<const EnumType *> exportedEnumTypes;
  std::vector<const VectorType *> exportedVectorTypes;
  lGetExportedParamTypes(exportedFuncs, &exportedStructTypes,
                         &exportedEnumTypes, &exportedVectorTypes);

  // And print them
  lEmitVectorTypedefs(exportedVectorTypes, file);
  lEmitEnumDecls(exportedEnumTypes, file);
  lEmitStructDecls(exportedStructTypes, file);

  fprintf(file, "\n");
  fprintf(file, "///////////////////////////////////////////////////////////////////////////\n");
  fprintf(file, "// host-side stubs for dev-side ISPC fucntion(s)\n");
  fprintf(file, "///////////////////////////////////////////////////////////////////////////\n");
  for (unsigned int i = 0; i < exportedFuncs.size(); ++i) {
    const Symbol *sym = exportedFuncs[i];
    Assert(sym);
    const FunctionType *fct = CastType<FunctionType>(sym->type);
    Assert(fct);

    if (!fct->GetReturnType()->IsVoidType()) {
      Warning(sym->pos,"When emitting offload-stubs, ignoring \"export\"ed function with non-void return types.\n");
      continue;
    }



    // -------------------------------------------------------
    // first, emit a struct that holds the parameters
    // -------------------------------------------------------
    std::string paramStructName = std::string("__ispc_dev_stub_")+sym->name;
    std::string paramStruct = emitOffloadParamStruct(paramStructName,sym,fct);
    fprintf(file,"%s\n",paramStruct.c_str());
    // -------------------------------------------------------
    // then, emit a fct stub that unpacks the parameters and pointers
    // -------------------------------------------------------

    std::string decl = fct->GetCDeclaration(sym->name);
    fprintf(file, "extern %s {\n", decl.c_str());
    int numPointers = 0;
    fprintf(file, "  %s __args;\n",paramStructName.c_str());

    // ------------------------------------------------------------------
    // write args, and save pointers for later
    // ------------------------------------------------------------------
    std::stringstream pointerArgs;
    for (int i=0;i<fct->GetNumParameters();i++) {
      const Type *orgParamType = fct->GetParameterType(i);
      std::string paramName = fct->GetParameterName(i);
      if (orgParamType->IsPointerType() || orgParamType->IsArrayType()) {
        /* we're passing pointers separately -- no pointers in that struct... */
        if (numPointers)
          pointerArgs << ",";
        pointerArgs << "(void*)" << paramName;
        numPointers++;
        continue;
      }

      fprintf(file,"  __args.%s = %s;\n",
              paramName.c_str(),paramName.c_str());
    }
    // ------------------------------------------------------------------
    // writer pointer list
    // ------------------------------------------------------------------
    if (numPointers == 0)
      pointerArgs << "NULL";
    fprintf(file,"  void *ptr_args[] = { %s };\n" ,pointerArgs.str().c_str());

    // ------------------------------------------------------------------
    // ... and call the kernel with those args
    // ------------------------------------------------------------------
    fprintf(file,"  static ispc_kernel_handle_t kernel_handle = NULL;\n");
    fprintf(file,"  if (!kernel_handle) kernel_handle = ispc_host_get_kernel_handle(\"__ispc_dev_stub_%s\");\n",
            sym->name.c_str());
    fprintf(file,"  assert(kernel_handle);\n");
    fprintf(file,
            "  ispc_host_call_kernel(kernel_handle,\n"
            "                        &__args, sizeof(__args),\n"
            "                        ptr_args,%i);\n",
            numPointers);
    fprintf(file,"}\n\n");
  }

  // end extern "C"
  fprintf(file, "#ifdef __cplusplus\n");
  fprintf(file, "}/* namespace */\n");
  fprintf(file, "#endif // __cplusplus\n");
  // fprintf(file, "#ifdef __cplusplus\n");
  // fprintf(file, "}/* end extern C */\n");
  // fprintf(file, "#endif // __cplusplus\n");

  fclose(file);
  return true;
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

    if (g->emitInstrumentation) {
        fprintf(f, "#define ISPC_INSTRUMENTATION 1\n");
        fprintf(f, "extern \"C\" {\n");
        fprintf(f, "  void ISPCInstrument(const char *fn, const char *note, int line, uint64_t mask);\n");
        fprintf(f, "}\n");
    }

    // end namespace
    fprintf(f, "\n");
    fprintf(f, "\n#ifdef __cplusplus\nnamespace ispc { /* namespace */\n#endif // __cplusplus\n");


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

    // Go through the explicitly exported types
    for (int i = 0; i < (int)exportedTypes.size(); ++i) {
        if (const StructType *st = CastType<StructType>(exportedTypes[i].first))
            exportedStructTypes.push_back(st->GetAsUniformType());
        else if (const EnumType *et = CastType<EnumType>(exportedTypes[i].first))
            exportedEnumTypes.push_back(et->GetAsUniformType());
        else if (const VectorType *vt = CastType<VectorType>(exportedTypes[i].first))
            exportedVectorTypes.push_back(vt->GetAsUniformType());
        else
            FATAL("Unexpected type in export list");
    }

    // And print them
    lEmitVectorTypedefs(exportedVectorTypes, f);
    lEmitEnumDecls(exportedEnumTypes, f);
    lEmitStructDecls(exportedStructTypes, f);

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
    fprintf(f, "\n");
    fprintf(f, "\n#ifdef __cplusplus\n} /* namespace */\n#endif // __cplusplus\n");

    // end guard
    fprintf(f, "\n#endif // %s\n", guard.c_str());

    fclose(f);
    return true;
}

struct DispatchHeaderInfo {
  bool EmitUnifs;
  bool EmitFuncs;
  bool EmitFrontMatter;
  bool EmitBackMatter;
  bool Emit4;
  bool Emit8;
  bool Emit16;
  FILE *file;
  const char *fn;
};

bool
Module::writeDispatchHeader(DispatchHeaderInfo *DHI) {
  FILE *f = DHI->file;
  
  if (DHI->EmitFrontMatter) {
    fprintf(f, "//\n// %s\n// (Header automatically generated by the ispc compiler.)\n", DHI->fn);
    fprintf(f, "// DO NOT EDIT THIS FILE.\n//\n\n");
  }
    // Create a nice guard string from the filename, turning any
    // non-number/letter characters into underbars
    std::string guard = "ISPC_";
    const char *p = DHI->fn;
    while (*p) {
        if (isdigit(*p))
            guard += *p;
        else if (isalpha(*p))
            guard += toupper(*p);
        else
            guard += "_";
        ++p;
    }
    if (DHI->EmitFrontMatter) {
      fprintf(f, "#ifndef %s\n#define %s\n\n", guard.c_str(), guard.c_str());

      fprintf(f, "#include <stdint.h>\n\n");


      if (g->emitInstrumentation) {
        fprintf(f, "#define ISPC_INSTRUMENTATION 1\n");
        fprintf(f, "extern \"C\" {\n");
        fprintf(f, "  void ISPCInstrument(const char *fn, const char *note, int line, uint64_t mask);\n");
        fprintf(f, "}\n");
      }

      // end namespace
      fprintf(f, "\n");
      fprintf(f, "\n#ifdef __cplusplus\nnamespace ispc { /* namespace */\n#endif // __cplusplus\n\n");
      DHI->EmitFrontMatter = false;
    }


    // Collect single linear arrays of the exported and extern "C"
    // functions
    std::vector<Symbol *> exportedFuncs, externCFuncs;
    m->symbolTable->GetMatchingFunctions(lIsExported, &exportedFuncs);
    m->symbolTable->GetMatchingFunctions(lIsExternC, &externCFuncs);
     
    int programCount = g->target->getVectorWidth();
    
    if ((DHI->Emit4 && (programCount == 4)) || 
        (DHI->Emit8 && (programCount == 8)) ||
        (DHI->Emit16 && (programCount == 16))) {
        // Get all of the struct, vector, and enumerant types used as function
        // parameters.  These vectors may have repeats.
        std::vector<const StructType *> exportedStructTypes;
        std::vector<const EnumType *> exportedEnumTypes;
        std::vector<const VectorType *> exportedVectorTypes;
        lGetExportedParamTypes(exportedFuncs, &exportedStructTypes,
                               &exportedEnumTypes, &exportedVectorTypes);
        lGetExportedParamTypes(externCFuncs, &exportedStructTypes,
                               &exportedEnumTypes, &exportedVectorTypes);
        
        // Go through the explicitly exported types
        for (int i = 0; i < (int)exportedTypes.size(); ++i) {
          if (const StructType *st = CastType<StructType>(exportedTypes[i].first))
            exportedStructTypes.push_back(st->GetAsUniformType());
          else if (const EnumType *et = CastType<EnumType>(exportedTypes[i].first))
            exportedEnumTypes.push_back(et->GetAsUniformType());
          else if (const VectorType *vt = CastType<VectorType>(exportedTypes[i].first))
            exportedVectorTypes.push_back(vt->GetAsUniformType());
          else
            FATAL("Unexpected type in export list");
        }

        
        // And print them
        if (DHI->EmitUnifs) {
          lEmitVectorTypedefs(exportedVectorTypes, f);
          lEmitEnumDecls(exportedEnumTypes, f);
        }
        lEmitStructDecls(exportedStructTypes, f, DHI->EmitUnifs);
        
        // Update flags
        DHI->EmitUnifs = false;
        if (programCount == 4) {
          DHI->Emit4 = false;
        } 
        else if (programCount == 8) {
          DHI->Emit8 = false;
        }
        else if (programCount == 16) {
          DHI->Emit16 = false;
        }
    }
    if (DHI->EmitFuncs) {
      // emit function declarations for exported stuff...
      if (exportedFuncs.size() > 0) {
        fprintf(f, "\n");
        fprintf(f, "///////////////////////////////////////////////////////////////////////////\n");
        fprintf(f, "// Functions exported from ispc code\n");
        fprintf(f, "///////////////////////////////////////////////////////////////////////////\n");
        lPrintFunctionDeclarations(f, exportedFuncs, 1, true);
        fprintf(f, "\n");
      }
      DHI->EmitFuncs = false;
    }

    if (DHI->EmitBackMatter) {
      // end namespace
      fprintf(f, "\n");
      fprintf(f, "\n#ifdef __cplusplus\n} /* namespace */\n#endif // __cplusplus\n");
      
      // end guard
      fprintf(f, "\n#endif // %s\n", guard.c_str());
      DHI->EmitBackMatter = false;
    }
    
    return true;
}

void
Module::execPreprocessor(const char *infilename, llvm::raw_string_ostream *ostream) const
{
    clang::CompilerInstance inst;
    inst.createFileManager();

    llvm::raw_fd_ostream stderrRaw(2, false);

    clang::DiagnosticOptions *diagOptions = new clang::DiagnosticOptions();
    clang::TextDiagnosticPrinter *diagPrinter =
        new clang::TextDiagnosticPrinter(stderrRaw, diagOptions);
    
    llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagIDs(new clang::DiagnosticIDs);
    clang::DiagnosticsEngine *diagEngine =
        new clang::DiagnosticsEngine(diagIDs, diagOptions, diagPrinter);
    
    inst.setDiagnostics(diagEngine);

#if defined(LLVM_3_2) || defined(LLVM_3_3) || defined(LLVM_3_4)
    clang::TargetOptions &options = inst.getTargetOpts();
#else // LLVM 3.5+
    const std::shared_ptr< clang::TargetOptions > &options = 
          std::make_shared< clang::TargetOptions >(inst.getTargetOpts());
#endif

    llvm::Triple triple(module->getTargetTriple());
    if (triple.getTriple().empty()) {
        triple.setTriple(llvm::sys::getDefaultTargetTriple());
    }

#if defined(LLVM_3_2) || defined(LLVM_3_3) || defined(LLVM_3_4)
    options.Triple = triple.getTriple();
#else // LLVM 3.5+
   options->Triple = triple.getTriple();
#endif

#if defined(LLVM_3_2)
    clang::TargetInfo *target =
        clang::TargetInfo::CreateTargetInfo(inst.getDiagnostics(), options);
#elif defined(LLVM_3_3) || defined(LLVM_3_4)
    clang::TargetInfo *target =
        clang::TargetInfo::CreateTargetInfo(inst.getDiagnostics(), &options);
#else // LLVM 3.5+
    clang::TargetInfo *target =
        clang::TargetInfo::CreateTargetInfo(inst.getDiagnostics(), options);
#endif

    inst.setTarget(target);
    inst.createSourceManager(inst.getFileManager());
    clang::FrontendInputFile inputFile(infilename, clang::IK_None);
    inst.InitializeSourceManager(inputFile);

    // Don't remove comments in the preprocessor, so that we can accurately
    // track the source file position by handling them ourselves.
    inst.getPreprocessorOutputOpts().ShowComments = 1;

#if !defined(LLVM_3_2) // LLVM 3.3+
    inst.getPreprocessorOutputOpts().ShowCPP = 1;
#endif

    clang::HeaderSearchOptions &headerOpts = inst.getHeaderSearchOpts();
    headerOpts.UseBuiltinIncludes = 0;
    headerOpts.UseStandardSystemIncludes = 0;
    headerOpts.UseStandardCXXIncludes = 0;
    if (g->debugPrint)
        headerOpts.Verbose = 1;
    for (int i = 0; i < (int)g->includePath.size(); ++i) {
        headerOpts.AddPath(g->includePath[i], clang::frontend::Angled,
#if defined(LLVM_3_2)
                           true /* is user supplied */,
#endif
                           false /* not a framework */,
                           true /* ignore sys root */);
    }

    clang::PreprocessorOptions &opts = inst.getPreprocessorOpts();

    // Add defs for ISPC and PI
    opts.addMacroDef("ISPC");
    opts.addMacroDef("PI=3.1415926535");

    // Add #define for current compilation target
    char targetMacro[128];
    sprintf(targetMacro, "ISPC_TARGET_%s", g->target->GetISAString());
    char *p = targetMacro;
    while (*p) {
        *p = toupper(*p);
        if (*p == '-') *p = '_';
        ++p;
    }
    opts.addMacroDef(targetMacro);

    if (g->target->is32Bit())
        opts.addMacroDef("ISPC_POINTER_SIZE=32");
    else
        opts.addMacroDef("ISPC_POINTER_SIZE=64");

    if (g->target->hasHalf())
        opts.addMacroDef("ISPC_TARGET_HAS_HALF");
    if (g->target->hasRand())
        opts.addMacroDef("ISPC_TARGET_HAS_RAND");
    if (g->target->hasTranscendentals())
        opts.addMacroDef("ISPC_TARGET_HAS_TRANSCENDENTALS");
    if (g->opt.forceAlignedMemory)
        opts.addMacroDef("ISPC_FORCE_ALIGNED_MEMORY");

    opts.addMacroDef("ISPC_MAJOR_VERSION=1");
    opts.addMacroDef("ISPC_MINOR_VERSION=4");

    if (g->includeStdlib) {
        if (g->opt.disableAsserts)
            opts.addMacroDef("assert(x)=");
        else
            opts.addMacroDef("assert(x)=__assert(#x, x)");
    }

    for (unsigned int i = 0; i < g->cppArgs.size(); ++i) {
        // Sanity check--should really begin with -D
        if (g->cppArgs[i].substr(0,2) == "-D") {
            opts.addMacroDef(g->cppArgs[i].substr(2));
        }
    }

    inst.getLangOpts().LineComment = 1;

#if !defined(LLVM_3_2) && !defined(LLVM_3_3) && !defined(LLVM_3_4) // LLVM 3.5+
    inst.createPreprocessor(clang::TU_Complete);
#else
    inst.createPreprocessor();
#endif

    diagPrinter->BeginSourceFile(inst.getLangOpts(), &inst.getPreprocessor());
    clang::DoPrintPreprocessedInput(inst.getPreprocessor(),
                                    ostream, inst.getPreprocessorOutputOpts());
    diagPrinter->EndSourceFile();
}


// Given an output filename of the form "foo.obj", and an ISA name like
// "avx", return a string with the ISA name inserted before the original
// filename's suffix, like "foo_avx.obj".
static std::string
lGetTargetFileName(const char *outFileName, const char *isaString) {
    char *targetOutFileName = new char[strlen(outFileName) + 16];
    if (strrchr(outFileName, '.') != NULL) {
        // Copy everything up to the last '.'
        int count = strrchr(outFileName, '.') - outFileName;
        strncpy(targetOutFileName, outFileName, count);
        targetOutFileName[count] = '\0';

        // Add the ISA name
        strcat(targetOutFileName, "_");
        strcat(targetOutFileName, isaString);

        // And finish with the original file suffiz
        strcat(targetOutFileName, strrchr(outFileName, '.'));
    }
    else {
        // Can't find a '.' in the filename, so just append the ISA suffix
        // to what we weregiven
        strcpy(targetOutFileName, outFileName);
        strcat(targetOutFileName, "_");
        strcat(targetOutFileName, isaString);
    }
    return targetOutFileName;
}


// Given a comma-delimited string with one or more compilation targets of
// the form "sse2,avx-x2", return a vector of strings where each returned
// string holds one of the targets from the given string.
static std::vector<std::string>
lExtractTargets(const char *target) {
    std::vector<std::string> targets;
    const char *tstart = target;
    bool done = false;
    while (!done) {
        const char *tend = strchr(tstart, ',');
        if (tend == NULL) {
            done = true;
            tend = strchr(tstart, '\0');
        }
        targets.push_back(std::string(tstart, tend));
        tstart = tend+1;
    }
    return targets;
}


static bool
lSymbolIsExported(const Symbol *s) {
    return s->exportedFunction != NULL;
}


// Small structure to hold pointers to the various different versions of a
// llvm::Function that were compiled for different compilation target ISAs.
struct FunctionTargetVariants {
    FunctionTargetVariants() {
      for (int i = 0; i < Target::NUM_ISAS; ++i) {
            func[i] = NULL;
            FTs[i] = NULL;
      }
    }
    // The func array is indexed with the Target::ISA enumerant.  Some
    // values may be NULL, indicating that the original function wasn't
    // compiled to the corresponding target ISA.
    llvm::Function *func[Target::NUM_ISAS];
    const FunctionType *FTs[Target::NUM_ISAS];
};


// Given the symbol table for a module, return a map from function names to
// FunctionTargetVariants for each function that was defined with the
// 'export' qualifier in ispc.
static void
lGetExportedFunctions(SymbolTable *symbolTable,
                      std::map<std::string, FunctionTargetVariants> &functions) {
    std::vector<Symbol *> syms;
    symbolTable->GetMatchingFunctions(lSymbolIsExported, &syms);
    for (unsigned int i = 0; i < syms.size(); ++i) {
        FunctionTargetVariants &ftv = functions[syms[i]->name];
        ftv.func[g->target->getISA()] = syms[i]->exportedFunction;
        ftv.FTs[g->target->getISA()] = CastType<FunctionType>(syms[i]->type);
    }
}


struct RewriteGlobalInfo {
    RewriteGlobalInfo(llvm::GlobalVariable *g = NULL, llvm::Constant *i = NULL,
                      SourcePos p = SourcePos()) {
        gv = g;
        init = i;
        pos = p;
    }

    llvm::GlobalVariable *gv;
    llvm::Constant *init;
    SourcePos pos;
};

// Grab all of the global value definitions from the module and change them
// to be declarations; we'll emit a single definition of each global in the
// final module used with the dispatch functions, so that we don't have
// multiple definitions of them, one in each of the target-specific output
// files.
static void
lExtractAndRewriteGlobals(llvm::Module *module,
                          std::vector<RewriteGlobalInfo> *globals) {
    llvm::Module::global_iterator iter;
    for (iter = module->global_begin(); iter != module->global_end(); ++iter) {
        llvm::GlobalVariable *gv = iter;
        // Is it a global definition?
        if (gv->getLinkage() == llvm::GlobalValue::ExternalLinkage &&
            gv->hasInitializer()) {
            // Turn this into an 'extern 'declaration by clearing its
            // initializer.
            llvm::Constant *init = gv->getInitializer();
            gv->setInitializer(NULL);

            Symbol *sym =
                m->symbolTable->LookupVariable(gv->getName().str().c_str());
            Assert(sym != NULL);
            globals->push_back(RewriteGlobalInfo(gv, init, sym->pos));
        }
    }
}


// This function emits a global variable definition for each global that
// was turned into a declaration in the target-specific output file.
static void
lAddExtractedGlobals(llvm::Module *module,
                     std::vector<RewriteGlobalInfo> globals[Target::NUM_ISAS]) {
    // Find the first element in the globals[] array that has values stored
    // in it.  All elements of this array should either have empty vectors
    // (if we didn't compile to the corresponding ISA or if there are no
    // globals), or should have the same number of vector elements as the
    // other non-empty vectors.
    int firstActive = -1;
    for (int i = 0; i < Target::NUM_ISAS; ++i)
        if (globals[i].size() > 0) {
            firstActive = i;
            break;
        }

    if (firstActive == -1)
        // no globals
        return;

    for (unsigned int i = 0; i < globals[firstActive].size(); ++i) {
        RewriteGlobalInfo &rgi = globals[firstActive][i];
        llvm::GlobalVariable *gv = rgi.gv;
        llvm::Type *type = gv->getType()->getElementType();
        llvm::Constant *initializer = rgi.init;

        // Create a new global in the given model that matches the original
        // global
        llvm::GlobalVariable *newGlobal =
            new llvm::GlobalVariable(*module, type, gv->isConstant(),
                                     llvm::GlobalValue::ExternalLinkage,
                                     initializer, gv->getName());
        newGlobal->copyAttributesFrom(gv);

        // For all of the other targets that we actually generated code
        // for, make sure the global we just created is compatible with the
        // global from the module for that target.
        for (int j = firstActive + 1; j < Target::NUM_ISAS; ++j) {
            if (globals[j].size() > 0) {
                // There should be the same number of globals in the other
                // vectors, in the same order.
                Assert(globals[firstActive].size() == globals[j].size());
                llvm::GlobalVariable *gv2 = globals[j][i].gv;
                Assert(gv2->getName() == gv->getName());

                // It is possible that the types may not match, though--for
                // example, this happens with varying globals if we compile
                // to different vector widths.
                if (gv2->getType() != gv->getType())
                    Warning(rgi.pos, "Mismatch in size/layout of global "
                          "variable \"%s\" with different targets. "
                          "Globals must not include \"varying\" types or arrays "
                          "with size based on programCount when compiling to "
                          "targets with differing vector widths.",
                          gv->getName().str().c_str());
            }
        }
    }
}

static llvm::FunctionType *
lGetVaryingDispatchType(FunctionTargetVariants &funcs) {
  llvm::Type *ptrToInt8Ty = llvm::Type::getInt8PtrTy(*g->ctx);
  llvm::FunctionType *resultFuncTy = NULL;

  for (int i = 0; i < Target::NUM_ISAS; ++i) {
    if (funcs.func[i] == NULL)  {
      continue;
    }
    else {
      bool foundVarying = false;
      const FunctionType *ft = funcs.FTs[i];
      resultFuncTy = funcs.func[i]->getFunctionType();

      int numArgs = ft->GetNumParameters();
      llvm::SmallVector<llvm::Type *, 8> ftype;
      for (int j = 0; j < numArgs; ++j) {
        ftype.push_back(resultFuncTy->getParamType(j));
      }

      for (int j = 0; j < numArgs; ++j) {
        const Type *arg = ft->GetParameterType(j);

        if (arg->IsPointerType()) {
          const Type *baseType = CastType<PointerType>(arg)->GetBaseType();
          // For each varying type pointed to, swap the LLVM pointer type
          // with i8 * (as close as we can get to void *)
          if (baseType->IsVaryingType()) {
            ftype[j] = ptrToInt8Ty;
            foundVarying = true;
          }
        }
      }
      if (foundVarying) {
        resultFuncTy = llvm::FunctionType::get(resultFuncTy->getReturnType(), ftype, false);
      }
    }
  }
  
  // We should've found at least one variant here
  // or else something fishy is going on.
  Assert(resultFuncTy);
  
  return resultFuncTy;
}

/** Create the dispatch function for an exported ispc function.
    This function checks to see which vector ISAs the system the
    code is running on supports and calls out to the best available
    variant that was generated at compile time.

    @param module      Module in which to create the dispatch function.
    @param setISAFunc  Pointer to the __set_system_isa() function defined
                       in builtins-dispatch.ll (which is linked into the
                       given module before we get here.)
    @param systemBestISAPtr  Pointer to the module-local __system_best_isa
                             variable, which holds a value of the Target::ISA
                             enumerant giving the most capable ISA that the
                             system supports.
    @param name        Name of the function for which we're generating a
                       dispatch function
    @param funcs       Target-specific variants of the exported function.
*/
static void
lCreateDispatchFunction(llvm::Module *module, llvm::Function *setISAFunc,
                        llvm::Value *systemBestISAPtr, const std::string &name,
                        FunctionTargetVariants &funcs) {
    // The llvm::Function pointers in funcs are pointers to functions in
    // different llvm::Modules, so we can't call them directly.  Therefore,
    // we'll start by generating an 'extern' declaration of each one that
    // we have in the current module so that we can then call out to that.
    llvm::Function *targetFuncs[Target::NUM_ISAS];
    llvm::FunctionType *ftypes[Target::NUM_ISAS];

    for (int i = 0; i < Target::NUM_ISAS; ++i) {
        if (funcs.func[i] == NULL) {
            targetFuncs[i] = NULL;
            ftypes[i] = NULL;
            continue;
        }

        // Grab the type of the function as well.  Note that the various
        // functions will have different types if they have arguments that
        // are pointers to structs, due to the fact that we mangle LLVM
        // struct type names with the target vector width.  However,
        // because we only allow uniform stuff to pass through the
        // export'ed function layer, they should all have the same memory
        // layout, so this is benign..
        // JCB nomosoa - not anymore...
        // add a helper to see if this type has any varying thingies? 
        // might be hard to detect....
        // If so, return a new type with the pointers to those replaced
        // by i8 *'s.
        //        if (ftype == NULL)
        ftypes[i] = funcs.func[i]->getFunctionType();

        targetFuncs[i] =
            llvm::Function::Create(ftypes[i], llvm::GlobalValue::ExternalLinkage,
                                   funcs.func[i]->getName(), module);
    }

    // New helper function checks to see if we need to rewrite the
    // type for the dispatch function in case of pointers to varyings
    llvm::FunctionType *ftype = lGetVaryingDispatchType(funcs);
    
    bool voidReturn = ftype->getReturnType()->isVoidTy();

    // Now we can emit the definition of the dispatch function..
    llvm::Function *dispatchFunc =
        llvm::Function::Create(ftype, llvm::GlobalValue::ExternalLinkage,
                               name.c_str(), module);
    llvm::BasicBlock *bblock =
        llvm::BasicBlock::Create(*g->ctx, "entry", dispatchFunc);

    // Start by calling out to the function that determines the system's
    // ISA and sets __system_best_isa, if it hasn't been set yet.
    llvm::CallInst::Create(setISAFunc, "", bblock);

    // Now we can load the system's ISA enuemrant
    llvm::Value *systemISA =
        new llvm::LoadInst(systemBestISAPtr, "system_isa", bblock);

    // Now emit code that works backwards though the available variants of
    // the function.  We'll call out to the first one we find that will run
    // successfully on the system the code is running on.  In working
    // through the candidate ISAs here backward, we're taking advantage of
    // the expectation that they are ordered in the Target::ISA enumerant
    // from least to most capable.
    for (int i = Target::NUM_ISAS-1; i >= 0; --i) {
        if (targetFuncs[i] == NULL)
            continue;

        // Emit code to see if the system can run the current candidate
        // variant successfully--"is the system's ISA enuemrant value >=
        // the enumerant value of the current candidate?"
        llvm::Value *ok =
            llvm::CmpInst::Create(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_SGE,
                                  systemISA, LLVMInt32(i), "isa_ok", bblock);
        llvm::BasicBlock *callBBlock =
            llvm::BasicBlock::Create(*g->ctx, "do_call", dispatchFunc);
        llvm::BasicBlock *nextBBlock =
            llvm::BasicBlock::Create(*g->ctx, "next_try", dispatchFunc);
        llvm::BranchInst::Create(callBBlock, nextBBlock, ok, bblock);

        // Emit the code to make the call call in callBBlock.
        // Just pass through all of the args from the dispatch function to
        // the target-specific function.
        std::vector<llvm::Value *> args;
        llvm::Function::arg_iterator argIter = dispatchFunc->arg_begin();
        llvm::Function::arg_iterator targsIter = targetFuncs[i]->arg_begin();
        for (; argIter != dispatchFunc->arg_end(); ++argIter, ++targsIter) {
          // Check to see if we rewrote any types in the dispatch function.
          // If so, create bitcasts for the appropriate pointer types.
          if (argIter->getType() == targsIter->getType()) {
            args.push_back(argIter);
          }
          else {
            llvm::CastInst *argCast = 
              llvm::CastInst::CreatePointerCast(argIter, targsIter->getType(),
                                                "dpatch_arg_bitcast", callBBlock);
            args.push_back(argCast);
          }

        }
        if (voidReturn) {
            llvm::CallInst::Create(targetFuncs[i], args, "", callBBlock);
            llvm::ReturnInst::Create(*g->ctx, callBBlock);
        }
        else {
            llvm::Value *retValue =
                llvm::CallInst::Create(targetFuncs[i], args, "ret_value",
                                       callBBlock);
            llvm::ReturnInst::Create(*g->ctx, retValue, callBBlock);
        }

        // Otherwise we'll go on to the next candidate and see about that
        // one...
        bblock = nextBBlock;
    }

    // We couldn't find a match that the current system was capable of
    // running.  We'll call abort(); this is a bit of a blunt hammer--it
    // might be preferable to call a user-supplied callback--ISPCError(...)
    // or some such, but we don't want to start imposing too much of a
    // runtime library requirement either...
    llvm::Function *abortFunc = module->getFunction("abort");
    Assert(abortFunc);
    llvm::CallInst::Create(abortFunc, "", bblock);

    // Return an undef value from the function here; we won't get to this
    // point at runtime, but LLVM needs all of the basic blocks to be
    // terminated...
    if (voidReturn)
        llvm::ReturnInst::Create(*g->ctx, bblock);
    else {
        llvm::Value *undefRet = llvm::UndefValue::get(ftype->getReturnType());
        llvm::ReturnInst::Create(*g->ctx, undefRet, bblock);
    }
}

// Given a map that holds the mapping from each of the 'export'ed functions
// in the ispc program to the target-specific variants of the function,
// create a llvm::Module that has a dispatch function for each exported
// function that checks the system's capabilities and picks the most
// appropriate compiled variant of the function.
static llvm::Module *
lCreateDispatchModule(std::map<std::string, FunctionTargetVariants> &functions) {
    llvm::Module *module = new llvm::Module("dispatch_module", *g->ctx);

    // First, link in the definitions from the builtins-dispatch.ll file.
    extern unsigned char builtins_bitcode_dispatch[];
    extern int builtins_bitcode_dispatch_length;
    AddBitcodeToModule(builtins_bitcode_dispatch,
                       builtins_bitcode_dispatch_length, module);

    // Get pointers to things we need below
    llvm::Function *setFunc = module->getFunction("__set_system_isa");
    Assert(setFunc != NULL);
    llvm::Value *systemBestISAPtr =
        module->getGlobalVariable("__system_best_isa", true);
    Assert(systemBestISAPtr != NULL);

    // For each exported function, create the dispatch function
    std::map<std::string, FunctionTargetVariants>::iterator iter;
    for (iter = functions.begin(); iter != functions.end(); ++iter)
        lCreateDispatchFunction(module, setFunc, systemBestISAPtr,
                                iter->first, iter->second);

    // Do some rudimentary cleanup of the final result and make sure that
    // the module is all ok.
    llvm::PassManager optPM;
    optPM.add(llvm::createGlobalDCEPass());
    optPM.add(llvm::createVerifierPass());
    optPM.run(*module);

    return module;
}


int
Module::CompileAndOutput(const char *srcFile,
                         const char *arch,
                         const char *cpu,
                         const char *target,
                         bool generatePIC,
                         OutputType outputType,
                         const char *outFileName,
                         const char *headerFileName,
                         const char *includeFileName,
                         const char *depsFileName,
                         const char *hostStubFileName,
                         const char *devStubFileName)
{
    if (target == NULL || strchr(target, ',') == NULL) {
        // We're only compiling to a single target
        g->target = new Target(arch, cpu, target, generatePIC);
        if (!g->target->isValid())
            return 1;

        m = new Module(srcFile);
        if (m->CompileFile() == 0) {
            if (outputType == CXX) {
                if (target == NULL || strncmp(target, "generic-", 8) != 0) {
                    Error(SourcePos(), "When generating C++ output, one of the \"generic-*\" "
                          "targets must be used.");
                    return 1;
                }
            }
            else if (outputType == Asm || outputType == Object) {
                if (target != NULL && strncmp(target, "generic-", 8) == 0) {
                    Error(SourcePos(), "When using a \"generic-*\" compilation target, "
                          "%s output can not be used.",
                          (outputType == Asm) ? "assembly" : "object file");
                    return 1;
                }
            }

            if (outFileName != NULL)
                if (!m->writeOutput(outputType, outFileName, includeFileName))
                    return 1;
            if (headerFileName != NULL)
                if (!m->writeOutput(Module::Header, headerFileName))
                    return 1;
            if (depsFileName != NULL)
              if (!m->writeOutput(Module::Deps,depsFileName))
                return 1;
            if (hostStubFileName != NULL)
              if (!m->writeOutput(Module::HostStub,hostStubFileName))
                return 1;
            if (devStubFileName != NULL)
              if (!m->writeOutput(Module::DevStub,devStubFileName))
                return 1;
        }
        else
            ++m->errorCount;

        int errorCount = m->errorCount;
        delete m;
        m = NULL;

        delete g->target;
        g->target = NULL;

        return errorCount > 0;
    }
    else {
        if (outputType == CXX) {
            Error(SourcePos(), "Illegal to specify more than one target when "
                  "compiling C++ output.");
            return 1;
        }
        if (srcFile == NULL || !strcmp(srcFile, "-")) {
            Error(SourcePos(), "Compiling programs from standard input isn't "
                  "supported when compiling for multiple targets.  Please use "
                  "an intermediate temporary file.");
            return 1;
        }

        // The user supplied multiple targets
        std::vector<std::string> targets = lExtractTargets(target);
        Assert(targets.size() > 1);

        if (outFileName != NULL && strcmp(outFileName, "-") == 0) {
            Error(SourcePos(), "Multi-target compilation can't generate output "
                  "to stdout.  Please provide an output filename.\n");
            return 1;
        }

        // Make sure that the function names for 'export'ed functions have
        // the target ISA appended to them.
        g->mangleFunctionsWithTarget = true;

        llvm::TargetMachine *targetMachines[Target::NUM_ISAS];
        for (int i = 0; i < Target::NUM_ISAS; ++i)
            targetMachines[i] = NULL;

        std::map<std::string, FunctionTargetVariants> exportedFunctions;
        std::vector<RewriteGlobalInfo> globals[Target::NUM_ISAS];
        int errorCount = 0;
        
        // Handle creating a "generic" header file for multiple targets
        // that use exported varyings
        DispatchHeaderInfo DHI;
        if ((targets.size() > 1) && (headerFileName != NULL)) {
          DHI.file  = fopen(headerFileName, "w");
          if (!DHI.file) {
            perror("fopen");
            return false;
          }
          DHI.fn = headerFileName;
          DHI.EmitUnifs = true;
          DHI.EmitFuncs = true;
          DHI.EmitFrontMatter = true;
          DHI.Emit4 = true;
          DHI.Emit8 = true;
          DHI.Emit16 = true;
          // This is toggled later.
          DHI.EmitBackMatter = false;
        }


        for (unsigned int i = 0; i < targets.size(); ++i) {
            g->target = new Target(arch, cpu, targets[i].c_str(), generatePIC);
            if (!g->target->isValid())
                return 1;

            // Issue an error if we've already compiled to a variant of
            // this target ISA.  (It doesn't make sense to compile to both
            // avx and avx-x2, for example.)
            if (targetMachines[g->target->getISA()] != NULL) {
                Error(SourcePos(), "Can't compile to multiple variants of %s "
                      "target!\n", g->target->GetISAString());
                return 1;
            }
            targetMachines[g->target->getISA()] = g->target->GetTargetMachine();

            m = new Module(srcFile);
            if (m->CompileFile() == 0) {
                // Grab pointers to the exported functions from the module we
                // just compiled, for use in generating the dispatch function
                // later.
                lGetExportedFunctions(m->symbolTable, exportedFunctions);

                lExtractAndRewriteGlobals(m->module, &globals[i]);

                if (outFileName != NULL) {
                    const char *isaName = g->target->GetISAString();
                    std::string targetOutFileName =
                        lGetTargetFileName(outFileName, isaName);
                    if (!m->writeOutput(outputType, targetOutFileName.c_str()))
                        return 1;
                }
            }
            errorCount += m->errorCount;

            // Only write the generate header file, if desired, the first
            // time through the loop here.
            if (headerFileName != NULL) {
              if (i == targets.size()-1) {
                // only print backmatter on the last target.
                DHI.EmitBackMatter = true;
              }
              
              const char *isaName = g->target->GetISAString();
              std::string targetHeaderFileName = 
                lGetTargetFileName(headerFileName, isaName);
              // write out a header w/o target name for the first target only
              if (!m->writeOutput(Module::Header, headerFileName, "", &DHI)) {
                return 1;
              }
              if (!m->writeOutput(Module::Header, targetHeaderFileName.c_str())) {
                return 1;
              }
              if (i == targets.size()-1) {
                fclose(DHI.file);
              }
            }

            delete g->target;
            g->target = NULL;

            // Important: Don't delete the llvm::Module *m here; we need to
            // keep it around so the llvm::Functions *s stay valid for when
            // we generate the dispatch module's functions...
        }

        // Find the first non-NULL target machine from the targets we
        // compiled to above.  We'll use this as the target machine for
        // compiling the dispatch module--this is safe in that it is the
        // least-common-denominator of all of the targets we compiled to.
        llvm::TargetMachine *firstTargetMachine = NULL;
        int i = 0;
        const char *firstISA;
        while (i < Target::NUM_ISAS && firstTargetMachine == NULL) {
            firstISA = Target::ISAToTargetString((Target::ISA) i);
            firstTargetMachine = targetMachines[i++];
        }
        Assert(firstTargetMachine != NULL);

        g->target = new Target(arch, cpu, firstISA, generatePIC);
        if (!g->target->isValid()) {
            return 1;
        }

        llvm::Module *dispatchModule =
            lCreateDispatchModule(exportedFunctions);

        lAddExtractedGlobals(dispatchModule, globals);

        if (outFileName != NULL) {
            if (outputType == Bitcode)
                writeBitcode(dispatchModule, outFileName);
            else
                writeObjectFileOrAssembly(firstTargetMachine, dispatchModule,
                                          outputType, outFileName);
        }

        delete g->target;
        g->target = NULL;


        return errorCount > 0;
    }
}
