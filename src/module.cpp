/*
  Copyright (c) 2010-2022, Intel Corporation
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
#include "builtins.h"
#include "ctx.h"
#include "expr.h"
#include "func.h"
#include "ispc_version.h"
#include "llvmutil.h"
#include "opt.h"
#include "stmt.h"
#include "sym.h"
#include "type.h"
#include "util.h"

#include <algorithm>
#include <ctype.h>
#include <fcntl.h>
#include <iostream>
#include <set>
#include <sstream>
#include <stdarg.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <clang/Basic/TargetInfo.h>
#include <clang/Basic/Version.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Frontend/Utils.h>
#include <clang/Lex/PreprocessorOptions.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IRPrintingPasses.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/PassRegistry.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/DynamicLibrary.h>
#include <llvm/Support/FileUtilities.h>
#include <llvm/Support/FormattedStream.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/Utils/ValueMapper.h>

#ifdef ISPC_XE_ENABLED
#include <llvm/GenXIntrinsics/GenXIntrinsics.h>
#endif
#include <llvm/Target/TargetIntrinsicInfo.h>

#ifdef ISPC_XE_ENABLED
#include <LLVMSPIRVLib/LLVMSPIRVLib.h>
#include <fstream>
#if defined(_WIN64)
#define OCLOC_LIBRARY_NAME "ocloc64.dll"
#elif defined(_WIN32)
#define OCLOC_LIBRARY_NAME "ocloc32.dll"
#elif defined(__linux__)
#define OCLOC_LIBRARY_NAME "libocloc.so"
#else
#error "Unexpected platform"
#endif
#endif

#ifdef ISPC_HOST_IS_WINDOWS
#include <io.h>
// windows.h defines CALLBACK as __stdcall
// clang/Analysis/CFG.h contains typename with name CALLBACK, which is got screwed up.
// So we include it after clang includes.
#include <windows.h>
// Note that this define must be after clang includes, as they undefining this symbol.
#define strcasecmp stricmp
#endif

using namespace ispc;

/*! list of files encountered by the parser. this allows emitting of
    the module file's dependencies via the -MMM option */
std::set<std::string> registeredDependencies;

/*! this is where the parser tells us that it has seen the given file
    name in the CPP hash */
void RegisterDependency(const std::string &fileName) {
    if (fileName[0] != '<' && fileName != "stdlib.ispc")
        registeredDependencies.insert(fileName);
}

static void lDeclareSizeAndPtrIntTypes(SymbolTable *symbolTable) {
    const Type *ptrIntType = (g->target->is32Bit()) ? AtomicType::VaryingInt32 : AtomicType::VaryingInt64;
    ptrIntType = ptrIntType->GetAsUnboundVariabilityType();

    symbolTable->AddType("intptr_t", ptrIntType, SourcePos());
    symbolTable->AddType("uintptr_t", ptrIntType->GetAsUnsignedType(), SourcePos());
    symbolTable->AddType("ptrdiff_t", ptrIntType, SourcePos());

    const Type *sizeType =
        (g->target->is32Bit() || g->opt.force32BitAddressing) ? AtomicType::VaryingUInt32 : AtomicType::VaryingUInt64;
    sizeType = sizeType->GetAsUnboundVariabilityType();
    symbolTable->AddType("size_t", sizeType, SourcePos());
}

/** After compilation completes, there's often a lot of extra debugging
    metadata left around that isn't needed any more--for example, for
    static functions that weren't actually used, function information for
    functions that were inlined, etc.  This function takes a llvm::Module
    and tries to strip out all of this extra stuff.
 */
static void lStripUnusedDebugInfo(llvm::Module *module) { return; }

///////////////////////////////////////////////////////////////////////////
// Module

Module::Module(const char *fn) : bufferCPP{nullptr} {
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

    module = new llvm::Module(!IsStdin(filename) ? filename : "<stdin>", *g->ctx);
    module->setTargetTriple(g->target->GetTripleString());

    diBuilder = NULL;
    diCompileUnit = NULL;

    // DataLayout information supposed to be managed in single place in Target class.
    module->setDataLayout(g->target->getDataLayout()->getStringRepresentation());

    // Version strings.
    // Have ISPC details and LLVM details as two separate strings attached to !llvm.ident.
    llvm::NamedMDNode *identMetadata = module->getOrInsertNamedMetadata("llvm.ident");
    std::string ispcVersion = std::string(ISPC_VERSION_STRING);
    std::string llvmVersion = clang::getClangToolFullVersion("LLVM");

    llvm::Metadata *identNode[] = {llvm::MDString::get(*g->ctx, ispcVersion)};
    identMetadata->addOperand(llvm::MDNode::get(*g->ctx, identNode));
    llvm::Metadata *identNode2[] = {llvm::MDString::get(*g->ctx, llvmVersion)};
    identMetadata->addOperand(llvm::MDNode::get(*g->ctx, identNode2));

    if (g->generateDebuggingSymbols) {
        llvm::TimeTraceScope TimeScope("Create Debug Data");
        // To enable debug information on Windows, we have to let llvm know, that
        // debug information should be emitted in CodeView format.
        if (g->target_os == TargetOS::windows) {
            module->addModuleFlag(llvm::Module::Warning, "CodeView", 1);
        } else {
            module->addModuleFlag(llvm::Module::Warning, "Dwarf Version", g->generateDWARFVersion);
        }
        diBuilder = new llvm::DIBuilder(*module);

        // Let the DIBuilder know that we're starting a new compilation
        // unit.
        if (IsStdin(filename)) {
            // Unfortunately we can't yet call Error() since the global 'm'
            // variable hasn't been initialized yet.
            Error(SourcePos(), "Can't emit debugging information with no "
                               "source file on disk.\n");
            ++errorCount;
            delete diBuilder;
            diBuilder = NULL;
        } else {
            std::string directory, name;
            GetDirectoryAndFileName(g->currentDirectory, filename, &directory, &name);
            auto srcFile = diBuilder->createFile(name, directory);
            // Use DW_LANG_C_plus_plus to avoid problems with debigging on Xe.
            // The debugger reads symbols partially when a solib file is loaded.
            // The kernel name is one of these read symbols. ISPC produces namespace
            // for example "ispc::simple_ispc". Matching the breakpoint location
            // "simple_ispc" with the symbol name fails if module language is C and not C++.
            diCompileUnit =
                diBuilder->createCompileUnit(llvm::dwarf::DW_LANG_C_plus_plus,          /* lang */
                                             srcFile,                                   /* filename */
                                             ispcVersion.c_str(),                       /* producer */
                                             g->opt.level > 0 /* is optimized */, "-g", /* command line args */
                                             0 /* run time version */);
        }
    }
}

extern FILE *yyin;
extern int yyparse();
typedef struct yy_buffer_state *YY_BUFFER_STATE;
extern void yy_switch_to_buffer(YY_BUFFER_STATE);
extern YY_BUFFER_STATE yy_scan_string(const char *);
extern YY_BUFFER_STATE yy_create_buffer(FILE *, int);
extern void yy_delete_buffer(YY_BUFFER_STATE);
extern void ParserInit();

int Module::CompileFile() {
    llvm::TimeTraceScope CompileFileTimeScope(
        "CompileFile", llvm::StringRef(filename + ("_" + std::string(g->target->GetISAString()))));
    ParserInit();

    // FIXME: it'd be nice to do this in the Module constructor, but this
    // function ends up calling into routines that expect the global
    // variable 'm' to be initialized and available (which it isn't until
    // the Module constructor returns...)
    {
        llvm::TimeTraceScope TimeScope("DefineStdlib");
        DefineStdlib(symbolTable, g->ctx, module, g->includeStdlib);
    }

    bool runPreprocessor = g->runCPP;

    if (runPreprocessor) {
        llvm::TimeTraceScope TimeScope("Frontend parser");
        if (!IsStdin(filename)) {
            // Try to open the file first, since otherwise we crash in the
            // preprocessor if the file doesn't exist.
            FILE *f = fopen(filename, "r");
            if (!f) {
                perror(filename);
                return 1;
            }
            fclose(f);
        }

        // If the CPP stream has been initialized, we have unexpected behavior.
        if (bufferCPP) {
            perror(filename);
            return 1;
        }

        // Replace the CPP stream with a newly allocated one.
        bufferCPP.reset(new CPPBuffer{});

        const int numErrors = execPreprocessor(!IsStdin(filename) ? filename : "-", bufferCPP->os.get());
        errorCount += (g->ignoreCPPErrors) ? 0 : numErrors;

        if (g->onlyCPP) {
            return errorCount; // Return early
        }

        YY_BUFFER_STATE strbuf = yy_scan_string(bufferCPP->str.c_str());
        yyparse();
        yy_delete_buffer(strbuf);
        clearCPPBuffer();
    } else {
        llvm::TimeTraceScope TimeScope("Frontend parser");
        // No preprocessor, just open up the file if it's not stdin..
        FILE *f = NULL;
        if (IsStdin(filename)) {
            f = stdin;
        } else {
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

    ast->Print(g->astDump);

    if (g->NoOmitFramePointer)
        for (llvm::Function &f : *module)
            f.addFnAttr("no-frame-pointer-elim", "true");
    for (llvm::Function &f : *module)
        g->target->markFuncWithTargetAttr(&f);
    ast->GenerateIR();

    if (diBuilder)
        diBuilder->finalize();
    llvm::TimeTraceScope TimeScope("Optimize");
    if (errorCount == 0)
        Optimize(module, g->opt.level);

    return errorCount;
}

Symbol *Module::AddLLVMIntrinsicDecl(const std::string &name, ExprList *args, SourcePos pos) {
    if (g->enableLLVMIntrinsics == false) {
        Error(SourcePos(), "Calling LLVM intrinsics from ISPC source code is an experimental feature,"
                           " which can be enabled by passing \"--enable-llvm-intrinsics\" switch to the compiler.\n");
        return nullptr;
    }

    llvm::Function *funcDecl = nullptr;
#ifdef ISPC_XE_ENABLED
    if (g->target->isXeTarget()) {
        llvm::GenXIntrinsic::ID ID = llvm::GenXIntrinsic::lookupGenXIntrinsicID(name);
        if (ID == llvm::GenXIntrinsic::not_any_intrinsic) {
            Error(pos, "LLVM intrinsic \"%s\" not supported.", name.c_str());
            return nullptr;
        }
        std::vector<llvm::Type *> exprType;
        int nInits = args->exprs.size();
        if (llvm::GenXIntrinsic::isOverloadedRet(ID) || llvm::GenXIntrinsic::isOverloadedArg(ID, nInits)) {
            for (int i = 0; i < nInits; ++i) {
                exprType.push_back((args->exprs[i])->GetType()->LLVMType(g->ctx));
            }
        }
        llvm::ArrayRef<llvm::Type *> argArr(exprType);
        funcDecl = llvm::GenXIntrinsic::getGenXDeclaration(module, ID, argArr);
    }
#endif
    if (!g->target->isXeTarget()) {
        llvm::TargetMachine *targetMachine = g->target->GetTargetMachine();
        const llvm::TargetIntrinsicInfo *TII = targetMachine->getIntrinsicInfo();
        llvm::Intrinsic::ID ID = llvm::Function::lookupIntrinsicID(llvm::StringRef(name));
        if (ID == llvm::Intrinsic::not_intrinsic && TII) {
            ID = static_cast<llvm::Intrinsic::ID>(TII->lookupName(llvm::StringRef(name)));
        }
        if (ID == llvm::Intrinsic::not_intrinsic) {
            Error(pos, "LLVM intrinsic \"%s\" not supported.", name.c_str());
            return nullptr;
        }
        std::vector<llvm::Type *> exprType;
        if (llvm::Intrinsic::isOverloaded(ID)) {
            int nInits = args->exprs.size();
            for (int i = 0; i < nInits; ++i) {
                exprType.push_back((args->exprs[i])->GetType()->LLVMType(g->ctx));
            }
        }
        llvm::ArrayRef<llvm::Type *> argArr(exprType);
        funcDecl = llvm::Intrinsic::getDeclaration(module, ID, argArr);
        llvm::StringRef funcName = funcDecl->getName();

        if (g->target->checkIntrinsticSupport(funcName, pos) == false) {
            return nullptr;
        }
    }

    Symbol *funcSym = CreateISPCSymbolForLLVMIntrinsic(funcDecl, symbolTable);
    return funcSym;
}

void Module::AddTypeDef(const std::string &name, const Type *type, SourcePos pos) {
    // Typedefs are easy; just add the mapping between the given name and
    // the given type.
    symbolTable->AddType(name.c_str(), type, pos);
}

void Module::AddGlobalVariable(const std::string &name, const Type *type, Expr *initExpr, bool isConst,
                               StorageClass storageClass, SourcePos pos) {
    // These may be NULL due to errors in parsing; just gracefully return
    // here if so.
    if (name == "" || type == NULL) {
        Assert(errorCount > 0);
        return;
    }

    if (symbolTable->LookupFunction(name.c_str())) {
        Error(pos,
              "Global variable \"%s\" shadows previously-declared "
              "function.",
              name.c_str());
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

    llvm::Type *llvmType = type->LLVMStorageType(g->ctx);
    if (llvmType == NULL)
        return;

    // See if we have an initializer expression for the global.  If so,
    // make sure it's a compile-time constant!
    llvm::Constant *llvmInitializer = NULL;
    ConstExpr *constValue = NULL;
    if (storageClass == SC_EXTERN) {
        if (initExpr != NULL)
            Error(pos,
                  "Initializer can't be provided with \"extern\" "
                  "global variable \"%s\".",
                  name.c_str());
    } else {
        if (initExpr != NULL) {
            initExpr = TypeCheck(initExpr);
            if (initExpr != NULL) {
                // We need to make sure the initializer expression is
                // the same type as the global.  (But not if it's an
                // ExprList; they don't have types per se / can't type
                // convert themselves anyway.)
                if (llvm::dyn_cast<ExprList>(initExpr) == NULL)
                    initExpr = TypeConvertExpr(initExpr, type, "initializer");

                if (initExpr != NULL) {
                    initExpr = Optimize(initExpr);
                    // Fingers crossed, now let's see if we've got a
                    // constant value..
                    std::pair<llvm::Constant *, bool> initPair = initExpr->GetStorageConstant(type);
                    llvmInitializer = initPair.first;

                    // If compiling for multitarget, skip initialization for
                    // indentified scenarios unless it's static
                    if (llvmInitializer != NULL) {
                        if ((storageClass != SC_STATIC) && (initPair.second == true)) {
                            if (g->isMultiTargetCompilation == true) {
                                Error(initExpr->pos,
                                      "Initializer for global variable \"%s\" "
                                      "is not a constant for multi-target compilation.",
                                      name.c_str());
                                return;
                            }
                            Warning(initExpr->pos,
                                    "Initializer for global variable \"%s\" "
                                    "is a constant for single-target compilation "
                                    "but not for multi-target compilation.",
                                    name.c_str());
                        }

                        if (type->IsConstType())
                            // Try to get a ConstExpr associated with
                            // the symbol.  This llvm::dyn_cast can
                            // validly fail, for example for types like
                            // StructTypes where a ConstExpr can't
                            // represent their values.
                            constValue = llvm::dyn_cast<ConstExpr>(initExpr);
                    } else
                        Error(initExpr->pos,
                              "Initializer for global variable \"%s\" "
                              "must be a constant.",
                              name.c_str());
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
            (sym->storageClass != SC_EXTERN && sym->storageClass != SC_EXTERN_C && sym->storageClass != storageClass)) {
            Error(pos,
                  "Definition of variable \"%s\" conflicts with "
                  "definition at %s:%d.",
                  name.c_str(), sym->pos.name, sym->pos.first_line);
            return;
        }

        llvm::GlobalVariable *gv = llvm::dyn_cast<llvm::GlobalVariable>(sym->storagePtr);
        Assert(gv != NULL);

        // And issue an error if this is a redefinition of a variable
        if (gv->hasInitializer() && sym->storageClass != SC_EXTERN && sym->storageClass != SC_EXTERN_C) {
            Error(pos,
                  "Redefinition of variable \"%s\" is illegal. "
                  "(Previous definition at %s:%d.)",
                  sym->name.c_str(), sym->pos.name, sym->pos.first_line);
            return;
        }

        // Now, we either have a redeclaration of a global, or a definition
        // of a previously-declared global.  First, save the pointer to the
        // previous llvm::GlobalVariable
        oldGV = gv;
    } else {
        sym = new Symbol(name, pos, type, storageClass);
        symbolTable->AddVariable(sym);
    }
    sym->constValue = constValue;

    llvm::GlobalValue::LinkageTypes linkage =
        (sym->storageClass == SC_STATIC) ? llvm::GlobalValue::InternalLinkage : llvm::GlobalValue::ExternalLinkage;

    // Note that the NULL llvmInitializer is what leads to "extern"
    // declarations coming up extern and not defining storage (a bit
    // subtle)...
    sym->storagePtr = new llvm::GlobalVariable(*module, llvmType, isConst, linkage, llvmInitializer, sym->name.c_str());

    // Patch up any references to the previous GlobalVariable (e.g. from a
    // declaration of a global that was later defined.)
    if (oldGV != NULL) {
        oldGV->replaceAllUsesWith(sym->storagePtr);
        oldGV->removeFromParent();
        sym->storagePtr->setName(sym->name.c_str());
    }

    if (diBuilder) {
        llvm::DIFile *file = pos.GetDIFile();
        llvm::DINamespace *diSpace = pos.GetDINamespace();
        // llvm::MDFile *file = pos.GetDIFile();
        llvm::GlobalVariable *sym_GV_storagePtr = llvm::dyn_cast<llvm::GlobalVariable>(sym->storagePtr);
        Assert(sym_GV_storagePtr);
        llvm::DIGlobalVariableExpression *var = diBuilder->createGlobalVariableExpression(
            diSpace, name, name, file, pos.first_line, sym->type->GetDIType(diSpace), (sym->storageClass == SC_STATIC));
        sym_GV_storagePtr->addDebugInfo(var);
        /*#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_6
                Assert(var.Verify());
        #else // LLVM 3.7+
              // comming soon
        #endif*/
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
static bool lRecursiveCheckValidParamType(const Type *t, bool vectorOk, bool soaOk, const std::string &name,
                                          SourcePos pos) {
    const StructType *st = CastType<StructType>(t);
    if (st != NULL) {
        for (int i = 0; i < st->GetElementCount(); ++i)
            if (!lRecursiveCheckValidParamType(st->GetElementType(i), soaOk, vectorOk, name, pos))
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
        return lRecursiveCheckValidParamType(seqt->GetElementType(), soaOk, vectorOk, name, pos);

    const PointerType *pt = CastType<PointerType>(t);
    if (pt != NULL) {
        // Only allow exported uniform pointers
        // Uniform pointers to varying data, however, are ok.
        if (pt->IsVaryingType())
            return false;
        else
            return lRecursiveCheckValidParamType(pt->GetBaseType(), true, true, name, pos);
    }

    if (t->IsSOAType() && soaOk) {
        Warning(pos, "Exported function parameter \"%s\" points to SOA type.", name.c_str());
        return false;
    }

    if (t->IsVaryingType() && !vectorOk)
        return false;
    else {
        if (t->IsVaryingType()) {
            Warning(pos, "Exported function parameter \"%s\" points to varying type", name.c_str());
        }
        return true;
    }
}

/** Given a Symbol representing a function parameter, see if it or any
    contained types are varying.  If so, issue an error.  (This function
    should only be called for parameters to 'export'ed functions, where
    varying parameters is illegal.
 */
static void lCheckExportedParameterTypes(const Type *type, const std::string &name, SourcePos pos) {
    if (lRecursiveCheckValidParamType(type, false, false, name, pos) == false) {
        if (const PointerType *pt = CastType<PointerType>(type)) {
            bool isSOAType = false;
            while (pt) {
                if (pt->GetBaseType()->IsSOAType()) {
                    isSOAType = true;
                    Error(pos,
                          "SOA type parameter \"%s\" is illegal "
                          "in an exported function.",
                          name.c_str());
                }
                pt = CastType<PointerType>(pt->GetBaseType());
            }
            if (!isSOAType) {
                Error(pos,
                      "Varying pointer type parameter \"%s\" is illegal "
                      "in an exported function.",
                      name.c_str());
            }
        }
        if (CastType<StructType>(type->GetBaseType()))
            Error(pos,
                  "Struct parameter \"%s\" with vector typed "
                  "member(s) is illegal in an exported function.",
                  name.c_str());
        else if (CastType<VectorType>(type))
            Error(pos,
                  "Vector-typed parameter \"%s\" is illegal in an exported "
                  "function.",
                  name.c_str());
        else
            Error(pos, "Varying parameter \"%s\" is illegal in an exported function.", name.c_str());
    }
}

#ifdef ISPC_XE_ENABLED
// For Xe target we have the same limitations in input parameters as for "export" functions
static void lCheckTaskParameterTypes(const Type *type, const std::string &name, SourcePos pos) {
    if (lRecursiveCheckValidParamType(type, false, false, name, pos) == false) {
        if (CastType<PointerType>(type))
            Error(pos,
                  "Varying pointer type parameter \"%s\" is illegal "
                  "in an \"task\" for Xe targets.",
                  name.c_str());
        if (CastType<StructType>(type->GetBaseType()))
            Error(pos,
                  "Struct parameter \"%s\" with vector typed "
                  "member(s) is illegal in an \"task\" for Xe targets.",
                  name.c_str());
        else if (CastType<VectorType>(type))
            Error(pos,
                  "Vector-typed parameter \"%s\" is illegal in an \"task\" "
                  "for Xe targets.",
                  name.c_str());
        else
            Error(pos, "Varying parameter \"%s\" is illegal in an \"task\" for Xe targets.", name.c_str());
    }
}
#endif

/** Given a function type, loop through the function parameters and see if
    any are StructTypes.  If so, issue an error; this is currently broken
    (https://github.com/ispc/ispc/issues/3).
 */
static void lCheckForStructParameters(const FunctionType *ftype, SourcePos pos) {
    for (int i = 0; i < ftype->GetNumParameters(); ++i) {
        const Type *type = ftype->GetParameterType(i);
        if (CastType<StructType>(type) != NULL) {
            Error(pos, "Passing structs to/from application functions by value "
                       "is currently not supported. Use a reference, a const reference, "
                       "a pointer, or a const pointer to the struct instead.");
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
void Module::AddFunctionDeclaration(const std::string &name, const FunctionType *functionType,
                                    StorageClass storageClass, bool isInline, bool isNoInline, bool isVectorCall,
                                    SourcePos pos) {
    Assert(functionType != NULL);

    // If a global variable with the same name has already been declared
    // issue an error.
    if (symbolTable->LookupVariable(name.c_str()) != NULL) {
        Error(pos,
              "Function \"%s\" shadows previously-declared global variable. "
              "Ignoring this definition.",
              name.c_str());
        return;
    }

    std::vector<Symbol *> overloadFuncs;
    symbolTable->LookupFunction(name.c_str(), &overloadFuncs);
    if (overloadFuncs.size() > 0) {
        for (unsigned int i = 0; i < overloadFuncs.size(); ++i) {
            Symbol *overloadFunc = overloadFuncs[i];

            const FunctionType *overloadType = CastType<FunctionType>(overloadFunc->type);
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
                Error(pos,
                      "Illegal to provide \"export\" qualifier for "
                      "functions with the same name but different types. "
                      "(Previous function declaration (%s:%d).)",
                      overloadFunc->pos.name, overloadFunc->pos.first_line);

            // If all of the parameter types match but the return type is
            // different, return an error--overloading by return type isn't
            // allowed.
            const FunctionType *ofType = CastType<FunctionType>(overloadFunc->type);
            Assert(ofType != NULL);
            if (ofType->GetNumParameters() == functionType->GetNumParameters()) {
                int i;
                for (i = 0; i < functionType->GetNumParameters(); ++i) {
                    if (Type::Equal(ofType->GetParameterType(i), functionType->GetParameterType(i)) == false)
                        break;
                }
                if (i == functionType->GetNumParameters()) {
                    std::string thisRetType = functionType->GetReturnTypeString();
                    std::string otherRetType = ofType->GetReturnTypeString();
                    Error(pos,
                          "Illegal to overload function by return "
                          "type only.  This function returns \"%s\" while "
                          "previous declaration at %s:%d returns \"%s\".",
                          thisRetType.c_str(), overloadFunc->pos.name, overloadFunc->pos.first_line,
                          otherRetType.c_str());
                    return;
                }
            }
        }
    }

    if (storageClass == SC_EXTERN_C) {
        // Make sure the user hasn't supplied both an 'extern "C"' and a
        // 'task' qualifier with the function
        if (functionType->isTask) {
            Error(pos,
                  "\"task\" qualifier is illegal with C-linkage extern "
                  "function \"%s\".  Ignoring this function.",
                  name.c_str());
            return;
        }

        std::vector<Symbol *> funcs;
        symbolTable->LookupFunction(name.c_str(), &funcs);
        if (funcs.size() > 0) {
            if (funcs.size() > 1) {
                // Multiple functions with this name have already been declared;
                // can't overload here
                Error(pos,
                      "Can't overload extern \"C\" function \"%s\"; "
                      "%d functions with the same name have already been declared.",
                      name.c_str(), (int)funcs.size());
                return;
            }

            // One function with the same name has been declared; see if it
            // has the same type as this one, in which case it's ok.
            if (Type::Equal(funcs[0]->type, functionType))
                return;
            else {
                Error(pos, "Can't overload extern \"C\" function \"%s\".", name.c_str());
                return;
            }
        }
    }

    // Get the LLVM FunctionType
    bool disableMask = (storageClass == SC_EXTERN_C);
    llvm::FunctionType *llvmFunctionType = functionType->LLVMFunctionType(g->ctx, disableMask);
    if (llvmFunctionType == NULL)
        return;

    // And create the llvm::Function
    llvm::GlobalValue::LinkageTypes linkage = (storageClass == SC_STATIC || isInline)
                                                  ? llvm::GlobalValue::InternalLinkage
                                                  : llvm::GlobalValue::ExternalLinkage;

    // For Xe target all functions except Xe kernels, external functions and explicitly marked extern functions must
    // be internal.
    if (g->target->isXeTarget() && !functionType->IsISPCKernel() && !functionType->IsISPCExternal() &&
        (storageClass != SC_EXTERN))
        linkage = llvm::GlobalValue::InternalLinkage;

    std::string functionName = name;
    if (storageClass != SC_EXTERN_C) {
        functionName += functionType->Mangle();
        // If we treat generic as smth, we should have appropriate mangling
        if (g->mangleFunctionsWithTarget) {
            functionName += g->target->GetISAString();
        }
    }
    llvm::Function *function = llvm::Function::Create(llvmFunctionType, linkage, functionName.c_str(), module);

    if (g->target_os == TargetOS::windows) {
        // Make export functions callable from DLLs.
        if ((g->dllExport) && (storageClass != SC_STATIC)) {
            function->setDLLStorageClass(llvm::GlobalValue::DLLExportStorageClass);
        }
    }

    if (isNoInline && isInline) {
        Error(pos, "Illegal to use \"noinline\" and \"inline\" qualifiers together on function \"%s\".", name.c_str());
        return;
    }
    // Set function attributes: we never throw exceptions
    function->setDoesNotThrow();
    if (storageClass != SC_EXTERN_C && isInline) {
        function->addFnAttr(llvm::Attribute::AlwaysInline);
    }

    if (isVectorCall) {
        if ((storageClass != SC_EXTERN_C)) {
            Error(pos, "Illegal to use \"__vectorcall\" qualifier on non-extern function \"%s\".", name.c_str());
            return;
        }
        if (g->target_os != TargetOS::windows) {
            Error(pos, "Illegal to use \"__vectorcall\" qualifier on function \"%s\" for non-Windows OS.",
                  name.c_str());
            return;
        }
    }

    if (isNoInline) {
        function->addFnAttr(llvm::Attribute::NoInline);
    }

    if (functionType->isTask) {
        if (!g->target->isXeTarget()) {
            // This also applies transitively to members I think?
            function->addParamAttr(0, llvm::Attribute::NoAlias);
        }
    }
    if (((isVectorCall) && (storageClass == SC_EXTERN_C)) || (storageClass != SC_EXTERN_C)) {
        g->target->markFuncWithCallingConv(function);
    }

    g->target->markFuncWithTargetAttr(function);

    // Make sure that the return type isn't 'varying' or vector typed if
    // the function is 'export'ed.
    if (functionType->isExported &&
        lRecursiveCheckValidParamType(functionType->GetReturnType(), false, false, name, pos) == false)
        Error(pos,
              "Illegal to return a \"varying\" or vector type from "
              "exported function \"%s\"",
              name.c_str());

    if (functionType->isTask && functionType->GetReturnType()->IsVoidType() == false)
        Error(pos, "Task-qualified functions must have void return type.");

    // This limitation is due to ABI incompatibility between ISPC/ESIMD.
    // ESIMD makes return value optimization transferring return value to the
    // argument of the function.
    if (functionType->IsISPCExternal() && functionType->GetReturnType()->IsVoidType() == false)
        Warning(pos, "Export and extern \"C\"-qualified functions must have void return type for Xe target.");

    if (functionType->isExported || functionType->isExternC || functionType->IsISPCExternal() ||
        functionType->IsISPCKernel()) {
        lCheckForStructParameters(functionType, pos);
    }

    // Mark ISPC external functions as SPIR_FUNC for Xe.
    if (functionType->IsISPCExternal() && disableMask) {
        function->setCallingConv(llvm::CallingConv::SPIR_FUNC);
        function->setDSOLocal(true);
    }
    // Mark with corresponding attribute
    if (g->target->isXeTarget()) {
        if (functionType->IsISPCKernel()) {
            function->addFnAttr("CMGenxMain");
        } else {
            function->addFnAttr("CMStackCall");
        }
    }

    // Loop over all of the arguments; process default values if present
    // and do other checks and parameter attribute setting.
    bool seenDefaultArg = false;
    int nArgs = functionType->GetNumParameters();
    for (int i = 0; i < nArgs; ++i) {
        const Type *argType = functionType->GetParameterType(i);
        const std::string &argName = functionType->GetParameterName(i);
        Expr *defaultValue = functionType->GetParameterDefault(i);
        const SourcePos &argPos = functionType->GetParameterSourcePos(i);

        // If the function is exported or in case of Xe target is task, make sure that the parameter
        // doesn't have any funky stuff going on in it.
        // JCB nomosoa - Varying is now a-ok.
        if (functionType->isExported)
            lCheckExportedParameterTypes(argType, argName, argPos);
#ifdef ISPC_XE_ENABLED
        if (functionType->IsISPCKernel())
            lCheckTaskParameterTypes(argType, argName, argPos);
#endif

        // ISPC assumes that no pointers alias.  (It should be possible to
        // specify when this is not the case, but this should be the
        // default.)  Set parameter attributes accordingly.  (Only for
        // uniform pointers, since varying pointers are int vectors...)
        if (!functionType->isTask && ((CastType<PointerType>(argType) != NULL && argType->IsUniformType() &&
                                       // Exclude SOA argument because it is a pair {struct *, int}
                                       // instead of pointer
                                       !CastType<PointerType>(argType)->IsSlice()) ||

                                      CastType<ReferenceType>(argType) != NULL)) {

            function->addParamAttr(i, llvm::Attribute::NoAlias);
#if 0
            int align = 4 * RoundUpPow2(g->target->nativeVectorWidth);
            function->addAttribute(i+1, llvm::Attribute::constructAlignmentFromInt(align));
#endif
        }

        if (symbolTable->LookupFunction(argName.c_str()))
            Warning(argPos,
                    "Function parameter \"%s\" shadows a function "
                    "declared in global scope.",
                    argName.c_str());

        if (defaultValue != NULL)
            seenDefaultArg = true;
        else if (seenDefaultArg) {
            // Once one parameter has provided a default value, then all of
            // the following ones must have them as well.
            Error(argPos,
                  "Parameter \"%s\" is missing default: all "
                  "parameters after the first parameter with a default value "
                  "must have default values as well.",
                  argName.c_str());
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

void Module::AddFunctionDefinition(const std::string &name, const FunctionType *type, Stmt *code) {
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

void Module::AddExportedTypes(const std::vector<std::pair<const Type *, SourcePos>> &types) {
    for (int i = 0; i < (int)types.size(); ++i) {
        if (CastType<StructType>(types[i].first) == NULL && CastType<VectorType>(types[i].first) == NULL &&
            CastType<EnumType>(types[i].first) == NULL)
            Error(types[i].second,
                  "Only struct, vector, and enum types, "
                  "not \"%s\", are allowed in type export lists.",
                  types[i].first->GetString().c_str());
        else
            exportedTypes.push_back(types[i]);
    }
}

bool Module::writeOutput(OutputType outputType, OutputFlags flags, const char *outFileName,
                         const char *depTargetFileName, const char *sourceFileName, DispatchHeaderInfo *DHI) {
    if (diBuilder && (outputType != Header) && (outputType != Deps))
        lStripUnusedDebugInfo(module);

    Assert(module);

    // In LLVM_3_4 after r195494 and r195504 revisions we should pass
    // "Debug Info Version" constant to the module. LLVM will ignore
    // our Debug Info metadata without it.
    if (g->generateDebuggingSymbols == true) {
        module->addModuleFlag(llvm::Module::Warning, "Debug Info Version", llvm::DEBUG_METADATA_VERSION);
    }

    // SIC! (verifyModule() == TRUE) means "failed", see llvm-link code.
    if ((outputType != Header) && (outputType != Deps) && (outputType != HostStub) && (outputType != DevStub) &&
        (outputType != CPPStub) && llvm::verifyModule(*module)) {
        FATAL("Resulting module verification failed!");
    }

    if (outFileName) {
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
            case BitcodeText:
                if (strcasecmp(suffix, "ll"))
                    fileType = "LLVM assembly";
                break;
            case Object:
                if (strcasecmp(suffix, "o") && strcasecmp(suffix, "obj"))
                    fileType = "object";
                break;
#ifdef ISPC_XE_ENABLED
            case ZEBIN:
                if (strcasecmp(suffix, "bin"))
                    fileType = "L0 binary";
                break;
            case SPIRV:
                if (strcasecmp(suffix, "spv"))
                    fileType = "spir-v";
                break;
#endif
            case Header:
                if (strcasecmp(suffix, "h") && strcasecmp(suffix, "hh") && strcasecmp(suffix, "hpp"))
                    fileType = "header";
                break;
            case Deps:
                break;
            case DevStub:
                if (strcasecmp(suffix, "c") && strcasecmp(suffix, "cc") && strcasecmp(suffix, "c++") &&
                    strcasecmp(suffix, "cxx") && strcasecmp(suffix, "cpp"))
                    fileType = "dev-side offload stub";
                break;
            case HostStub:
                if (strcasecmp(suffix, "c") && strcasecmp(suffix, "cc") && strcasecmp(suffix, "c++") &&
                    strcasecmp(suffix, "cxx") && strcasecmp(suffix, "cpp"))
                    fileType = "host-side offload stub";
                break;
            case CPPStub:
                if (strcasecmp(suffix, "ispi") && strcasecmp(suffix, "i"))
                    fileType = "preprocessed stub";
                break;
            default:
                Assert(0 /* swtich case not handled */);
                return 1;
            }
            if (fileType != NULL)
                Warning(SourcePos(),
                        "Emitting %s file, but filename \"%s\" "
                        "has suffix \"%s\"?",
                        fileType, outFileName, suffix);
        }
    }
    if (outputType == Header) {
        if (DHI)
            return writeDispatchHeader(DHI);
        else
            return writeHeader(outFileName);
    } else if (outputType == Deps)
        return writeDeps(outFileName, 0 != (flags & GenerateMakeRuleForDeps), depTargetFileName, sourceFileName);
    else if (outputType == CPPStub)
        return writeCPPStub(outFileName);
    else if (outputType == HostStub)
        return writeHostStub(outFileName);
    else if (outputType == DevStub)
        return writeDevStub(outFileName);
    else if ((outputType == Bitcode) || (outputType == BitcodeText))
        return writeBitcode(module, outFileName, outputType);
#ifdef ISPC_XE_ENABLED
    else if (outputType == SPIRV)
        return writeSPIRV(module, outFileName);
    else if (outputType == ZEBIN)
        return writeZEBin(module, outFileName);
#endif
    else
        return writeObjectFileOrAssembly(outputType, outFileName);
}

bool Module::writeBitcode(llvm::Module *module, const char *outFileName, OutputType outputType) {
    // Get a file descriptor corresponding to where we want the output to
    // go.  If we open it, it'll be closed by the llvm::raw_fd_ostream
    // destructor.
    int fd;
    if (!strcmp(outFileName, "-"))
        fd = 1; // stdout
    else {
        int flags = O_CREAT | O_WRONLY | O_TRUNC;
#ifdef ISPC_HOST_IS_WINDOWS
        flags |= O_BINARY;
        fd = _open(outFileName, flags, 0644);
#else
        fd = open(outFileName, flags, 0644);
#endif // ISPC_HOST_IS_WINDOWS
        if (fd == -1) {
            perror(outFileName);
            return false;
        }
    }

    llvm::raw_fd_ostream fos(fd, (fd != 1), false);
    if (outputType == Bitcode)
        llvm::WriteBitcodeToFile(*module, fos);
    else if (outputType == BitcodeText)
        module->print(fos, nullptr);
    return true;
}

#ifdef ISPC_XE_ENABLED
bool Module::translateToSPIRV(llvm::Module *module, std::stringstream &ss) {
    std::string err;
    SPIRV::TranslatorOpts Opts;
    Opts.enableAllExtensions();
    llvm::cl::opt<bool> SPIRVAllowExtraDIExpressions(
        "spirv-allow-extra-diexpressions", llvm::cl::init(true),
        llvm::cl::desc("Allow DWARF operations not listed in the OpenCL.DebugInfo.100 "
                       "specification (experimental, may produce incompatible SPIR-V "
                       "module)"));
#if ISPC_LLVM_VERSION == ISPC_LLVM_12_0
    llvm::cl::opt<bool> SPIRVAllowUnknownIntrinsics(
        "spirv-allow-unknown-intrinsics", llvm::cl::init(true),
        llvm::cl::desc("Unknown LLVM intrinsics will be translated as external function "
                       "calls in SPIR-V"));

    Opts.setSPIRVAllowUnknownIntrinsicsEnabled(SPIRVAllowUnknownIntrinsics);
#else
    Opts.setSPIRVAllowUnknownIntrinsics({"llvm.genx"});
#endif
    Opts.setAllowExtraDIExpressionsEnabled(SPIRVAllowExtraDIExpressions);
    Opts.setDesiredBIsRepresentation(SPIRV::BIsRepresentation::SPIRVFriendlyIR);
    Opts.setDebugInfoEIS(SPIRV::DebugInfoEIS::OpenCL_DebugInfo_100);
    bool success = llvm::writeSpirv(module, Opts, ss, err);
    if (!success) {
        fprintf(stderr, "Fails to save LLVM as SPIR-V: %s \n", err.c_str());
        return false;
    }
    return true;
}

bool Module::writeSPIRV(llvm::Module *module, const char *outFileName) {
    std::stringstream translatedStream;
    bool success = translateToSPIRV(module, translatedStream);
    if (!success) {
        return false;
    }
    if (!strcmp(outFileName, "-")) {
        std::cout << translatedStream.rdbuf();
    } else {
        std::ofstream fos(outFileName, std::ios::binary);
        fos << translatedStream.rdbuf();
    }
    return true;
}

// Translate ISPC CPU name to Neo representation
static std::string translateCPU(const std::string &CPU) {
    auto It = ISPCToNeoCPU.find(CPU);
    Assert(It != ISPCToNeoCPU.end() && "Unexpected CPU");
    return It->second;
}

// Copy outputs. Required file should have provided extension (it is
// different for different binary kinds).
static void saveOutput(uint32_t numOutputs, uint8_t **dataOutputs, uint64_t *lenOutputs, char **nameOutputs,
                       std::vector<char> &oclocRes) {
    const std::string &requiredExtension = ".bin";
    llvm::ArrayRef<const uint8_t *> outBins{dataOutputs, numOutputs};
    llvm::ArrayRef<uint64_t> outLens{lenOutputs, numOutputs};
    llvm::ArrayRef<const char *> outNames{nameOutputs, numOutputs};
    auto zip = llvm::zip(outBins, outLens, outNames);
    using ZipTy = typename decltype(zip)::value_type;
    auto binIt = std::find_if(zip.begin(), zip.end(), [&requiredExtension](ZipTy File) {
        llvm::StringRef name{std::get<2>(File)};
        return name.endswith(requiredExtension);
    });
    Assert(binIt != zip.end() && "Output binary is missing");

    llvm::ArrayRef<uint8_t> binRef{std::get<0>(*binIt), static_cast<std::size_t>(std::get<1>(*binIt))};
    oclocRes.assign(binRef.begin(), binRef.end());
}

bool Module::writeZEBin(llvm::Module *module, const char *outFileName) {
    std::stringstream translatedStream;
    bool success = translateToSPIRV(module, translatedStream);
    if (!success) {
        return false;
    }
    const std::string &translatedStr = translatedStream.str();
    std::vector<char> spirStr(translatedStr.begin(), translatedStr.end());

    const std::string CPUName = g->target->getCPU();
    const std::string neoCPU = translateCPU(CPUName);

    invokePtr invoke;
    freeOutputPtr freeOutput;
    auto oclocLib = llvm::sys::DynamicLibrary::getPermanentLibrary(OCLOC_LIBRARY_NAME);
    if (!oclocLib.isValid()) {
        Error(SourcePos(), "Cannot open '" OCLOC_LIBRARY_NAME "'\n");
        return false;
    }
    invoke = reinterpret_cast<invokePtr>(oclocLib.getAddressOfSymbol("oclocInvoke"));
    if (!invoke) {
        Error(SourcePos(), "oclocInvoke symbol is missing \n");
        return false;
    }
    freeOutput = reinterpret_cast<freeOutputPtr>(oclocLib.getAddressOfSymbol("oclocFreeOutput"));
    if (!freeOutput) {
        Error(SourcePos(), "oclocFreeOutput symbol is missing \n");
        return false;
    }

    std::string options{"-vc-codegen -no-optimize -Xfinalizer '-presched'"};
    // Add debug option to VC backend of "-g" is set to ISPC
    if (g->generateDebuggingSymbols) {
        options.append(" -g");
    }

    if (g->vcOpts != "") {
        options.append(" " + g->vcOpts);
    }

    // Add stack size info
    // If stackMemSize has default value 0, do not set -stateless-stack-mem-size,
    // it will be set to 8192 in VC backend by default.
    if (g->stackMemSize > 0) {
        options.append(" -stateless-stack-mem-size=" + std::to_string(g->stackMemSize));
    }
    std::string internalOptions;

    // Use L0 binary
    internalOptions.append(" -binary-format=").append("ze");

    const char *spvFileName = "ispc_spirv";

    std::vector<const char *> oclocArgs;
    oclocArgs.push_back("ocloc");
    oclocArgs.push_back("compile");
    oclocArgs.push_back("-device");
    oclocArgs.push_back(neoCPU.c_str());
    oclocArgs.push_back("-spirv_input");
    oclocArgs.push_back("-file");
    oclocArgs.push_back(spvFileName);
    oclocArgs.push_back("-options");
    oclocArgs.push_back(options.c_str());
    oclocArgs.push_back("-internal_options");
    oclocArgs.push_back(internalOptions.c_str());

    uint32_t numOutputs = 0;
    uint8_t **dataOutputs = nullptr;
    uint64_t *lenOutputs = nullptr;
    char **nameOutputs = nullptr;

    static_assert(alignof(uint8_t) == alignof(char), "Possible unaligned access");
    auto *spvSource = reinterpret_cast<const uint8_t *>(spirStr.data());
    const uint64_t spvLen = spirStr.size();
    if (invoke(oclocArgs.size(), oclocArgs.data(), 1, &spvSource, &spvLen, &spvFileName, 0, nullptr, nullptr, nullptr,
               &numOutputs, &dataOutputs, &lenOutputs, &nameOutputs)) {
        Error(SourcePos(), "Call to oclocInvoke failed \n");
        return false;
    }

    std::vector<char> oclocRes;
    saveOutput(numOutputs, dataOutputs, lenOutputs, nameOutputs, oclocRes);
    if (freeOutput(&numOutputs, &dataOutputs, &lenOutputs, &nameOutputs)) {
        Error(SourcePos(), "Call to oclocFreeOutput failed \n");
        return false;
    }

    if (!strcmp(outFileName, "-")) {
        std::cout.write(oclocRes.data(), oclocRes.size());
    } else {
        std::ofstream fos(outFileName, std::ios::binary);
        fos.write(oclocRes.data(), oclocRes.size());
    }
    return true;
}
#endif // ISPC_XE_ENABLED

bool Module::writeObjectFileOrAssembly(OutputType outputType, const char *outFileName) {
    llvm::TargetMachine *targetMachine = g->target->GetTargetMachine();
    return writeObjectFileOrAssembly(targetMachine, module, outputType, outFileName);
}

bool Module::writeObjectFileOrAssembly(llvm::TargetMachine *targetMachine, llvm::Module *module, OutputType outputType,
                                       const char *outFileName) {
    // Figure out if we're generating object file or assembly output, and
    // set binary output for object files
    llvm::CodeGenFileType fileType = (outputType == Object) ? llvm::CGFT_ObjectFile : llvm::CGFT_AssemblyFile;
    bool binary = (fileType == llvm::CGFT_ObjectFile);

    llvm::sys::fs::OpenFlags flags = binary ? llvm::sys::fs::OF_None : llvm::sys::fs::OF_Text;

    std::error_code error;

    std::unique_ptr<llvm::ToolOutputFile> of(new llvm::ToolOutputFile(outFileName, error, flags));

    if (error) {
        Error(SourcePos(), "Cannot open output file \"%s\".\n", outFileName);
        return false;
    }

    llvm::legacy::PassManager pm;

    {
        llvm::raw_fd_ostream &fos(of->os());
        // Third parameter is for generation of .dwo file, which is separate DWARF
        // file for ELF targets. We don't support it currently.
        if (targetMachine->addPassesToEmitFile(pm, fos, nullptr, fileType)) {
            FATAL("Failed to add passes to emit object file!");
        }

        // Finally, run the passes to emit the object file/assembly
        pm.run(*module);

        // Success; tell tool_output_file to keep the final output file.
        of->keep();
    }
    return true;
}

/** Given a pointer to an element of a structure, see if it is a struct
    type or an array of a struct type.  If so, return a pointer to the
    underlying struct type. */
static const StructType *lGetElementStructType(const Type *t) {
    const StructType *st = CastType<StructType>(t);
    if (st != NULL)
        return st;

    const ArrayType *at = CastType<ArrayType>(t);
    if (at != NULL)
        return lGetElementStructType(at->GetElementType());

    return NULL;
}

static bool lContainsPtrToVarying(const StructType *st) {
    int numElts = st->GetElementCount();

    for (int j = 0; j < numElts; ++j) {
        const Type *t = st->GetElementType(j);

        if (t->IsVaryingType())
            return true;
    }

    return false;
}

/** Emits a declaration for the given struct to the given file.  This
    function first makes sure that declarations for any structs that are
    (recursively) members of this struct are emitted first.
 */
static void lEmitStructDecl(const StructType *st, std::vector<const StructType *> *emittedStructs, FILE *file,
                            bool emitUnifs = true) {

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
        const StructType *elementStructType = lGetElementStructType(st->GetElementType(i));
        if (elementStructType != NULL)
            lEmitStructDecl(elementStructType, emittedStructs, file, emitUnifs);
    }

    // And now it's safe to declare this one
    emittedStructs->push_back(st);

    fprintf(file, "#ifndef __ISPC_STRUCT_%s__\n", st->GetCStructName().c_str());
    fprintf(file, "#define __ISPC_STRUCT_%s__\n", st->GetCStructName().c_str());

    char sSOA[48];
    bool pack, needsAlign = false;
    llvm::Type *stype = st->LLVMType(g->ctx);
    const llvm::DataLayout *DL = g->target->getDataLayout();

    if (!(pack = llvm::dyn_cast<llvm::StructType>(stype)->isPacked())) {
        for (int i = 0; !needsAlign && (i < st->GetElementCount()); ++i) {
            const Type *ftype = st->GetElementType(i)->GetAsNonConstType();
            needsAlign |= ftype->IsVaryingType() && (CastType<StructType>(ftype) == NULL);
        }
    }
    if (st->GetSOAWidth() > 0) {
        // This has to match the naming scheme in
        // StructType::GetCDeclaration().
        snprintf(sSOA, sizeof(sSOA), "_SOA%d", st->GetSOAWidth());
    } else {
        *sSOA = '\0';
    }
    if (!needsAlign) {
        fprintf(file, "%sstruct %s%s {\n", (pack) ? "packed " : "", st->GetCStructName().c_str(), sSOA);
    } else {
        unsigned uABI = DL->getABITypeAlignment(stype);
        fprintf(file, "__ISPC_ALIGNED_STRUCT__(%u) %s%s {\n", uABI, st->GetCStructName().c_str(), sSOA);
    }
    for (int i = 0; i < st->GetElementCount(); ++i) {
        const Type *ftype = st->GetElementType(i)->GetAsNonConstType();
        std::string d = ftype->GetCDeclaration(st->GetElementName(i));

        fprintf(file, "    ");
        if (needsAlign && ftype->IsVaryingType() && (CastType<StructType>(ftype) == NULL)) {
            unsigned uABI = DL->getABITypeAlignment(ftype->LLVMStorageType(g->ctx));
            fprintf(file, "__ISPC_ALIGN__(%u) ", uABI);
        }
        // Don't expand arrays, pointers and structures:
        // their insides will be expanded automatically.
        if (!ftype->IsArrayType() && !ftype->IsPointerType() && ftype->IsVaryingType() &&
            (CastType<StructType>(ftype) == NULL)) {
            fprintf(file, "%s[%d];\n", d.c_str(), g->target->getVectorWidth());
        } else {
            fprintf(file, "%s;\n", d.c_str());
        }
    }
    fprintf(file, "};\n");
    fprintf(file, "#endif\n\n");
}

/** Given a set of structures that we want to print C declarations of in a
    header file, emit their declarations.
 */
static void lEmitStructDecls(std::vector<const StructType *> &structTypes, FILE *file, bool emitUnifs = true) {
    std::vector<const StructType *> emittedStructs;

    fprintf(file, "\n#ifndef __ISPC_ALIGN__\n"
                  "#if defined(__clang__) || !defined(_MSC_VER)\n"
                  "// Clang, GCC, ICC\n"
                  "#define __ISPC_ALIGN__(s) __attribute__((aligned(s)))\n"
                  "#define __ISPC_ALIGNED_STRUCT__(s) struct __ISPC_ALIGN__(s)\n"
                  "#else\n"
                  "// Visual Studio\n"
                  "#define __ISPC_ALIGN__(s) __declspec(align(s))\n"
                  "#define __ISPC_ALIGNED_STRUCT__(s) __ISPC_ALIGN__(s) struct\n"
                  "#endif\n"
                  "#endif\n\n");

    for (unsigned int i = 0; i < structTypes.size(); ++i)
        lEmitStructDecl(structTypes[i], &emittedStructs, file, emitUnifs);
}

/** Emit C declarations of enumerator types to the generated header file.
 */
static void lEmitEnumDecls(const std::vector<const EnumType *> &enumTypes, FILE *file) {
    if (enumTypes.size() == 0)
        return;

    fprintf(file, "///////////////////////////////////////////////////////////////////////////\n");
    fprintf(file, "// Enumerator types with external visibility from ispc code\n");
    fprintf(file, "///////////////////////////////////////////////////////////////////////////\n\n");

    for (unsigned int i = 0; i < enumTypes.size(); ++i) {
        fprintf(file, "#ifndef __ISPC_ENUM_%s__\n", enumTypes[i]->GetEnumName().c_str());
        fprintf(file, "#define __ISPC_ENUM_%s__\n", enumTypes[i]->GetEnumName().c_str());
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
static void lEmitVectorTypedefs(const std::vector<const VectorType *> &types, FILE *file) {
    if (types.size() == 0)
        return;

    fprintf(file, "///////////////////////////////////////////////////////////////////////////\n");
    fprintf(file, "// Vector types with external visibility from ispc code\n");
    fprintf(file, "///////////////////////////////////////////////////////////////////////////\n\n");

    for (unsigned int i = 0; i < types.size(); ++i) {
        std::string baseDecl;
        const VectorType *vt = types[i]->GetAsNonConstType();
        if (!vt->IsUniformType())
            // Varying stuff shouldn't be visibile to / used by the
            // application, so at least make it not simple to access it by
            // not declaring the type here...
            continue;

        int size = vt->GetElementCount();

        llvm::Type *ty = vt->LLVMStorageType(g->ctx);
        int align = g->target->getDataLayout()->getABITypeAlignment(ty);
        baseDecl = vt->GetBaseType()->GetCDeclaration("");
        fprintf(file, "#ifndef __ISPC_VECTOR_%s%d__\n", baseDecl.c_str(), size);
        fprintf(file, "#define __ISPC_VECTOR_%s%d__\n", baseDecl.c_str(), size);
        fprintf(file, "#ifdef _MSC_VER\n__declspec( align(%d) ) ", align);
        fprintf(file, "struct %s%d { %s v[%d]; };\n", baseDecl.c_str(), size, baseDecl.c_str(), size);
        fprintf(file, "#else\n");
        fprintf(file, "struct %s%d { %s v[%d]; } __attribute__ ((aligned(%d)));\n", baseDecl.c_str(), size,
                baseDecl.c_str(), size, align);
        fprintf(file, "#endif\n");
        fprintf(file, "#endif\n\n");
    }
    fprintf(file, "\n");
}

/** Add the given type to the vector, if that type isn't already in there.
 */
template <typename T> static void lAddTypeIfNew(const Type *type, std::vector<const T *> *exportedTypes) {
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
static void lGetExportedTypes(const Type *type, std::vector<const StructType *> *exportedStructTypes,
                              std::vector<const EnumType *> *exportedEnumTypes,
                              std::vector<const VectorType *> *exportedVectorTypes) {
    const ArrayType *arrayType = CastType<ArrayType>(type);
    const StructType *structType = CastType<StructType>(type);
    const FunctionType *ftype = CastType<FunctionType>(type);

    if (CastType<ReferenceType>(type) != NULL)
        lGetExportedTypes(type->GetReferenceTarget(), exportedStructTypes, exportedEnumTypes, exportedVectorTypes);
    else if (CastType<PointerType>(type) != NULL)
        lGetExportedTypes(type->GetBaseType(), exportedStructTypes, exportedEnumTypes, exportedVectorTypes);
    else if (arrayType != NULL)
        lGetExportedTypes(arrayType->GetElementType(), exportedStructTypes, exportedEnumTypes, exportedVectorTypes);
    else if (structType != NULL) {
        lAddTypeIfNew(type, exportedStructTypes);
        for (int i = 0; i < structType->GetElementCount(); ++i)
            lGetExportedTypes(structType->GetElementType(i), exportedStructTypes, exportedEnumTypes,
                              exportedVectorTypes);
    } else if (CastType<UndefinedStructType>(type) != NULL)
        // do nothing
        ;
    else if (CastType<EnumType>(type) != NULL)
        lAddTypeIfNew(type, exportedEnumTypes);
    else if (CastType<VectorType>(type) != NULL)
        lAddTypeIfNew(type, exportedVectorTypes);
    else if (ftype != NULL) {
        // Handle Return Types
        lGetExportedTypes(ftype->GetReturnType(), exportedStructTypes, exportedEnumTypes, exportedVectorTypes);

        // And now the parameter types...
        for (int j = 0; j < ftype->GetNumParameters(); ++j)
            lGetExportedTypes(ftype->GetParameterType(j), exportedStructTypes, exportedEnumTypes, exportedVectorTypes);
    } else
        Assert(CastType<AtomicType>(type) != NULL);
}

/** Given a set of functions, return the set of structure and vector types
    present in the parameters to them.
 */
static void lGetExportedParamTypes(const std::vector<Symbol *> &funcs,
                                   std::vector<const StructType *> *exportedStructTypes,
                                   std::vector<const EnumType *> *exportedEnumTypes,
                                   std::vector<const VectorType *> *exportedVectorTypes) {
    for (unsigned int i = 0; i < funcs.size(); ++i) {
        const FunctionType *ftype = CastType<FunctionType>(funcs[i]->type);
        Assert(ftype != NULL);

        // Handle the return type
        lGetExportedTypes(ftype->GetReturnType(), exportedStructTypes, exportedEnumTypes, exportedVectorTypes);

        // And now the parameter types...
        for (int j = 0; j < ftype->GetNumParameters(); ++j) {
            lGetExportedTypes(ftype->GetParameterType(j), exportedStructTypes, exportedEnumTypes, exportedVectorTypes);
        }
    }
}

static void lPrintFunctionDeclarations(FILE *file, const std::vector<Symbol *> &funcs, bool useExternC = 1,
                                       bool rewriteForDispatch = false) {
    if (useExternC)
        fprintf(file, "#if defined(__cplusplus) && (! defined(__ISPC_NO_EXTERN_C) || !__ISPC_NO_EXTERN_C )\nextern "
                      "\"C\" {\n#endif // __cplusplus\n");
    // fprintf(file, "#ifdef __cplusplus\nextern \"C\" {\n#endif // __cplusplus\n");
    for (unsigned int i = 0; i < funcs.size(); ++i) {
        const FunctionType *ftype = CastType<FunctionType>(funcs[i]->type);
        Assert(ftype);
        std::string decl;
        std::string fname = funcs[i]->name;
        if (g->calling_conv == CallingConv::x86_vectorcall) {
            fname = "__vectorcall " + fname;
        }
        if (rewriteForDispatch) {
            decl = ftype->GetCDeclarationForDispatch(fname);
        } else {
            decl = ftype->GetCDeclaration(fname);
        }
        fprintf(file, "    extern %s;\n", decl.c_str());
    }
    if (useExternC)

        fprintf(file, "#if defined(__cplusplus) && (! defined(__ISPC_NO_EXTERN_C) || !__ISPC_NO_EXTERN_C )\n} /* end "
                      "extern C */\n#endif // __cplusplus\n");
    // fprintf(file, "#ifdef __cplusplus\n} /* end extern C */\n#endif // __cplusplus\n");
}

static bool lIsExported(const Symbol *sym) {
    const FunctionType *ft = CastType<FunctionType>(sym->type);
    Assert(ft);
    return ft->isExported;
}

static bool lIsExternC(const Symbol *sym) {
    const FunctionType *ft = CastType<FunctionType>(sym->type);
    Assert(ft);
    return ft->isExternC;
}

static void lUnescapeStringInPlace(std::string &str) {
    // There are many more escape sequences, but since this is a path,
    // we can get away with only supporting the basic ones (i.e. no
    // octal, hexadecimal or unicode values).
    for (std::string::iterator it = str.begin(); it != str.end(); ++it) {
        size_t pos = it - str.begin();
        std::string::iterator next = it + 1;
        if (*it == '\\' && next != str.end()) {
            switch (*next) {
#define UNESCAPE_SEQ(c, esc)                                                                                           \
    case c:                                                                                                            \
        *it = esc;                                                                                                     \
        str.erase(next);                                                                                               \
        it = str.begin() + pos;                                                                                        \
        break
                UNESCAPE_SEQ('\'', '\'');
                UNESCAPE_SEQ('?', '?');
                UNESCAPE_SEQ('\\', '\\');
                UNESCAPE_SEQ('a', '\a');
                UNESCAPE_SEQ('b', '\b');
                UNESCAPE_SEQ('f', '\f');
                UNESCAPE_SEQ('n', '\n');
                UNESCAPE_SEQ('r', '\r');
                UNESCAPE_SEQ('t', '\t');
                UNESCAPE_SEQ('v', '\v');
#undef UNESCAPE_SEQ
            }
        }
    }
}

bool Module::writeDeps(const char *fn, bool generateMakeRule, const char *tn, const char *sn) {
    if (fn && g->debugPrint) { // We may be passed nullptr for stdout output.
        printf("\nWriting dependencies to file %s\n", fn);
    }
    FILE *file = fn ? fopen(fn, "w") : stdout;
    if (!file) {
        perror("fopen");
        return false;
    }

    if (generateMakeRule) {
        Assert(tn);
        fprintf(file, "%s:", tn);
        // Rules always emit source first.
        if (sn && !IsStdin(sn)) {
            fprintf(file, " %s", sn);
        }
        std::string unescaped;

        for (std::set<std::string>::const_iterator it = registeredDependencies.begin();
             it != registeredDependencies.end(); ++it) {
            unescaped = *it; // As this is preprocessor output, paths come escaped.
            lUnescapeStringInPlace(unescaped);
            if (sn && !IsStdin(sn) &&
                0 == strcmp(sn, unescaped.c_str())) // If source has been passed, it's already emitted.
                continue;
            fprintf(file, " \\\n");
            fprintf(file, " %s", unescaped.c_str());
        }
        fprintf(file, "\n");
    } else {
        for (std::set<std::string>::const_iterator it = registeredDependencies.begin();
             it != registeredDependencies.end(); ++it)
            fprintf(file, "%s\n", it->c_str());
    }
    fclose(file);
    return true;
}

std::string emitOffloadParamStruct(const std::string &paramStructName, const Symbol *sym, const FunctionType *fct) {
    std::stringstream out;
    out << "struct " << paramStructName << " {" << std::endl;

    for (int i = 0; i < fct->GetNumParameters(); i++) {
        const Type *orgParamType = fct->GetParameterType(i);
        if (orgParamType->IsPointerType() || orgParamType->IsArrayType()) {
            /* we're passing pointers separately -- no pointers in that struct... */
            continue;
        }

        // const reference parameters can be passed as copies.
        const Type *paramType;
        if (orgParamType->IsReferenceType()) {
            if (!orgParamType->IsConstType()) {
                Error(sym->pos, "When emitting offload-stubs, \"export\"ed functions cannot have non-const "
                                "reference-type parameters.\n");
            }
            const ReferenceType *refType = static_cast<const ReferenceType *>(orgParamType);
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

bool Module::writeDevStub(const char *fn) {
    FILE *file = fopen(fn, "w");
    if (!file) {
        perror("fopen");
        return false;
    }
    fprintf(file, "//\n// %s\n// (device stubs automatically generated by the ispc compiler.)\n", fn);
    fprintf(file, "// DO NOT EDIT THIS FILE.\n//\n\n");
    fprintf(file, "#include \"ispc/dev/offload.h\"\n\n");

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
    lGetExportedParamTypes(exportedFuncs, &exportedStructTypes, &exportedEnumTypes, &exportedVectorTypes);

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
            // Error(sym->pos,"When emitting offload-stubs, \"export\"ed functions cannot have non-void return
            // types.\n");
            Warning(sym->pos,
                    "When emitting offload-stubs, ignoring \"export\"ed function with non-void return types.\n");
            continue;
        }

        // -------------------------------------------------------
        // first, emit a struct that holds the parameters
        // -------------------------------------------------------
        std::string paramStructName = std::string("__ispc_dev_stub_") + sym->name;
        std::string paramStruct = emitOffloadParamStruct(paramStructName, sym, fct);
        fprintf(file, "%s\n", paramStruct.c_str());
        // -------------------------------------------------------
        // then, emit a fct stub that unpacks the parameters and pointers
        // -------------------------------------------------------
        fprintf(file,
                "void __ispc_dev_stub_%s(\n"
                "            uint32_t         in_BufferCount,\n"
                "            void**           in_ppBufferPointers,\n"
                "            uint64_t*        in_pBufferLengths,\n"
                "            void*            in_pMiscData,\n"
                "            uint16_t         in_MiscDataLength,\n"
                "            void*            in_pReturnValue,\n"
                "            uint16_t         in_ReturnValueLength)\n",
                sym->name.c_str());
        fprintf(file, "{\n");
        fprintf(file, "  struct %s args;\n  memcpy(&args,in_pMiscData,sizeof(args));\n", paramStructName.c_str());
        std::stringstream funcall;

        funcall << "ispc::" << sym->name << "(";
        for (int i = 0; i < fct->GetNumParameters(); i++) {
            // get param type and make it non-const, so we can write while unpacking
            // const Type *paramType = fct->GetParameterType(i)->GetAsNonConstType();
            const Type *paramType; // = fct->GetParameterType(i)->GetAsNonConstType();
            const Type *orgParamType = fct->GetParameterType(i);
            if (orgParamType->IsReferenceType()) {
                if (!orgParamType->IsConstType()) {
                    Error(sym->pos, "When emitting offload-stubs, \"export\"ed functions cannot have non-const "
                                    "reference-type parameters.\n");
                }
                const ReferenceType *refType = static_cast<const ReferenceType *>(orgParamType);
                paramType = refType->GetReferenceTarget()->GetAsNonConstType();
            } else {
                paramType = orgParamType->GetAsNonConstType();
            }

            std::string paramName = fct->GetParameterName(i);
            std::string paramTypeName = paramType->GetString();

            if (i)
                funcall << ", ";
            std::string tmpArgName = std::string("_") + paramName;
            if (paramType->IsPointerType() || paramType->IsArrayType()) {
                std::string tmpArgDecl = paramType->GetCDeclaration(tmpArgName);
                fprintf(file, "  %s;\n", tmpArgDecl.c_str());
                fprintf(file, "  (void *&)%s = ispc_dev_translate_pointer(*in_ppBufferPointers++);\n",
                        tmpArgName.c_str());
                funcall << tmpArgName;
            } else {
                funcall << "args." << paramName;
            }
        }
        funcall << ");";
        fprintf(file, "  %s\n", funcall.str().c_str());
        fprintf(file, "}\n\n");
    }

    // end extern "C"
    fprintf(file, "}/* end extern C */\n");

    fclose(file);
    return true;
}

bool Module::writeCPPStub(const char *outFileName) { return writeCPPStub(this, outFileName); }

bool Module::writeCPPStub(Module *module, const char *outFileName) {
    // Get a file descriptor corresponding to where we want the output to
    // go.  If we open it, it'll be closed by the llvm::raw_fd_ostream
    // destructor.
    int fd;
    int flags = O_CREAT | O_WRONLY | O_TRUNC;

    if (!outFileName) {
        return false;
    } else if (!strcmp("-", outFileName)) {
        fd = 1;
    } else {
#ifdef ISPC_HOST_IS_WINDOWS
        fd = _open(outFileName, flags, 0644);
#else
        fd = open(outFileName, flags, 0644);
#endif // ISPC_HOST_IS_WINDOWS
        if (fd == -1) {
            perror(outFileName);
            return false;
        }
    }

    // The CPP stream should have been initialized: print and clean it up.
    Assert(module->bufferCPP && "`bufferCPP` should not be null");
    llvm::raw_fd_ostream fos(fd, (fd != 1), false);
    fos << module->bufferCPP->str;
    module->bufferCPP.reset();

    return true;
}

bool Module::writeHostStub(const char *fn) {
    FILE *file = fopen(fn, "w");
    if (!file) {
        perror("fopen");
        return false;
    }
    fprintf(file, "//\n// %s\n// (device stubs automatically generated by the ispc compiler.)\n", fn);
    fprintf(file, "// DO NOT EDIT THIS FILE.\n//\n\n");
    fprintf(file, "#include \"ispc/host/offload.h\"\n\n");
    fprintf(
        file,
        "// note(iw): Host stubs do not get extern C linkage -- dev-side already uses that for the same symbols.\n\n");
    // fprintf(file,"#ifdef __cplusplus\nextern \"C\" {\n#endif // __cplusplus\n");

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
    lGetExportedParamTypes(exportedFuncs, &exportedStructTypes, &exportedEnumTypes, &exportedVectorTypes);

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
            Warning(sym->pos,
                    "When emitting offload-stubs, ignoring \"export\"ed function with non-void return types.\n");
            continue;
        }

        // -------------------------------------------------------
        // first, emit a struct that holds the parameters
        // -------------------------------------------------------
        std::string paramStructName = std::string("__ispc_dev_stub_") + sym->name;
        std::string paramStruct = emitOffloadParamStruct(paramStructName, sym, fct);
        fprintf(file, "%s\n", paramStruct.c_str());
        // -------------------------------------------------------
        // then, emit a fct stub that unpacks the parameters and pointers
        // -------------------------------------------------------

        std::string decl = fct->GetCDeclaration(sym->name);
        fprintf(file, "extern %s {\n", decl.c_str());
        int numPointers = 0;
        fprintf(file, "  %s __args;\n", paramStructName.c_str());

        // ------------------------------------------------------------------
        // write args, and save pointers for later
        // ------------------------------------------------------------------
        std::stringstream pointerArgs;
        for (int i = 0; i < fct->GetNumParameters(); i++) {
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

            fprintf(file, "  __args.%s = %s;\n", paramName.c_str(), paramName.c_str());
        }
        // ------------------------------------------------------------------
        // writer pointer list
        // ------------------------------------------------------------------
        if (numPointers == 0)
            pointerArgs << "NULL";
        fprintf(file, "  void *ptr_args[] = { %s };\n", pointerArgs.str().c_str());

        // ------------------------------------------------------------------
        // ... and call the kernel with those args
        // ------------------------------------------------------------------
        fprintf(file, "  static ispc_kernel_handle_t kernel_handle = NULL;\n");
        fprintf(file, "  if (!kernel_handle) kernel_handle = ispc_host_get_kernel_handle(\"__ispc_dev_stub_%s\");\n",
                sym->name.c_str());
        fprintf(file, "  assert(kernel_handle);\n");
        fprintf(file,
                "  ispc_host_call_kernel(kernel_handle,\n"
                "                        &__args, sizeof(__args),\n"
                "                        ptr_args,%i);\n",
                numPointers);
        fprintf(file, "}\n\n");
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

bool Module::writeHeader(const char *fn) {
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

    if (g->noPragmaOnce)
        fprintf(f, "#ifndef %s\n#define %s\n\n", guard.c_str(), guard.c_str());
    else
        fprintf(f, "#pragma once\n");

    fprintf(f, "#include <stdint.h>\n\n");

    if (g->emitInstrumentation) {
        fprintf(f, "#define ISPC_INSTRUMENTATION 1\n");
        fprintf(f, "#if defined(__cplusplus) && (! defined(__ISPC_NO_EXTERN_C) || !__ISPC_NO_EXTERN_C )\nextern \"C\" "
                   "{\n#endif // __cplusplus\n");
        fprintf(f, "  void ISPCInstrument(const char *fn, const char *note, int line, uint64_t mask);\n");
        fprintf(f, "#if defined(__cplusplus) && (! defined(__ISPC_NO_EXTERN_C) || !__ISPC_NO_EXTERN_C )\n} /* end "
                   "extern C */\n#endif // __cplusplus\n");
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
    lGetExportedParamTypes(exportedFuncs, &exportedStructTypes, &exportedEnumTypes, &exportedVectorTypes);
    lGetExportedParamTypes(externCFuncs, &exportedStructTypes, &exportedEnumTypes, &exportedVectorTypes);

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
    if (g->noPragmaOnce)
        fprintf(f, "\n#endif // %s\n", guard.c_str());

    fclose(f);
    return true;
}

struct ispc::DispatchHeaderInfo {
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

bool Module::writeDispatchHeader(DispatchHeaderInfo *DHI) {
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
        if (g->noPragmaOnce)
            fprintf(f, "#ifndef %s\n#define %s\n\n", guard.c_str(), guard.c_str());
        else
            fprintf(f, "#pragma once\n");

        fprintf(f, "#include <stdint.h>\n\n");

        if (g->emitInstrumentation) {
            fprintf(f, "#define ISPC_INSTRUMENTATION 1\n");
            fprintf(f, "#if defined(__cplusplus) && (! defined(__ISPC_NO_EXTERN_C) || !__ISPC_NO_EXTERN_C )\nextern "
                       "\"C\" {\n#endif // __cplusplus\n");
            fprintf(f, "  void ISPCInstrument(const char *fn, const char *note, int line, uint64_t mask);\n");
            fprintf(f, "#if defined(__cplusplus) && (! defined(__ISPC_NO_EXTERN_C) || !__ISPC_NO_EXTERN_C )\n} /* end "
                       "extern C */\n#endif // __cplusplus\n");
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

    if ((DHI->Emit4 && (programCount == 4)) || (DHI->Emit8 && (programCount == 8)) ||
        (DHI->Emit16 && (programCount == 16))) {
        // Get all of the struct, vector, and enumerant types used as function
        // parameters.  These vectors may have repeats.
        std::vector<const StructType *> exportedStructTypes;
        std::vector<const EnumType *> exportedEnumTypes;
        std::vector<const VectorType *> exportedVectorTypes;
        lGetExportedParamTypes(exportedFuncs, &exportedStructTypes, &exportedEnumTypes, &exportedVectorTypes);
        lGetExportedParamTypes(externCFuncs, &exportedStructTypes, &exportedEnumTypes, &exportedVectorTypes);

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
        } else if (programCount == 8) {
            DHI->Emit8 = false;
        } else if (programCount == 16) {
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
        if (g->noPragmaOnce)
            fprintf(f, "\n#endif // %s\n", guard.c_str());
        DHI->EmitBackMatter = false;
    }
    return true;
}

int Module::execPreprocessor(const char *infilename, llvm::raw_string_ostream *ostream) const {
    clang::CompilerInstance inst;

    llvm::raw_fd_ostream stderrRaw(2, false);

    clang::DiagnosticOptions *diagOptions = new clang::DiagnosticOptions();
    clang::TextDiagnosticPrinter *diagPrinter = new clang::TextDiagnosticPrinter(stderrRaw, diagOptions);

    llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagIDs(new clang::DiagnosticIDs);
    clang::DiagnosticsEngine *diagEngine = new clang::DiagnosticsEngine(diagIDs, diagOptions, diagPrinter);
    diagEngine->setSuppressAllDiagnostics(g->ignoreCPPErrors);

    inst.setDiagnostics(diagEngine);

    inst.createFileManager();

    const std::shared_ptr<clang::TargetOptions> &options = std::make_shared<clang::TargetOptions>(inst.getTargetOpts());

    llvm::Triple triple(module->getTargetTriple());
    if (triple.getTriple().empty()) {
        triple.setTriple(llvm::sys::getDefaultTargetTriple());
    }

    options->Triple = triple.getTriple();

    clang::TargetInfo *target = clang::TargetInfo::CreateTargetInfo(inst.getDiagnostics(), options);
    inst.setTarget(target);
    inst.createSourceManager(inst.getFileManager());

    clang::FrontendInputFile inputFile(infilename, clang::InputKind());

    inst.InitializeSourceManager(inputFile);

    // Don't remove comments in the preprocessor, so that we can accurately
    // track the source file position by handling them ourselves.
    inst.getPreprocessorOutputOpts().ShowComments = 1;

    inst.getPreprocessorOutputOpts().ShowCPP = 1;
    clang::HeaderSearchOptions &headerOpts = inst.getHeaderSearchOpts();
    headerOpts.UseBuiltinIncludes = 0;
    headerOpts.UseStandardSystemIncludes = 0;
    headerOpts.UseStandardCXXIncludes = 0;
#ifndef ISPC_NO_DUMPS
    if (g->debugPrint)
        headerOpts.Verbose = 1;
#endif
    for (int i = 0; i < (int)g->includePath.size(); ++i) {
        headerOpts.AddPath(g->includePath[i], clang::frontend::Angled, false /* not a framework */,
                           true /* ignore sys root */);
    }

    clang::PreprocessorOptions &opts = inst.getPreprocessorOpts();

    // Add defs for ISPC and PI
    opts.addMacroDef("ISPC");
    opts.addMacroDef("PI=3.1415926535");

    if (g->enableLLVMIntrinsics) {
        opts.addMacroDef("ISPC_LLVM_INTRINSICS_ENABLED");
    }
    // Add defs for ISPC_UINT_IS_DEFINED.
    // This lets the user know uint* is part of language.
    opts.addMacroDef("ISPC_UINT_IS_DEFINED");

    // Add #define for current compilation target
    char targetMacro[128];
    snprintf(targetMacro, sizeof(targetMacro), "ISPC_TARGET_%s", g->target->GetISAString());
    char *p = targetMacro;
    while (*p) {
        *p = toupper(*p);
        if (*p == '-')
            *p = '_';
        ++p;
    }

    // Add 'TARGET_WIDTH' macro to expose vector width to user.
    std::string TARGET_WIDTH = "TARGET_WIDTH=" + std::to_string(g->target->getVectorWidth());
    opts.addMacroDef(TARGET_WIDTH);

    // Add 'TARGET_ELEMENT_WIDTH' macro to expose element width to user.
    std::string TARGET_ELEMENT_WIDTH = "TARGET_ELEMENT_WIDTH=" + std::to_string(g->target->getDataTypeWidth() / 8);
    opts.addMacroDef(TARGET_ELEMENT_WIDTH);

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

    constexpr int buf_size = 25;
    char ispc_major[buf_size], ispc_minor[buf_size];
    snprintf(ispc_major, buf_size, "ISPC_MAJOR_VERSION=%d", ISPC_VERSION_MAJOR);
    snprintf(ispc_minor, buf_size, "ISPC_MINOR_VERSION=%d", ISPC_VERSION_MINOR);
    opts.addMacroDef(ispc_major);
    opts.addMacroDef(ispc_minor);

    if (g->target->hasFp64Support()) {
        opts.addMacroDef("ISPC_FP64_SUPPORTED ");
    }

    if (g->includeStdlib) {
        if (g->opt.disableAsserts)
            opts.addMacroDef("assert(x)=");
        else
            opts.addMacroDef("assert(x)=__assert(#x, x)");
    }

    for (unsigned int i = 0; i < g->cppArgs.size(); ++i) {
        // Sanity check--should really begin with -D
        if (g->cppArgs[i].substr(0, 2) == "-D") {
            opts.addMacroDef(g->cppArgs[i].substr(2));
        }
    }

    if (g->target->isXeTarget()) {
        opts.addMacroDef("taskIndex=__taskIndex()");
        opts.addMacroDef("taskCount=__taskCount()");
        opts.addMacroDef("taskCount0=__taskCount0()");
        opts.addMacroDef("taskCount1=__taskCount1()");
        opts.addMacroDef("taskCount2=__taskCount2()");
        opts.addMacroDef("taskIndex0=__taskIndex0()");
        opts.addMacroDef("taskIndex1=__taskIndex1()");
        opts.addMacroDef("taskIndex2=__taskIndex2()");
    }

    inst.getLangOpts().LineComment = 1;
    inst.createPreprocessor(clang::TU_Complete);

    diagPrinter->BeginSourceFile(inst.getLangOpts(), &inst.getPreprocessor());
    clang::DoPrintPreprocessedInput(inst.getPreprocessor(), ostream, inst.getPreprocessorOutputOpts());
    diagPrinter->EndSourceFile();

    // Return preprocessor diagnostic errors after processing
    return static_cast<int>(diagEngine->hasErrorOccurred());
}

// Given an output filename of the form "foo.obj", and an ISA name like
// "avx", return a string with the ISA name inserted before the original
// filename's suffix, like "foo_avx.obj".
static std::string lGetTargetFileName(const char *outFileName, const char *isaString) {
    int bufferSize = strlen(outFileName) + 16;
    char *targetOutFileName = new char[bufferSize];
    if (strrchr(outFileName, '.') != NULL) {
        // Copy everything up to the last '.'
        int count = strrchr(outFileName, '.') - outFileName;
        strncpy(targetOutFileName, outFileName, count);
        targetOutFileName[count] = '\0';

        // Add the ISA name
        strncat(targetOutFileName, "_", bufferSize - strlen(targetOutFileName) - 1);
        strncat(targetOutFileName, isaString, bufferSize - strlen(targetOutFileName) - 1);

        // And finish with the original file suffix
        strncat(targetOutFileName, strrchr(outFileName, '.'), bufferSize - strlen(targetOutFileName) - 1);
    } else {
        // Can't find a '.' in the filename, so just append the ISA suffix
        // to what we weregiven
        strncpy(targetOutFileName, outFileName, bufferSize - 1);
        targetOutFileName[bufferSize - 1] = '\0';
        strncat(targetOutFileName, "_", bufferSize - strlen(targetOutFileName) - 1);
        strncat(targetOutFileName, isaString, bufferSize - strlen(targetOutFileName) - 1);
    }
    return targetOutFileName;
}

static bool lSymbolIsExported(const Symbol *s) { return s->exportedFunction != NULL; }

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
static void lGetExportedFunctions(SymbolTable *symbolTable, std::map<std::string, FunctionTargetVariants> &functions) {
    std::vector<Symbol *> syms;
    symbolTable->GetMatchingFunctions(lSymbolIsExported, &syms);
    for (unsigned int i = 0; i < syms.size(); ++i) {
        FunctionTargetVariants &ftv = functions[syms[i]->name];
        ftv.func[g->target->getISA()] = syms[i]->exportedFunction;
        ftv.FTs[g->target->getISA()] = CastType<FunctionType>(syms[i]->type);
    }
}

static llvm::FunctionType *lGetVaryingDispatchType(FunctionTargetVariants &funcs) {
    llvm::Type *ptrToInt8Ty = llvm::Type::getInt8PtrTy(*g->ctx);
    llvm::FunctionType *resultFuncTy = NULL;

    for (int i = 0; i < Target::NUM_ISAS; ++i) {
        if (funcs.func[i] == NULL) {
            continue;
        } else {
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
static void lCreateDispatchFunction(llvm::Module *module, llvm::Function *setISAFunc, llvm::Value *systemBestISAPtr,
                                    const std::string &name, FunctionTargetVariants &funcs) {
    // The llvm::Function pointers in funcs are pointers to functions in
    // different llvm::Modules, so we can't call them directly.  Therefore,
    // we'll start by generating an 'extern' declaration of each one that
    // we have in the current module so that we can then call out to that.
    llvm::Function *targetFuncs[Target::NUM_ISAS];

    // New helper function checks to see if we need to rewrite the
    // type for the dispatch function in case of pointers to varyings
    llvm::FunctionType *ftype = lGetVaryingDispatchType(funcs);

    // Now we insert type-punned declarations for dispatched functions.
    // This is needed when compiling modules for a set of architectures
    // with different vector lengths. Due to restrictions, the return
    // type is the same across all architectures, however in different
    // modules it may have dissimilar names. The loop below works this
    // around.

    for (int i = 0; i < Target::NUM_ISAS; ++i) {
        if (funcs.func[i]) {

            targetFuncs[i] =
                llvm::Function::Create(ftype, llvm::GlobalValue::ExternalLinkage, funcs.func[i]->getName(), module);
            g->target->markFuncWithCallingConv(targetFuncs[i]);
        } else
            targetFuncs[i] = NULL;
    }

    bool voidReturn = ftype->getReturnType()->isVoidTy();

    // Now we can emit the definition of the dispatch function..
    llvm::Function *dispatchFunc =
        llvm::Function::Create(ftype, llvm::GlobalValue::ExternalLinkage, name.c_str(), module);
    g->target->markFuncWithCallingConv(dispatchFunc);
    llvm::BasicBlock *bblock = llvm::BasicBlock::Create(*g->ctx, "entry", dispatchFunc);

    // Start by calling out to the function that determines the system's
    // ISA and sets __system_best_isa, if it hasn't been set yet.
    llvm::CallInst::Create(setISAFunc, "", bblock);

    // Now we can load the system's ISA enumerant
#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
    llvm::PointerType *ptr_type = llvm::dyn_cast<llvm::PointerType>(systemBestISAPtr->getType());
    Assert(ptr_type);
    llvm::Value *systemISA =
        new llvm::LoadInst(ptr_type->getPointerElementType(), systemBestISAPtr, "system_isa", bblock);
#else
    llvm::Value *systemISA = new llvm::LoadInst(systemBestISAPtr, "system_isa", bblock);
#endif

    // Now emit code that works backwards though the available variants of
    // the function.  We'll call out to the first one we find that will run
    // successfully on the system the code is running on.  In working
    // through the candidate ISAs here backward, we're taking advantage of
    // the expectation that they are ordered in the Target::ISA enumerant
    // from least to most capable.
    for (int i = Target::NUM_ISAS - 1; i >= 0; --i) {
        if (targetFuncs[i] == NULL)
            continue;

        // Emit code to see if the system can run the current candidate
        // variant successfully--"is the system's ISA enumerant value >=
        // the enumerant value of the current candidate?"

        llvm::Value *ok = llvm::CmpInst::Create(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_SGE, systemISA,
                                                LLVMInt32(i), "isa_ok", bblock);
        llvm::BasicBlock *callBBlock = llvm::BasicBlock::Create(*g->ctx, "do_call", dispatchFunc);
        llvm::BasicBlock *nextBBlock = llvm::BasicBlock::Create(*g->ctx, "next_try", dispatchFunc);
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
                args.push_back(&*argIter);
            } else {
                llvm::CastInst *argCast = llvm::CastInst::CreatePointerCast(&*argIter, targsIter->getType(),
                                                                            "dpatch_arg_bitcast", callBBlock);
                args.push_back(argCast);
            }
        }
        if (voidReturn) {
            llvm::CallInst *callInst = llvm::CallInst::Create(targetFuncs[i], args, "", callBBlock);
            if (g->calling_conv == CallingConv::x86_vectorcall) {
                callInst->setCallingConv(llvm::CallingConv::X86_VectorCall);
            }
            llvm::ReturnInst::Create(*g->ctx, callBBlock);
        } else {
            llvm::CallInst *callInst = llvm::CallInst::Create(targetFuncs[i], args, "ret_value", callBBlock);
            if (g->calling_conv == CallingConv::x86_vectorcall) {
                callInst->setCallingConv(llvm::CallingConv::X86_VectorCall);
            }
            llvm::ReturnInst::Create(*g->ctx, callInst, callBBlock);
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

// Initialize a dispatch module
static llvm::Module *lInitDispatchModule() {
    llvm::Module *module = new llvm::Module("dispatch_module", *g->ctx);

    module->setTargetTriple(g->target->GetTripleString());

    // DataLayout information supposed to be managed in single place in Target class.
    module->setDataLayout(g->target->getDataLayout()->getStringRepresentation());

    // First, link in the definitions from the builtins-dispatch.ll file.
    const BitcodeLib *dispatch = g->target_registry->getDispatchLib(g->target_os);
    Assert(dispatch);
    AddBitcodeToModule(dispatch, module);

    return module;
}

// Complete the creation of a dispatch module.
// Given a map that holds the mapping from each of the 'export'ed functions
// in the ispc program to the target-specific variants of the function,
// create a llvm::Module that has a dispatch function for each exported
// function that checks the system's capabilities and picks the most
// appropriate compiled variant of the function.
static void lEmitDispatchModule(llvm::Module *module, std::map<std::string, FunctionTargetVariants> &functions) {
    // Get pointers to things we need below
    llvm::Function *setFunc = module->getFunction("__set_system_isa");
    Assert(setFunc != NULL);
    llvm::Value *systemBestISAPtr = module->getGlobalVariable("__system_best_isa", true);
    Assert(systemBestISAPtr != NULL);

    // For each exported function, create the dispatch function
    std::map<std::string, FunctionTargetVariants>::iterator iter;
    for (iter = functions.begin(); iter != functions.end(); ++iter)
        lCreateDispatchFunction(module, setFunc, systemBestISAPtr, iter->first, iter->second);

    // Do some rudimentary cleanup of the final result and make sure that
    // the module is all ok.
    llvm::legacy::PassManager optPM;
    optPM.add(llvm::createGlobalDCEPass());
    optPM.add(llvm::createVerifierPass());
    optPM.run(*module);
}

// Determines if two types are compatible
static bool lCompatibleTypes(llvm::Type *Ty1, llvm::Type *Ty2) {
    while (Ty1->getTypeID() == Ty2->getTypeID())
        switch (Ty1->getTypeID()) {
        case llvm::ArrayType::ArrayTyID:
            if (Ty1->getArrayNumElements() != Ty2->getArrayNumElements())
                return false;
            Ty1 = Ty1->getArrayElementType();
            Ty2 = Ty2->getArrayElementType();
            break;

        case llvm::ArrayType::PointerTyID:
            Ty1 = Ty1->getPointerElementType();
            Ty2 = Ty2->getPointerElementType();
            break;

        case llvm::ArrayType::StructTyID: {
            llvm::StructType *STy1 = llvm::dyn_cast<llvm::StructType>(Ty1);
            llvm::StructType *STy2 = llvm::dyn_cast<llvm::StructType>(Ty2);
            return STy1 && STy2 && STy1->isLayoutIdentical(STy2);
        }
        default:
            // Pointers for compatible simple types are assumed equal
            return Ty1 == Ty2;
        }
    return false;
}

// Grab all of the global value definitions from the module and change them
// to be declarations; we'll emit a single definition of each global in the
// final module used with the dispatch functions, so that we don't have
// multiple definitions of them, one in each of the target-specific output
// files.
static void lExtractOrCheckGlobals(llvm::Module *msrc, llvm::Module *mdst, bool check) {
    llvm::Module::global_iterator iter;
    llvm::ValueToValueMapTy VMap;

    for (iter = msrc->global_begin(); iter != msrc->global_end(); ++iter) {
        llvm::GlobalVariable *gv = &*iter;
        // Is it a global definition?
        if (gv->getLinkage() == llvm::GlobalValue::ExternalLinkage && gv->hasInitializer()) {
            llvm::Type *type = gv->getType()->PTR_ELT_TYPE();
            Symbol *sym = m->symbolTable->LookupVariable(gv->getName().str().c_str());
            Assert(sym != NULL);

            // Check presence and compatibility for the current global
            if (check) {
                llvm::GlobalVariable *exist = mdst->getGlobalVariable(gv->getName());
                Assert(exist != NULL);

                // It is possible that the types may not match: for
                // example, this happens with varying globals if we
                // compile to different vector widths
                if (!lCompatibleTypes(exist->getType(), gv->getType())) {
                    Warning(sym->pos,
                            "Mismatch in size/layout of global "
                            "variable \"%s\" with different targets. "
                            "Globals must not include \"varying\" types or arrays "
                            "with size based on programCount when compiling to "
                            "targets with differing vector widths.",
                            gv->getName().str().c_str());
                }
            }
            // Alternatively, create it anew and make it match the original
            else {
                llvm::GlobalVariable *newGlobal =
                    new llvm::GlobalVariable(*mdst, type, gv->isConstant(), llvm::GlobalValue::ExternalLinkage,
                                             (llvm::Constant *)nullptr, gv->getName());
                VMap[&*iter] = newGlobal;

                newGlobal->setInitializer(llvm::MapValue(iter->getInitializer(), VMap));
                newGlobal->copyAttributesFrom(gv);
            }
            // Turn this into an 'extern' declaration by clearing its
            // initializer.
            gv->setInitializer(NULL);
        }
    }
}

int Module::CompileAndOutput(const char *srcFile, Arch arch, const char *cpu, std::vector<ISPCTarget> targets,
                             OutputFlags outputFlags, OutputType outputType, const char *outFileName,
                             const char *headerFileName, const char *depsFileName, const char *depsTargetName,
                             const char *hostStubFileName, const char *devStubFileName) {
    if (targets.size() == 0 || targets.size() == 1) {
        // We're only compiling to a single target
        // TODO something wrong here
        ISPCTarget target = ISPCTarget::none;
        if (targets.size() == 1) {
            target = targets[0];
        }
        g->target = new Target(arch, cpu, target, 0 != (outputFlags & GeneratePIC), g->printTarget);
        if (!g->target->isValid())
            return 1;

        m = new Module(srcFile);
        const int compileResult = m->CompileFile();

        llvm::TimeTraceScope TimeScope("Backend");

        if (compileResult == 0) {
#ifdef ISPC_XE_ENABLED
            if (outputType == Asm || outputType == Object) {
                if (g->target->isXeTarget()) {
                    Error(SourcePos(), "%s output is not supported yet for Xe targets. ",
                          (outputType == Asm) ? "assembly" : "binary");
                    return 1;
                }
            }
            if (g->target->isXeTarget() && outputType == OutputType::Object) {
                outputType = OutputType::ZEBIN;
            }
            if (!g->target->isXeTarget() && (outputType == OutputType::ZEBIN || outputType == OutputType::SPIRV)) {
                Error(SourcePos(), "SPIR-V and L0 binary formats are supported for Xe target only");
                return 1;
            }
#endif
            if (outFileName != NULL)
                if (!m->writeOutput(outputType, outputFlags, outFileName))
                    return 1;
            if (headerFileName != NULL)
                if (!m->writeOutput(Module::Header, outputFlags, headerFileName))
                    return 1;
            if (depsFileName != NULL || (outputFlags & Module::OutputDepsToStdout)) {
                std::string targetName;
                if (depsTargetName)
                    targetName = depsTargetName;
                else if (outFileName)
                    targetName = outFileName;
                else if (!IsStdin(srcFile)) {
                    targetName = srcFile;
                    size_t dot = targetName.find_last_of('.');
                    if (dot != std::string::npos)
                        targetName.erase(dot, std::string::npos);
                    targetName.append(".o");
                } else
                    targetName = "a.out";
                if (!m->writeOutput(Module::Deps, outputFlags, depsFileName, targetName.c_str(), srcFile))
                    return 1;
            }
            if (hostStubFileName != NULL)
                if (!m->writeOutput(Module::HostStub, outputFlags, hostStubFileName))
                    return 1;
            if (devStubFileName != NULL)
                if (!m->writeOutput(Module::DevStub, outputFlags, devStubFileName))
                    return 1;
        } else {
            ++m->errorCount;
        }

        int errorCount = m->errorCount;
        delete m;
        m = NULL;

        delete g->target;
        g->target = NULL;

        return errorCount > 0;
    } else {
        if (IsStdin(srcFile)) {
            Error(SourcePos(), "Compiling programs from standard input isn't "
                               "supported when compiling for multiple targets.  Please use "
                               "an intermediate temporary file.");
            return 1;
        }
        if (cpu != NULL) {
            Error(SourcePos(), "Illegal to specify cpu type when compiling "
                               "for multiple targets.");
            return 1;
        }

        // The user supplied multiple targets
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

        llvm::Module *dispatchModule = NULL;

        std::map<std::string, FunctionTargetVariants> exportedFunctions;
        int errorCount = 0;

        // Handle creating a "generic" header file for multiple targets
        // that use exported varyings
        DispatchHeaderInfo DHI;
        if (headerFileName != NULL) {
            DHI.file = fopen(headerFileName, "w");
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
            g->target = new Target(arch, cpu, targets[i], 0 != (outputFlags & GeneratePIC), g->printTarget);
            if (!g->target->isValid())
                return 1;

            // Issue an error if we've already compiled to a variant of
            // this target ISA.  (It doesn't make sense to compile to both
            // avx and avx-x2, for example.)
            if (targetMachines[g->target->getISA()] != NULL) {
                Error(SourcePos(),
                      "Can't compile to multiple variants of %s "
                      "target!\n",
                      g->target->GetISAString());
                return 1;
            }
            targetMachines[g->target->getISA()] = g->target->GetTargetMachine();

            m = new Module(srcFile);
            const int compileResult = m->CompileFile();

            llvm::TimeTraceScope TimeScope("Backend");

            if (compileResult == 0) {
                // Create the dispatch module, unless already created;
                // in the latter case, just do the checking
                bool check = (dispatchModule != NULL);
                if (!check)
                    dispatchModule = lInitDispatchModule();
                lExtractOrCheckGlobals(m->module, dispatchModule, check);

                // Grab pointers to the exported functions from the module we
                // just compiled, for use in generating the dispatch function
                // later.
                lGetExportedFunctions(m->symbolTable, exportedFunctions);

                if (outFileName != NULL) {
                    std::string targetOutFileName;
                    const char *isaName = g->target->GetISAString();
                    targetOutFileName = lGetTargetFileName(outFileName, isaName);
                    if (!m->writeOutput(outputType, outputFlags, targetOutFileName.c_str())) {
                        return 1;
                    }
                }
            } else {
                ++m->errorCount;
            }

            errorCount += m->errorCount;
            if (errorCount != 0) {
                return 1;
            }

            // Only write the generate header file, if desired, the first
            // time through the loop here.
            if (headerFileName != NULL) {
                if (i == targets.size() - 1) {
                    // only print backmatter on the last target.
                    DHI.EmitBackMatter = true;
                }

                const char *isaName;
                isaName = g->target->GetISAString();
                std::string targetHeaderFileName = lGetTargetFileName(headerFileName, isaName);
                // write out a header w/o target name for the first target only
                if (!m->writeOutput(Module::Header, outputFlags, headerFileName, nullptr, nullptr, &DHI)) {
                    return 1;
                }
                if (!m->writeOutput(Module::Header, outputFlags, targetHeaderFileName.c_str())) {
                    return 1;
                }
                if (i == targets.size() - 1) {
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
        const char *firstISA = "";
        ISPCTarget firstTarget = ISPCTarget::none;
        while (i < Target::NUM_ISAS && firstTargetMachine == NULL) {
            firstISA = Target::ISAToTargetString((Target::ISA)i);
            firstTarget = ParseISPCTarget(firstISA);
            firstTargetMachine = targetMachines[i++];
        }
        Assert(strcmp(firstISA, "") != 0);
        Assert(firstTarget != ISPCTarget::none);
        Assert(firstTargetMachine != NULL);

        g->target = new Target(arch, cpu, firstTarget, 0 != (outputFlags & GeneratePIC), false);
        if (!g->target->isValid()) {
            return 1;
        }

        if (dispatchModule == NULL) {
            Error(SourcePos(), "Failed to create dispatch module.\n");
            return 1;
        }

        lEmitDispatchModule(dispatchModule, exportedFunctions);

        if (outFileName != NULL) {
            switch (outputType) {
            case CPPStub:
                // No preprocessor output for dispatch module.
                break;

            case Bitcode:
            case BitcodeText:
                if (!writeBitcode(dispatchModule, outFileName, outputType))
                    return 1;
                break;

            case Asm:
            case Object:
                if (!writeObjectFileOrAssembly(firstTargetMachine, dispatchModule, outputType, outFileName))
                    return 1;
                break;

            default:
                FATAL("Unexpected `outputType`");
            }
        }

        if (depsFileName != NULL || (outputFlags & Module::OutputDepsToStdout)) {
            std::string targetName;
            if (depsTargetName)
                targetName = depsTargetName;
            else if (outFileName)
                targetName = outFileName;
            else if (!IsStdin(srcFile)) {
                targetName = srcFile;
                size_t dot = targetName.find_last_of('.');
                if (dot != std::string::npos)
                    targetName.erase(dot, std::string::npos);
                targetName.append(".o");
            } else
                targetName = "a.out";
            if (!m->writeOutput(Module::Deps, outputFlags, depsFileName, targetName.c_str(), srcFile))
                return 1;
        }

        delete g->target;
        g->target = NULL;
        return errorCount > 0;
    }
}

void Module::clearCPPBuffer() {
    if (bufferCPP)
        bufferCPP.reset();
}
