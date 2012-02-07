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
#include <llvm/PassManager.h>
#include <llvm/PassRegistry.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Support/FormattedStream.h>
#include <llvm/Support/FileUtilities.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Target/TargetData.h>
#include <llvm/Analysis/Verifier.h>
#include <llvm/Support/CFG.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Frontend/Utils.h>
#include <clang/Basic/TargetInfo.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Support/Host.h>
#include <llvm/Assembly/PrintModulePass.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Bitcode/ReaderWriter.h>

///////////////////////////////////////////////////////////////////////////
// Module

Module::Module(const char *fn) {
    // It's a hack to do this here, but it must be done after the target
    // information has been set (so e.g. the vector width is known...)  In
    // particular, if we're compiling to multiple targets with different
    // vector widths, this needs to be redone each time through.
    InitLLVMUtil(g->ctx, g->target);

    filename = fn;
    errorCount = 0;
    symbolTable = new SymbolTable;
    ast = new AST;

    module = new llvm::Module(filename ? filename : "<stdin>", *g->ctx);
    module->setTargetTriple(g->target.GetTripleString());

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
            diBuilder->createCompileUnit(llvm::dwarf::DW_LANG_C99,  /* lang */
                                         name,  /* filename */
                                         directory, /* directory */
                                         "ispc", /* producer */
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
#ifndef LLVM_3_1svn
    if (g->opt.fastMath == true)
        llvm::UnsafeFPMath = true;
#endif // !LLVM_3_1svn

    // FIXME: it'd be nice to do this in the Module constructor, but this
    // function ends up calling into routines that expect the global
    // variable 'm' to be initialized and available (which it isn't until
    // the Module constructor returns...)
    DefineStdlib(symbolTable, g->ctx, module, g->includeStdlib);

    bool runPreprocessor = g->runCPP;

    extern void ParserInit();
    ParserInit();

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
Module::AddTypeDef(Symbol *sym) {
    // Typedefs are easy; just add the mapping between the given name and
    // the given type.
    symbolTable->AddType(sym->name.c_str(), sym->type, sym->pos);
}


void
Module::AddGlobalVariable(Symbol *sym, Expr *initExpr, bool isConst) {
    // These may be NULL due to errors in parsing; just gracefully return
    // here if so.
    if (sym == NULL || sym->type == NULL) {
        // But if these are NULL and there haven't been any previous
        // errors, something surprising is going on
        Assert(errorCount > 0);
        return;
    }

    if (symbolTable->LookupFunction(sym->name.c_str())) {
        Error(sym->pos, "Global variable \"%s\" shadows previously-declared "
              "function.", sym->name.c_str());
        return;
    }

    if (sym->storageClass == SC_EXTERN_C) {
        Error(sym->pos, "extern \"C\" qualifier can only be used for "
              "functions.");
        return;
    }

    if (sym->type == AtomicType::Void) {
        Error(sym->pos, "\"void\" type global variable is illegal.");
        return;
    }

    sym->type = ArrayType::SizeUnsizedArrays(sym->type, initExpr);
    if (sym->type == NULL)
        return;

    const ArrayType *at = dynamic_cast<const ArrayType *>(sym->type);
    if (at != NULL && at->TotalElementCount() == 0) {
        Error(sym->pos, "Illegal to declare a global variable with unsized "
              "array dimensions that aren't set with an initializer "
              "expression.");
        return;
    }
        
    LLVM_TYPE_CONST llvm::Type *llvmType = sym->type->LLVMType(g->ctx);
    if (llvmType == NULL)
        return;

    // See if we have an initializer expression for the global.  If so,
    // make sure it's a compile-time constant!
    llvm::Constant *llvmInitializer = NULL;
    if (sym->storageClass == SC_EXTERN || sym->storageClass == SC_EXTERN_C) {
        if (initExpr != NULL)
            Error(sym->pos, "Initializer can't be provided with \"extern\" "
                  "global variable \"%s\".", sym->name.c_str());
    }
    else if (initExpr != NULL) {
        initExpr = TypeCheck(initExpr);
        if (initExpr != NULL) {
            // We need to make sure the initializer expression is
            // the same type as the global.  (But not if it's an
            // ExprList; they don't have types per se / can't type
            // convert themselves anyway.)
            if (dynamic_cast<ExprList *>(initExpr) == NULL)
                initExpr = TypeConvertExpr(initExpr, sym->type, "initializer");
            
            if (initExpr != NULL) {
                initExpr = Optimize(initExpr);
                // Fingers crossed, now let's see if we've got a
                // constant value..
                llvmInitializer = initExpr->GetConstant(sym->type);

                if (llvmInitializer != NULL) {
                    if (sym->type->IsConstType())
                        // Try to get a ConstExpr associated with
                        // the symbol.  This dynamic_cast can
                        // validly fail, for example for types like
                        // StructTypes where a ConstExpr can't
                        // represent their values.
                        sym->constValue = 
                            dynamic_cast<ConstExpr *>(initExpr);
                }
                else
                    Error(initExpr->pos, "Initializer for global variable \"%s\" "
                          "must be a constant.", sym->name.c_str());
            }
        }
    }

    // If no initializer was provided or if we couldn't get a value
    // above, initialize it with zeros..
    if (llvmInitializer == NULL)
        llvmInitializer = llvm::Constant::getNullValue(llvmType);

    llvm::GlobalValue::LinkageTypes linkage =
        (sym->storageClass == SC_STATIC) ? llvm::GlobalValue::InternalLinkage :
        llvm::GlobalValue::ExternalLinkage;
    sym->storagePtr = new llvm::GlobalVariable(*module, llvmType, isConst, 
                                               linkage, llvmInitializer, 
                                               sym->name.c_str());
    symbolTable->AddVariable(sym);

    if (diBuilder && (sym->storageClass != SC_EXTERN)) {
        llvm::DIFile file = sym->pos.GetDIFile();
        diBuilder->createGlobalVariable(sym->name, 
                                        file,
                                        sym->pos.first_line,
                                        sym->type->GetDIType(file),
                                        (sym->storageClass == SC_STATIC),
                                        sym->storagePtr);
    }
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
lCheckForVaryingParameter(const Type *type, const std::string &name, 
                          SourcePos pos) {
    if (lRecursiveCheckVarying(type)) {
        const Type *t = type->GetBaseType();
        if (dynamic_cast<const StructType *>(t))
            Error(pos, "Struct parameter \"%s\" with varying member(s) is illegal "
                  "in an exported function.", name.c_str());
        else
            Error(pos, "Varying parameter \"%s\" is illegal in an exported function.",
                  name.c_str());
    }
}


/** Given a function type, loop through the function parameters and see if
    any are StructTypes.  If so, issue an error (this seems to be broken
    currently).

    @todo Fix passing structs from C/C++ to ispc functions.
 */
static void
lCheckForStructParameters(const FunctionType *ftype, SourcePos pos) {
    for (int i = 0; i < ftype->GetNumParameters(); ++i) {
        const Type *type = ftype->GetParameterType(i);
        if (dynamic_cast<const StructType *>(type) != NULL) {
            Error(pos, "Passing structs to/from application functions is "
                  "currently broken. Use a pointer or const pointer to the "
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
Module::AddFunctionDeclaration(Symbol *funSym, bool isInline) {
    const FunctionType *functionType = 
        dynamic_cast<const FunctionType *>(funSym->type);
    Assert(functionType != NULL);

    // If a global variable with the same name has already been declared
    // issue an error.
    if (symbolTable->LookupVariable(funSym->name.c_str()) != NULL) {
        Error(funSym->pos, "Function \"%s\" shadows previously-declared global variable. "
              "Ignoring this definition.",
              funSym->name.c_str());
        return;
    }

    std::vector<Symbol *> overloadFuncs;
    symbolTable->LookupFunction(funSym->name.c_str(), &overloadFuncs);
    if (overloadFuncs.size() > 0) {
        for (unsigned int i = 0; i < overloadFuncs.size(); ++i) {
            Symbol *overloadFunc = overloadFuncs[i];

            // Check for a redeclaration of a function with the same
            // name and type
            if (Type::Equal(overloadFunc->type, functionType))
                return;

            // If all of the parameter types match but the return type is
            // different, return an error--overloading by return type isn't
            // allowed.
            const FunctionType *ofType = 
                dynamic_cast<const FunctionType *>(overloadFunc->type);
            Assert(ofType != NULL);
            if (ofType->GetNumParameters() == functionType->GetNumParameters()) {
                int i;
                for (i = 0; i < functionType->GetNumParameters(); ++i) {
                    if (Type::Equal(ofType->GetParameterType(i),
                                    functionType->GetParameterType(i)) == false)
                        break;
                }
                if (i == functionType->GetNumParameters()) {
                    Error(funSym->pos, "Illegal to overload function by return "
                          "type only (previous declaration was at line %d of "
                          "file %s).", overloadFunc->pos.first_line,
                          overloadFunc->pos.name);
                    return;
                }
            }
        }
    }

    if (funSym->storageClass == SC_EXTERN_C) {
        // Make sure the user hasn't supplied both an 'extern "C"' and a
        // 'task' qualifier with the function
        if (functionType->isTask) {
            Error(funSym->pos, "\"task\" qualifier is illegal with C-linkage extern "
                  "function \"%s\".  Ignoring this function.", funSym->name.c_str());
            return;
        }

        std::vector<Symbol *> funcs;
        symbolTable->LookupFunction(funSym->name.c_str(), &funcs);
        if (funcs.size() > 0) {
            if (funcs.size() > 1) {
                // Multiple functions with this name have already been declared; 
                // can't overload here
                Error(funSym->pos, "Can't overload extern \"C\" function \"%s\"; "
                      "%d functions with the same name have already been declared.",
                      funSym->name.c_str(), (int)funcs.size());
                return;
            }

            // One function with the same name has been declared; see if it
            // has the same type as this one, in which case it's ok.
            if (Type::Equal(funcs[0]->type, funSym->type))
                return;
            else {
                Error(funSym->pos, "Can't overload extern \"C\" function \"%s\".",
                      funSym->name.c_str());
                return;
            }
        }
    }

    // Get the LLVM FunctionType
    bool includeMask = (funSym->storageClass != SC_EXTERN_C);
    LLVM_TYPE_CONST llvm::FunctionType *llvmFunctionType = 
        functionType->LLVMFunctionType(g->ctx, includeMask);
    if (llvmFunctionType == NULL)
        return;

    // And create the llvm::Function
    llvm::GlobalValue::LinkageTypes linkage = (funSym->storageClass == SC_STATIC ||
                                               isInline) ?
        llvm::GlobalValue::InternalLinkage : llvm::GlobalValue::ExternalLinkage;
    std::string functionName = ((funSym->storageClass == SC_EXTERN_C) ?
                                funSym->name : funSym->MangledName());
    if (g->mangleFunctionsWithTarget)
        functionName += g->target.GetISAString();
    llvm::Function *function = 
        llvm::Function::Create(llvmFunctionType, linkage, functionName.c_str(), 
                               module);

    // Set function attributes: we never throw exceptions
    function->setDoesNotThrow(true);
    if (!(funSym->storageClass == SC_EXTERN_C) && 
        !g->generateDebuggingSymbols &&
        isInline)
        function->addFnAttr(llvm::Attribute::AlwaysInline);
    if (functionType->isTask)
        // This also applies transitively to members I think? 
        function->setDoesNotAlias(1, true);

    // Make sure that the return type isn't 'varying' if the function is
    // 'export'ed.
    if (funSym->storageClass == SC_EXPORT && 
        lRecursiveCheckVarying(functionType->GetReturnType()))
        Error(funSym->pos, "Illegal to return a \"varying\" type from exported "
              "function \"%s\"", funSym->name.c_str());

    if (functionType->isTask && (functionType->GetReturnType() != AtomicType::Void))
        Error(funSym->pos, "Task-qualified functions must have void return type.");

    if (functionType->isExported || functionType->isExternC)
        lCheckForStructParameters(functionType, funSym->pos);

    // Loop over all of the arguments; process default values if present
    // and do other checks and parameter attribute setting.
    bool seenDefaultArg = false;
    int nArgs = functionType->GetNumParameters();
    for (int i = 0; i < nArgs; ++i) {
        const Type *argType = functionType->GetParameterType(i);
        const std::string &argName = functionType->GetParameterName(i);
        ConstExpr *defaultValue = functionType->GetParameterDefault(i);
        const SourcePos &argPos = functionType->GetParameterSourcePos(i);

        // If the function is exported, make sure that the parameter
        // doesn't have any varying stuff going on in it.
        if (funSym->storageClass == SC_EXPORT)
            lCheckForVaryingParameter(argType, argName, argPos);

        // ISPC assumes that no pointers alias.  (It should be possible to
        // specify when this is not the case, but this should be the
        // default.)  Set parameter attributes accordingly.  (Only for
        // uniform pointers, since varying pointers are int vectors...)
        if (!functionType->isTask && 
            ((dynamic_cast<const PointerType *>(argType) != NULL &&
              argType->IsUniformType()) ||
             dynamic_cast<const ReferenceType *>(argType) != NULL)) {

            // NOTE: LLVM indexes function parameters starting from 1.
            // This is unintuitive.
            function->setDoesNotAlias(i+1, true);
#if 0
            int align = 4 * RoundUpPow2(g->target.nativeVectorWidth);
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
    funSym->function = function;

    // Finally, we know all is good and we can add the function to the
    // symbol table
    bool ok = symbolTable->AddFunction(funSym);
    Assert(ok);
}


void
Module::AddFunctionDefinition(Symbol *sym, const std::vector<Symbol *> &args,
                              Stmt *code) {
    ast->AddFunction(sym, args, code);
}


bool
Module::writeOutput(OutputType outputType, const char *outFileName,
                    const char *includeFileName) {
#if defined(LLVM_3_0) || defined(LLVM_3_0svn) || defined(LLVM_3_1svn)
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
#ifndef LLVM_2_9
        case CXX:
            if (strcasecmp(suffix, "c") && strcasecmp(suffix, "cc") &&
                strcasecmp(suffix, "c++") && strcasecmp(suffix, "cxx") &&
                strcasecmp(suffix, "cpp"))
                fileType = "c++";
            break;
#endif // !LLVM_2_9
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
    else if (outputType == Bitcode)
        return writeBitcode(module, outFileName);
#ifndef LLVM_2_9
    else if (outputType == CXX) {
        extern bool WriteCXXFile(llvm::Module *module, const char *fn, 
                                 int vectorWidth, const char *includeName);
        return WriteCXXFile(module, outFileName, g->target.vectorWidth,
                            includeFileName);
    }
#endif // !LLVM_2_9
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
    llvm::TargetMachine *targetMachine = g->target.GetTargetMachine();
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
    unsigned int flags = binary ? llvm::raw_fd_ostream::F_Binary : 0;

    std::string error;
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
        fprintf(stderr, "Fatal error adding passes to emit object file!");
        exit(1);
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
    Assert(structToNode.find(structType) != structToNode.end());
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
    Assert(sortedTypes.size() == structTypes.size());

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
            Assert(e->constValue != NULL);
            unsigned int enumValue;
            int count = e->constValue->AsUInt32(&enumValue);
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
        if (!vt->IsUniformType())
            // Varying stuff shouldn't be visibile to / used by the
            // application, so at least make it not simple to access it by
            // not declaring the type here...
            continue;

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
    const ArrayType *arrayType = dynamic_cast<const ArrayType *>(type);
    const StructType *structType = dynamic_cast<const StructType *>(type);

    if (dynamic_cast<const ReferenceType *>(type) != NULL)
        lGetExportedTypes(type->GetReferenceTarget(), exportedStructTypes, 
                          exportedEnumTypes, exportedVectorTypes);
    else if (dynamic_cast<const PointerType *>(type) != NULL)
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
    else if (dynamic_cast<const EnumType *>(type) != NULL)
        lAddTypeIfNew(type, exportedEnumTypes);
    else if (dynamic_cast<const VectorType *>(type) != NULL)
        lAddTypeIfNew(type, exportedVectorTypes);
    else
        Assert(dynamic_cast<const AtomicType *>(type) != NULL);
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
        for (int j = 0; j < ftype->GetNumParameters(); ++j)
            lGetExportedTypes(ftype->GetParameterType(j), exportedStructTypes,
                              exportedEnumTypes, exportedVectorTypes);
    }
}


static void
lPrintFunctionDeclarations(FILE *file, const std::vector<Symbol *> &funcs) {
    fprintf(file, "#ifdef __cplusplus\nextern \"C\" {\n#endif // __cplusplus\n");
    for (unsigned int i = 0; i < funcs.size(); ++i) {
        const FunctionType *ftype = dynamic_cast<const FunctionType *>(funcs[i]->type);
        Assert(ftype);
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
    Assert(ft);
    return ft->isExported;
}


static bool
lIsExternC(const Symbol *sym) {
    const FunctionType *ft = dynamic_cast<const FunctionType *>(sym->type);
    Assert(ft);
    return ft->isExternC;
}


static bool
lIsExternGlobal(const Symbol *sym) {
    return sym->storageClass == SC_EXTERN || sym->storageClass == SC_EXTERN_C;
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
    std::vector<Symbol *> externGlobals;
    symbolTable->GetMatchingVariables(lIsExternGlobal, &externGlobals);
    for (unsigned int i = 0; i < externGlobals.size(); ++i)
        lGetExportedTypes(externGlobals[i]->type, &exportedStructTypes,
                          &exportedEnumTypes, &exportedVectorTypes);

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
    fprintf(f, "\n#ifdef __cplusplus\n}\n#endif // __cplusplus\n");

    // and only now emit externs for globals, outside of the ispc namespace
    if (externGlobals.size() > 0) {
        fprintf(f, "\n");
        fprintf(f, "///////////////////////////////////////////////////////////////////////////\n");
        fprintf(f, "// Globals declared \"extern\" from ispc code\n");
        fprintf(f, "///////////////////////////////////////////////////////////////////////////\n");
        lPrintExternGlobals(f, externGlobals);
    }

    // end guard
    fprintf(f, "\n#endif // %s\n", guard.c_str());

    fclose(f);
    return true;
}


void
Module::execPreprocessor(const char* infilename, llvm::raw_string_ostream* ostream) const
{
    clang::CompilerInstance inst;
    inst.createFileManager();

    llvm::raw_fd_ostream stderrRaw(2, false);

#if defined(LLVM_3_0) || defined(LLVM_3_0svn) || defined(LLVM_3_1svn)
    clang::TextDiagnosticPrinter *diagPrinter =
        new clang::TextDiagnosticPrinter(stderrRaw, clang::DiagnosticOptions());
    llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagIDs(new clang::DiagnosticIDs);
    clang::DiagnosticsEngine *diagEngine = 
        new clang::DiagnosticsEngine(diagIDs, diagPrinter);
    inst.setDiagnostics(diagEngine);
#else
    clang::TextDiagnosticPrinter *diagPrinter = 
        new clang::TextDiagnosticPrinter(stderrRaw, clang::DiagnosticOptions());
    inst.createDiagnostics(0, NULL, diagPrinter);
#endif

    clang::TargetOptions &options = inst.getTargetOpts();
    llvm::Triple triple(module->getTargetTriple());
    if (triple.getTriple().empty()) {
#if defined(LLVM_3_1) || defined(LLVM_3_1svn)
        triple.setTriple(llvm::sys::getDefaultTargetTriple());
#else
        triple.setTriple(llvm::sys::getHostTriple());
#endif
    }
    options.Triple = triple.getTriple();

    clang::TargetInfo *target =
        clang::TargetInfo::CreateTargetInfo(inst.getDiagnostics(), options);

    inst.setTarget(target);
    inst.createSourceManager(inst.getFileManager());
    inst.InitializeSourceManager(infilename);

    // Don't remove comments in the preprocessor, so that we can accurately
    // track the source file position by handling them ourselves.
    inst.getPreprocessorOutputOpts().ShowComments = 1;

    clang::HeaderSearchOptions &headerOpts = inst.getHeaderSearchOpts();
    headerOpts.UseBuiltinIncludes = 0;
#ifndef LLVM_2_9
    headerOpts.UseStandardSystemIncludes = 0;
#endif // !LLVM_2_9
    headerOpts.UseStandardCXXIncludes = 0;
    if (g->debugPrint)
        headerOpts.Verbose = 1;
    for (int i = 0; i < (int)g->includePath.size(); ++i)
        headerOpts.AddPath(g->includePath[i], clang::frontend::Angled,
                           true /* is user supplied */,
                           false /* not a framework */,
                           true /* ignore sys root */);

    clang::PreprocessorOptions &opts = inst.getPreprocessorOpts();

    // Add defs for ISPC and PI
    opts.addMacroDef("ISPC");
    opts.addMacroDef("PI=3.1415926535");

    // Add #define for current compilation target
    char targetMacro[128];
    sprintf(targetMacro, "ISPC_TARGET_%s", g->target.GetISAString());
    char *p = targetMacro;
    while (*p) {
        *p = toupper(*p);
        ++p;
    }
    opts.addMacroDef(targetMacro);

    if (g->target.is32Bit)
        opts.addMacroDef("ISPC_POINTER_SIZE=32");
    else
        opts.addMacroDef("ISPC_POINTER_SIZE=64");

    opts.addMacroDef("ISPC_MAJOR_VERSION=1");
    opts.addMacroDef("ISPC_MINOR_VERSION=1");

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
    inst.createPreprocessor();

    clang::LangOptions langOptions;
    diagPrinter->BeginSourceFile(langOptions, &inst.getPreprocessor());
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
        for (int i = 0; i < Target::NUM_ISAS; ++i)
            func[i] = NULL;
    }
    // The func array is indexed with the Target::ISA enumerant.  Some
    // values may be NULL, indicating that the original function wasn't
    // compiled to the corresponding target ISA.
    llvm::Function *func[Target::NUM_ISAS];
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
        ftv.func[g->target.isa] = syms[i]->exportedFunction;    
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
        LLVM_TYPE_CONST llvm::Type *type = gv->getType()->getElementType();
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
                    Error(rgi.pos, "Mismatch in size/layout of global "
                          "variable \"%s\" with different targets. "
                          "Globals must not include \"varying\" types or arrays "
                          "with size based on programCount when compiling to "
                          "targets with differing vector widths.",
                          gv->getName().str().c_str());
            }
        }
    }
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
    LLVM_TYPE_CONST llvm::FunctionType *ftype = NULL;

    for (int i = 0; i < Target::NUM_ISAS; ++i) {
        if (funcs.func[i] == NULL) {
            targetFuncs[i] = NULL;
            continue;
        }

        // Grab the type of the function as well.
        if (ftype != NULL)
            Assert(ftype == funcs.func[i]->getFunctionType());
        else
            ftype = funcs.func[i]->getFunctionType();

        targetFuncs[i] = 
            llvm::Function::Create(ftype, llvm::GlobalValue::ExternalLinkage, 
                                   funcs.func[i]->getName(), module);
    }

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
        for (; argIter != dispatchFunc->arg_end(); ++argIter)
            args.push_back(argIter);
        if (voidReturn) {
#if defined(LLVM_3_0) || defined(LLVM_3_0svn) || defined(LLVM_3_1svn)
            llvm::CallInst::Create(targetFuncs[i], args, "", callBBlock);
#else
            llvm::CallInst::Create(targetFuncs[i], args.begin(), args.end(),
                                   "", callBBlock);
#endif
            llvm::ReturnInst::Create(*g->ctx, callBBlock);
        }
        else {
#if defined(LLVM_3_0) || defined(LLVM_3_0svn) || defined(LLVM_3_1svn)
            llvm::Value *retValue = 
                llvm::CallInst::Create(targetFuncs[i], args, "ret_value", 
                                       callBBlock);
#else
            llvm::Value *retValue = 
                llvm::CallInst::Create(targetFuncs[i], args.begin(), args.end(),
                                       "ret_value", callBBlock);
#endif
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
Module::CompileAndOutput(const char *srcFile, const char *arch, const char *cpu, 
                         const char *target, bool generatePIC, OutputType outputType, 
                         const char *outFileName, const char *headerFileName,
                         const char *includeFileName) {
    if (target == NULL || strchr(target, ',') == NULL) {
        // We're only compiling to a single target
        if (!Target::GetTarget(arch, cpu, target, generatePIC, &g->target))
            return 1;

        m = new Module(srcFile);
        if (m->CompileFile() == 0) {
            if (outFileName != NULL)
                if (!m->writeOutput(outputType, outFileName, includeFileName))
                    return 1;
            if (headerFileName != NULL)
                if (!m->writeOutput(Module::Header, headerFileName))
                    return 1;
        }
        int errorCount = m->errorCount;
        delete m;
        m = NULL;

        return errorCount > 0;
    }
    else {
#ifndef LLVM_2_9
        if (outputType == CXX) {
            Error(SourcePos(), "Illegal to specify more then one target when "
                  "compiling C++ output.");
            return 1;
        }
#endif // !LLVM_2_9

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
        for (unsigned int i = 0; i < targets.size(); ++i) {
            if (!Target::GetTarget(arch, cpu, targets[i].c_str(), generatePIC, 
                                   &g->target))
                return 1;

            // Issue an error if we've already compiled to a variant of
            // this target ISA.  (It doesn't make sense to compile to both
            // avx and avx-x2, for example.)
            if (targetMachines[g->target.isa] != NULL) {
                Error(SourcePos(), "Can't compile to multiple variants of %s "
                      "target!\n", g->target.GetISAString());
                return 1;
            }
            targetMachines[g->target.isa] = g->target.GetTargetMachine();

            m = new Module(srcFile);
            if (m->CompileFile() == 0) {
                // Grab pointers to the exported functions from the module we
                // just compiled, for use in generating the dispatch function
                // later.
                lGetExportedFunctions(m->symbolTable, exportedFunctions);

                lExtractAndRewriteGlobals(m->module, &globals[i]);

                if (outFileName != NULL) {
                    const char *isaName = g->target.GetISAString();
                    std::string targetOutFileName = 
                        lGetTargetFileName(outFileName, isaName);
                    if (!m->writeOutput(outputType, targetOutFileName.c_str()))
                        return 1;
                }
            }
            errorCount += m->errorCount;

            // Only write the generate header file, if desired, the first
            // time through the loop here.
            if (i == 0 && headerFileName != NULL)
                if (!m->writeOutput(Module::Header, headerFileName))
                    return 1;

            // Important: Don't delete the llvm::Module *m here; we need to
            // keep it around so the llvm::Functions *s stay valid for when
            // we generate the dispatch module's functions...
        }

        llvm::Module *dispatchModule = 
            lCreateDispatchModule(exportedFunctions);

        lAddExtractedGlobals(dispatchModule, globals);

        // Find the first non-NULL target machine from the targets we
        // compiled to above.  We'll use this as the target machine for
        // compiling the dispatch module--this is safe in that it is the
        // least-common-denominator of all of the targets we compiled to.
        llvm::TargetMachine *firstTargetMachine = targetMachines[0];
        int i = 1;
        while (i < Target::NUM_ISAS && firstTargetMachine == NULL)
            firstTargetMachine = targetMachines[i++];
        Assert(firstTargetMachine != NULL);

        if (outFileName != NULL) {
            if (outputType == Bitcode)
                writeBitcode(dispatchModule, outFileName);
            else
                writeObjectFileOrAssembly(firstTargetMachine, dispatchModule,
                                          outputType, outFileName);
        }
        
        return errorCount > 0;
    }
}
