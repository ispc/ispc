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

/** @file ispc.h
    @brief Main ispc.header file
*/

#ifndef ISPC_H
#define ISPC_H

#if defined(_WIN32) || defined(_WIN64)
#define ISPC_IS_WINDOWS
#elif defined(__linux__)
#define ISPC_IS_LINUX
#elif defined(__APPLE__)
#define ISPC_IS_APPLE
#endif

#include <assert.h>
#include <stdint.h>
#include <vector>
#include <string>

/** @def ISPC_MAX_NVEC maximum vector size of any of the compliation
    targets.
 */
#define ISPC_MAX_NVEC 16

// Forward declarations of a number of widely-used LLVM types
namespace llvm {
    class BasicBlock;
    class Constant;
    class ConstantValue;
    class DIBuilder;
    class DIDescriptor;
    class DIFile;
    class DIType;
    class Function;
    class FunctionType;
    class LLVMContext;
    class Module;
    class Type;
    class Value;
}

// llvm::Type *s are no longer const in llvm 3.0
#if defined(LLVM_3_0) || defined(LLVM_3_0svn)
#define LLVM_TYPE_CONST
#else
#define LLVM_TYPE_CONST const
#endif

class ArrayType;
class AtomicType;
class DeclSpecs;
class Declaration;
class Declarator;
class FunctionEmitContext;
class Expr;
class ExprList;
class FunctionType;
class GatherBuffer;
class Module;
class Stmt;
class Symbol;
class SymbolTable;
class Type;

/** @brief Representation of a range of positions in a source file.

    This class represents a range of characters in a source file
    (e.g. those that span a token's definition), from starting line and
    column to ending line and column.  (These values are tracked by the
    lexing code).  Both lines and columns are counted starting from one.
 */
struct SourcePos {
    SourcePos(const char *n = NULL, int l = 0, int c = 0);

    const char *name;
    int first_line;
    int first_column;
    int last_line;
    int last_column;

    /** Prints the filename and line/column range to standard output. */
    void Print() const;

    /** Returns a LLVM DIFile object that represents the SourcePos's file */
    llvm::DIFile GetDIFile() const;

    bool operator==(const SourcePos &p2) const;
};


/** @brief Abstract base class for nodes in the abstract syntax tree (AST).

    This class defines a basic interface that all abstract syntax tree
    (AST) nodes must implement.  The base classes for both expressions
    (Expr) and statements (Stmt) inherit from this class.
*/
class ASTNode {
public:
    ASTNode(SourcePos p) : pos(p) { }
    virtual ~ASTNode();

    /** The Optimize() method should perform any appropriate early-stage
        optimizations on the node (e.g. constant folding).  The caller
        should use the returned ASTNode * in place of the original node.
        This method may return NULL if an error is encountered during
        optimization. */
    virtual ASTNode *Optimize() = 0;

    /** Type checking should be performed by the node when this method is
        called.  In the event of an error, a NULL value may be returned.
        As with ASTNode::Optimize(), the caller should store the returned
        pointer in place of the original ASTNode *. */
    virtual ASTNode *TypeCheck() = 0;

    /** All AST nodes must track the file position where they are
        defined. */
    const SourcePos pos;
};

/** @brief Structure that defines a compilation target 

    This structure defines a compilation target for the ispc compiler.
*/
struct Target {
    Target();

    /** Enumerator giving the instruction sets that the compiler can
        target. */
    enum ISA { SSE2, SSE4, AVX };

    /** Instruction set being compiled to. */
    ISA isa;

    /** Target system architecture.  (e.g. "x86-64", "x86"). */
    std::string arch;

    /** Is the target architecture 32 or 64 bit */
    bool is32bit;

    /** Target CPU. (e.g. "corei7", "corei7-avx", ..) */
    std::string cpu;

    /** Native vector width of the vector instruction set.  Note that this
        value is directly derived from the ISA Being used (e.g. it's 4 for
        SSE, 8 for AVX, etc.) */
    int nativeVectorWidth;

    /** Actual vector width currently being compiled to.  This may be an
        integer multiple of the native vector width, for example if we're
        "doubling up" and compiling 8-wide on a 4-wide SSE system. */
    int vectorWidth;
};

/** @brief Structure that collects optimization options

    This structure collects all of the options related to optimization of
    generated code. 
*/
struct Opt {
    Opt();
    
    /** Optimization level.  Currently, the only valid values are 0,
        indicating essentially no optimization, and 1, indicating as much
        optimization as possible. */
    int level;

    /** Indicates whether "fast and loose" numerically unsafe optimizations
        should be performed.  This is false by default. */
    bool fastMath;

    /** On targets that don't have a masked store instruction but do have a
        blending instruction, by default, we simulate masked stores by
        loading the old value, blending, and storing the result.  This can
        potentially be unsafe in multi-threaded code, in that it writes to
        locations that aren't supposed to be written to.  Setting this
        value to true disables this work-around, and instead implements
        masked stores by 'scalarizing' them, so that we iterate over the
        ISIMD lanes and do a scalar write for the ones that are running. */
    bool disableBlendedMaskedStores;

    /** Disables the 'coherent control flow' constructs in the
        language. (e.g. this causes "cif" statements to be demoted to "if"
        statements.)  This is likely only useful for measuring the impact
        of coherent control flow. */
    bool disableCoherentControlFlow;

    /** Disables uniform control flow optimizations (e.g. this changes an
        "if" statement with a uniform condition to have a varying
        condition).  This is likely only useful for measuring the impact of
        uniform control flow. */
    bool disableUniformControlFlow;

    /** Disables the backend optimizations related to gather/scatter
        (e.g. transforming gather from sequential locations to an unaligned
        load, etc.)  This is likely only useful for measuring the impact of
        these optimizations. */
    bool disableGatherScatterOptimizations;

    /** Disables the optimization that demotes masked stores to regular
        stores when the store is happening at the same control flow level
        where the variable was declared.  This is likely only useful for
        measuring the impact of this optimization. */
    bool disableMaskedStoreToStore;

    /** Disables the optimization that detects when the execution mask is
        all on and emits code for gathers and scatters that doesn't loop
        over the SIMD lanes but just does the scalar loads and stores
        directly. */
    bool disableGatherScatterFlattening;

    /** Disables the optimizations that detect when arrays are being
        indexed with 'uniform' values and issue scalar loads/stores rather
        than gathers/scatters.  This is likely only useful for measuring
        the impact of this optimization. */
    bool disableUniformMemoryOptimizations;

    /** Disables optimizations for masked stores: masked stores with the
        mask all on are transformed to regular stores, and masked stores
        with the mask are all off are removed (which in turn can allow
        eliminating additional dead code related to computing the value
        stored).  This is likely only useful for measuring the impact of
        this optimization. */
    bool disableMaskedStoreOptimizations;
};

/** @brief This structure collects together a number of global variables. 

    This structure collects a number of global variables that mostly
    represent parameter settings for this compilation run.  In particular,
    none of these values should change after compilation befins; their
    values are all set during command-line argument processing or very
    early during the compiler's execution, before any files are parsed.
  */
struct Globals {
    Globals();

    /** Optimization option settings */
    Opt opt;
    /** Compilation target information */
    Target target;

    /** There are a number of math libraries that can be used for
        transcendentals and the like during program compilation. */
    enum MathLib { Math_ISPC, Math_ISPCFast, Math_SVML, Math_System };
    MathLib mathLib;

    /** Records whether the ispc standard library should be made available
        to the program during compilations. (Default is true.) */
    bool includeStdlib;

    /** Indicates whether the C pre-processor should be run over the
        program source before compiling it.  (Default is true.) */
    bool runCPP;

    /** When \c true, voluminous debugging output will be printed during
        ispc's execution. */
    bool debugPrint;

    /** Indicates whether all warning messages should be surpressed. */
    bool disableWarnings;

    /** Indicates whether additional warnings should be issued about
        possible performance pitfalls. */
    bool emitPerfWarnings;

    /** Indicates whether calls should be emitted in the program to an
        externally-defined program instrumentation function. (See the
        "Instrumenting your ispc programs" section in the user's
        manual.) */
    bool emitInstrumentation; 

    /** Indicates whether ispc should generate debugging symbols for the
        program in its output. */
    bool generateDebuggingSymbols;

    /** Global LLVMContext object */
    llvm::LLVMContext *ctx;

    /** Current working directory when the ispc compiler starts
        execution. */
    char currentDirectory[1024];

    /** Arguments to pass along to the C pre-processor, if it is run on the
        program before compilation. */
    std::vector<std::string> cppArgs;
};

extern Globals *g;
extern Module *m;

#endif // ISPC_H
