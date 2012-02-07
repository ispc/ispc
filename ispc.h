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

#if !defined(LLVM_2_9) && !defined(LLVM_3_0) && !defined(LLVM_3_0svn) && !defined(LLVM_3_1svn)
#error "Only LLVM 2.9, 3.0, and the 3.1 development branch are supported"
#endif

#if defined(_WIN32) || defined(_WIN64)
#define ISPC_IS_WINDOWS
#elif defined(__linux__)
#define ISPC_IS_LINUX
#elif defined(__APPLE__)
#define ISPC_IS_APPLE
#endif

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string>

#define Assert(expr)                                            \
    ((void)((expr) ? 0 : __Assert (#expr, __FILE__, __LINE__)))
#define __Assert(expr, file, line)                                      \
    ((void)fprintf(stderr, "%s:%u: Assertion failed: \"%s\"\n"          \
                   "***\n*** Please file a bug report at "              \
                   "https://github.com/ispc/ispc/issues\n*** (Including as much " \
                   "information as you can about how to reproduce this error).\n" \
                   "*** You have apparently encountered a bug in the compiler that " \
                   "we'd like to fix!\n***\n", file, line, expr), abort(), 0)

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
    class Target;
    class TargetMachine;
    class Type;
    class Value;
}

// llvm::Type *s are no longer const in llvm 3.0
#if defined(LLVM_3_0) || defined(LLVM_3_0svn) || defined(LLVM_3_1svn)
#define LLVM_TYPE_CONST
#else
#define LLVM_TYPE_CONST const
#endif

class ArrayType;
class AST;
class ASTNode;
class AtomicType;
class FunctionEmitContext;
class Expr;
class ExprList;
class Function;
class FunctionType;
class Module;
class Stmt;
class Symbol;
class SymbolTable;
class Type;
struct VariableDeclaration;

/** @brief Representation of a range of positions in a source file.

    This class represents a range of characters in a source file
    (e.g. those that span a token's definition), from starting line and
    column to ending line and column.  (These values are tracked by the
    lexing code).  Both lines and columns are counted starting from one.
 */
struct SourcePos {
    SourcePos(const char *n = NULL, int fl = 0, int fc = 0,
              int ll = 0, int lc = 0);

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

/** Returns a SourcePos that encompasses the extent of both of the given
    extents. */
SourcePos Union(const SourcePos &p1, const SourcePos &p2);


/** @brief Structure that defines a compilation target 

    This structure defines a compilation target for the ispc compiler.
*/
struct Target {
    /** Initializes the given Target pointer for a target of the given
        name, if the name is a known target.  Returns true if the
        target was initialized and false if the name is unknown. */
    static bool GetTarget(const char *arch, const char *cpu, const char *isa,
                          bool pic, Target *);

    /** Returns a comma-delimited string giving the names of the currently
        supported target ISAs. */
    static const char *SupportedTargetISAs();

    /** Returns a comma-delimited string giving the names of the currently
        supported target CPUs. */
    static const char *SupportedTargetCPUs();

    /** Returns a comma-delimited string giving the names of the currently
        supported target architectures. */
    static const char *SupportedTargetArchs();

    /** Returns a triple string specifying the target architecture, vendor,
        and environment. */
    std::string GetTripleString() const;

    /** Returns the LLVM TargetMachine object corresponding to this
        target. */
    llvm::TargetMachine *GetTargetMachine() const;
    
    /** Returns a string like "avx" encoding the target. */
    const char *GetISAString() const;

    /** Returns the size of the given type */
    llvm::Value *SizeOf(LLVM_TYPE_CONST llvm::Type *type,
                        llvm::BasicBlock *insertAtEnd);

    /** Given a structure type and an element number in the structure,
        returns a value corresponding to the number of bytes from the start
        of the structure where the element is located. */
    llvm::Value *StructOffset(LLVM_TYPE_CONST llvm::Type *type,
                              int element, llvm::BasicBlock *insertAtEnd);

    /** llvm Target object representing this target. */
    const llvm::Target *target;

    /** Enumerator giving the instruction sets that the compiler can
        target.  These should be ordered from "worse" to "better" in that
        if a processor supports multiple target ISAs, then the most
        flexible/performant of them will apear last in the enumerant.  Note
        also that __best_available_isa() needs to be updated if ISAs are
        added or the enumerant values are reordered.  */
    enum ISA { SSE2, SSE4, AVX, AVX2, GENERIC, NUM_ISAS };

    /** Instruction set being compiled to. */
    ISA isa;

    /** Target system architecture.  (e.g. "x86-64", "x86"). */
    std::string arch;

    /** Is the target architecture 32 or 64 bit */
    bool is32Bit;

    /** Target CPU. (e.g. "corei7", "corei7-avx", ..) */
    std::string cpu;

    /** Target-specific attributes to pass along to the LLVM backend */
    std::string attributes;

    /** Native vector width of the vector instruction set.  Note that this
        value is directly derived from the ISA Being used (e.g. it's 4 for
        SSE, 8 for AVX, etc.) */
    int nativeVectorWidth;

    /** Actual vector width currently being compiled to.  This may be an
        integer multiple of the native vector width, for example if we're
        "doubling up" and compiling 8-wide on a 4-wide SSE system. */
    int vectorWidth;

    /** Indicates whether position independent code should be generated. */
    bool generatePIC;

    /** Is there overhead associated with masking on the target
        architecture; e.g. there is on SSE, due to extra blends and the
        like, but there isn't with an ISA that supports masking
        natively. */
    bool maskingIsFree;

    /** Is it safe to run code with the mask all if: e.g. on SSE, the fast
        gather trick assumes that at least one program instance is running
        (so that it can safely assume that the array base pointer is
        valid). */
    bool allOffMaskIsSafe;

    /** How many bits are used to store each element of the mask: e.g. this
        is 32 on SSE/AVX, since that matches the HW better, but it's 1 for
        the generic target. */
    int maskBitCount;
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

    /** Indicates whether an vector load should be issued for masked loads
        on platforms that don't have a native masked vector load.  (This may
        lead to accessing memory up to programCount-1 elements past the end of
        arrays, so is unsafe in general.) */
    bool fastMaskedVload;

    /** Indicates when loops should be unrolled (when doing so seems like
        it will make sense. */
    bool unrollLoops;

    /** Indicates if addressing math will be done with 32-bit math, even on
        64-bit systems.  (This is generally noticably more efficient,
        though at the cost of addressing >2GB).
     */ 
    bool force32BitAddressing;

    /** Indicates whether Assert() statements should be ignored (for
        performance in the generated code). */
    bool disableAsserts;

    /** If enabled, disables the various optimizations that kick in when
        the execution mask can be determined to be "all on" at compile
        time. */
    bool disableMaskAllOnOptimizations;

    /** If enabled, the various __pseudo* memory ops (gather/scatter,
        masked load/store) are left in their __pseudo* form, for better
        understanding of the structure of generated code when reading
        it. */
    bool disableHandlePseudoMemoryOps;

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

    /** Indicates whether warnings should be issued as errors. */
    bool warningsAsErrors;

    /** Indicates whether line wrapping of error messages to the terminal
        width should be disabled. */
    bool disableLineWrap;

    /** Indicates whether additional warnings should be issued about
        possible performance pitfalls. */
    bool emitPerfWarnings;

    /** Indicates whether all printed output should be surpressed. */
    bool quiet;

    /** Indicates whether calls should be emitted in the program to an
        externally-defined program instrumentation function. (See the
        "Instrumenting your ispc programs" section in the user's
        manual.) */
    bool emitInstrumentation; 

    /** Indicates whether ispc should generate debugging symbols for the
        program in its output. */
    bool generateDebuggingSymbols;
   
    /** If true, function names are mangled by appending the target ISA and
        vector width to them. */
    bool mangleFunctionsWithTarget;

    /** If enabled, the lexer will randomly replace some tokens returned
        with other tokens, in order to test error condition handling in the
        compiler. */
    bool enableFuzzTest;

    /** Seed for random number generator used for fuzz testing. */
    int fuzzTestSeed;

    /** Global LLVMContext object */
    llvm::LLVMContext *ctx;

    /** Current working directory when the ispc compiler starts
        execution. */
    char currentDirectory[1024];

    /** Arguments to pass along to the C pre-processor, if it is run on the
        program before compilation. */
    std::vector<std::string> cppArgs;

    /** Additional user-provided directories to search when processing
        #include directives in the preprocessor. */
    std::vector<std::string> includePath;
};

enum {
    COST_ASSIGN = 1,
    COST_COHERENT_BREAK_CONTINE = 4,
    COST_COMPLEX_ARITH_OP = 4,
    COST_DELETE = 32,
    COST_DEREF = 4,
    COST_FUNCALL = 4,
    COST_FUNPTR_UNIFORM = 12,
    COST_FUNPTR_VARYING = 24,
    COST_GATHER = 8,
    COST_GOTO = 4,
    COST_LOAD = 2,
    COST_NEW = 32,
    COST_REGULAR_BREAK_CONTINUE = 2,
    COST_RETURN = 4,
    COST_SELECT = 4,
    COST_SIMPLE_ARITH_LOGIC_OP = 1,
    COST_SYNC = 32,
    COST_TASK_LAUNCH = 32,
    COST_TYPECAST_COMPLEX = 4,
    COST_TYPECAST_SIMPLE = 1,
    COST_UNIFORM_IF = 2,
    COST_VARYING_IF = 3,
    COST_UNIFORM_LOOP = 4,
    COST_VARYING_LOOP = 6,
    COST_UNIFORM_SWITCH = 4,
    COST_VARYING_SWITCH = 12,
    COST_ASSERT = 8,

    CHECK_MASK_AT_FUNCTION_START_COST = 16,
    PREDICATE_SAFE_IF_STATEMENT_COST = 6,
};

extern Globals *g;
extern Module *m;

#endif // ISPC_H
