# Implementing constexpr in ISPC: design, hacks removed, and tests

## Motivation

ISPC has long relied on a mix of constant folding and stdlib/builtins tricks to
feed compile-time constants to LLVM intrinsics. This work adds a first-class
`constexpr` feature, similar in spirit to modern C++, so users can express
compile-time computation directly and the front end can enforce it.

Key goals for v1:

- `constexpr` variables and functions usable in constant-expression contexts
  (array sizes, short-vector lengths, switch cases, enum values, template args,
  default parameter values).
- C++-style “constexpr-suitable” validation at function definition time.
- Support for both uniform and varying constexpr values.
- Aggregate support (arrays, structs) and short vectors.
- Immediate-operand (immarg) friendly lowering for LLVM intrinsics.

## Front-end design decisions

1. **C++-style validation at definition time**
   `constexpr` functions are validated as they are defined. The validator checks
   that only constexpr-safe statements and expressions are used. This keeps the
   language predictable and errors early.

2. **Uniform control flow**
   V1 requires uniform conditions in `if`, `switch`, `for`, `while`, and `do` in
   constexpr functions. This simplifies evaluation and mirrors constant
   evaluation expectations.

3. **Aggregate support with element-wise evaluation**
   `ConstExpr` now represents vectors, arrays, and structs as element lists, and
   constexpr evaluation folds aggregates element-wise. This is what enables
   `constexpr` arrays/structs and local aggregate initializers inside constexpr
   functions.

4. **Pointer constants with strict limits**
   Pointer support is intentionally narrow: constexpr may materialize null or
   addresses of globals/functions, allow pointer comparisons, pointer-to-bool,
   and pointer +/- integer arithmetic. Pointer dereference (and pointer-pointer
   arithmetic) remains disallowed.

5. **Deferral for forward usage**
   Global initializers and default parameter values can reference constexpr
   functions defined later in the file. The front end defers evaluation until
   after parsing and then resolves any remaining constexpr work. Array sizes,
   vector lengths, and template args are still evaluated immediately and must
   see definitions before use.

6. **Immediate-operand enforcement for intrinsics**
   LLVM requires certain intrinsic operands to be immediates. The front end now
   marks immarg parameters and rewrites constexpr arguments to literal
   `ConstExpr`s during type checking, so LLVM sees real immediate operands in IR.

## Constexpr evaluation pipeline

The implementation is layered end-to-end:

- **Lexer/Parser**: new `constexpr` keyword and grammar support for constexpr
  declarations and constexpr function definitions.
- **AST/Types**: `constexpr` flag on symbols and function types. Function
  parameter defaults can store constexpr expressions.
- **Validator**: definition-time check for constexpr-suitable bodies.
- **Evaluator**: an interpreter-style evaluator for constexpr functions,
  supporting control flow and local variables, plus aggregate folding and
  pointer constants.
- **Deferral**: postponed constexpr evaluation for globals/default params, with
  resolution prior to IR generation.

## Hacks removed (and what remains)

### Removed or replaced

- **`*l` immarg literal hacks** in `builtins/generic.ispc` were replaced with
  `constexpr uniform int` constants (`kImmArgZero/One/Two/Three`). This keeps
  immediates literal in IR while making the intent explicit.
- **Stdlib constants** (`programIndex`, `programCount`, and target selector
  values) were migrated from `static const` to `static constexpr` now that
  constexpr aggregates and vector initializers are supported.
- **Compiler-side immarg checks** no longer rely on fragile constant folding; the
  front end now enforces immarg parameters and rewrites constexpr arguments to
  immediate constants in IR.

### Still present (by design or future work)

- **`convert_scale_to_const*` switch macros** in `builtins/util.m4` and
  `builtins/target-avx512-utils.ll` remain as runtime fallbacks for dynamic
  gather/scatter scales. The front end can now enforce constant scales, but
  there is still an opportunity to bypass these switches for constexpr scales.
- **`__is_compile_time_constant_*` probes** in `builtins/util*.m4` and
  `builtins/target-*.ll` remain for backend optimization (shuffle/rotate and
  mask-known fast paths). These are optimization-only and can coexist with
  constexpr; a constexpr-first path could skip them in the front end.

## Test coverage

The new behavior is covered with lit tests across the feature surface:

- Basic constexpr variable folding and constexpr function calls.
- Varying constexpr lane-wise evaluation.
- Control flow validation (if/switch/loops) and error diagnostics.
- Non-type template args using constexpr expressions.
- Aggregate constexpr values (structs, arrays, short vectors), including
  initializer lists and element access.
- Pointer constexpr support (null and non-null addresses, pointer conditions,
  pointer arithmetic).
- Immarg enforcement for LLVM intrinsics using constexpr values.
- Deferred evaluation for globals and default parameters referencing constexpr
  functions defined later in the file.

These tests intentionally check both positive behavior (IR constants) and
negative behavior (errors for disallowed constructs) to keep the front end
honest.

## What remains

The feature is usable, but there are obvious follow-ups for a “v2”:

- Extend deferred constexpr evaluation to array sizes, vector lengths, and
  template arguments.
- Relax the `constant_expression_no_gt` grammar workaround for template args.
- Evaluate whether aggregate folding should be unified with general optimizer
  constant folding to remove constexpr-only paths.

