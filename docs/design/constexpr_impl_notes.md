=============================
Constexpr implementation notes
=============================

This file captures issues discovered while prototyping constexpr that were not
fully accounted for in the initial plan, plus adjustments to the design and
implementation approach.

Initial findings
----------------

1. The builtins IR in `builtins/util.m4` and `builtins/util-xe.m4` use
   `__is_compile_time_constant_*` to branch between a constant-specialized
   path and a generic path (e.g., `shuffle2`, `rotate`). This effectively
   emulates constexpr by relying on late IR constant folding. If constexpr
   is implemented as an earlier front-end evaluation, we need to decide
   whether to keep these IR-level checks (as a backend optimization) or
   replace them with constexpr-only semantics in the stdlib. This is both
   a design and layering consideration.

2. Some LLVM intrinsics require immediate operands (immarg), and the stdlib
   uses literal constants or runtime switches to satisfy this requirement:
   `@llvm.prefetch`/`@llvm.masked.gather`/`@llvm.masked.scatter` in
   `builtins/generic.ispc` use literal `*l` constants, while
   `convert_scale_to_const` in `builtins/util.m4` emits a `switch` to
   select an immarg scale at runtime. With constexpr, we must ensure that
   constexpr uniform values become true compile-time constants in IR (not
   loads from storage), otherwise immarg constraints will still be
   violated. This suggests that constexpr evaluation must happen early
   enough to emit literal constants, and/or we may need explicit front-end
   checks for "immediate-required" params.

3. Varying constexpr constants are supported in v1, but constant contexts
   often require uniform values (array sizes, vector lengths, template
   args). We need explicit diagnostics for "constexpr evaluates to varying"
   when a uniform constant is required. This requirement is already implied
   but should be implemented and tested explicitly.

4. The current constant expression machinery (`Expr::GetConstant`) produces
   `llvm::Constant` but does not model side effects or statement execution.
   For constexpr functions (with loops/branches), we need a separate
   evaluator and a definition-time validator. This requires a clear
   boundary between "compile-time constant expression" and
   "constexpr-evaluable function body" in the front-end.

5. Template non-type arguments currently accept only literals/enum/template
   params, not constant expressions. Extending this to constexpr requires
   changes in both parsing/type-checking and template argument mangling, to
   ensure constexpr-evaluated values are used rather than raw expressions.

Additional prototype notes
--------------------------

1. `constexpr` evaluation relies on function bodies being available at the
   time constant expressions are parsed. Deferral now handles forward usage
   for global initializers and default parameter values, but array sizes,
   vector lengths, and template arguments still require definitions to be
   visible before use.

2. Allowing full `constant_expression` inside `<...>` (vector sizes and
   non-type template args) is ambiguous in the LALR grammar: `T<WIDTH> x`
   can be parsed as `T < WIDTH > x`. The current prototype introduces
   `constant_expression_no_gt` in those contexts to prevent `>`/`>=`
   operators from consuming the closing `>` token. This is a pragmatic
   restriction; a cleaner long-term fix likely needs a dedicated
   template-arg parser or lexer context to disambiguate `<`/`>`.

3. The constexpr evaluator handles control flow only for uniform
   conditions; validator now enforces uniform conditions for `if`,
   `loop`, and `switch` in constexpr functions. Varying control flow is
   rejected at definition time in v1.

4. Pointer constexpr support initially only round-tripped null pointers;
   it now supports non-null addresses of globals/functions, pointer-to-bool
   conversion, and pointer +/- integer arithmetic. Pointer dereference and
   pointer-pointer arithmetic remain disallowed.

5. Target-dependent-constant metadata is now propagated through constexpr
   evaluation and deferred resolution so multi-target warnings/errors match
   constant folding behavior.

6. Constexpr evaluation for `ExprList` now covers varying atomic/enum lists,
   short vectors, and aggregate (array/struct) initializer lists, including
   nested lists in constexpr function locals.

7. Local variables inside constexpr functions currently must have
   initializers; uninitialized locals are rejected during constexpr
   validation to match the evaluator's limitations.

8. `programIndex`/`programCount` in `stdlib/include/core.isph` and math library
   selectors in `stdlib/include/target.isph` were migrated to
   `static constexpr` once aggregate/short-vector constexpr evaluation landed;
   no special-case remains.

9. Immarg enforcement for LLVM intrinsics ended up requiring parameter-level
   metadata in the front-end. The prototype attaches immarg flags to
   `FunctionType` when creating intrinsic symbols, and `FunctionCallExpr`
   checks immarg arguments with `ConstexprEvaluate`. This works for direct
   intrinsic calls but is not currently propagated through function pointer
   types or non-intrinsic builtins. A from-scratch design should consider
   a dedicated intrinsic metadata table or explicit parameter attributes.

10. Immarg validation alone was insufficient: LLVM requires literal
    immediates in IR. The prototype now rewrites immarg arguments to
    `ConstExpr` during type checking after `ConstexprEvaluate`, otherwise
    constexpr function calls still lower to runtime calls and trip the LLVM
    verifier (`immarg operand has non-immediate parameter`).

11. Const local variables were not visible to constexpr evaluation during
    immarg checks because `constValue` was set after type checking. The
    prototype now seeds `constValue` for `const` locals during declaration
    type checking so immarg validation can see them. This should be part of
    a from-scratch plan for constexpr evaluation ordering.

12. Intrinsic signatures are picky about varying vs. uniform types; e.g.,
    `llvm.x86.sse41.round.ps` expects `varying float` (vector width), not
    `uniform float<4>`. Tests and examples should mirror intrinsic types to
    avoid false negatives.

13. Error output is word-wrapped by `PrintWithWordBreaks`, so tests that
    check constexpr/immarg diagnostics must allow line breaks in messages.

14. `builtins/generic.ispc` now uses `constexpr` immarg constants
    (`kImmArgZero/One/Two/Three`) in prefetch/masked load/store/gather/scatter
    instead of `*l` literal hacks. Similar IR-level switches (e.g.
    `convert_scale_to_const` in `builtins/util.m4`) remain as future cleanup.

15. Adding short-vector/aggregate constexpr support required extending
    `ConstExpr` to represent collection types (vectors/arrays/structs). This
    had a knock-on effect: const array symbols now optimize to
    `ConstSymbolExpr`, and `AddressOfExpr::GetConstant` previously only
    accepted `SymbolExpr` bases. That broke constant evaluation for global
    pointer-array initializers like `b + 3` (tests/lit-tests/1596.ispc). The
    fix was to accept `ConstSymbolExpr` in the constant-address path. A
    from-scratch design should account for this interaction between const
    folding and address-of constant generation.

16. Constexpr evaluation now performs element-wise folding for vector
   expressions inside constexpr functions, while general optimizer constant
   folding still skips aggregates to avoid calling `ConstExpr::GetValues`
   on them. A from-scratch design should unify the constant representation
   so vector/aggregate folding does not need separate constexpr-only logic.

17. Pointer constants exposed a mismatch between `ConstExpr` representation
    and utility helpers: `ConstExpr::GetValues` historically asserted
    `!isPointer`, but constexpr evaluation can legitimately flow pointer
    constants into boolean contexts (e.g. `if (p)`). The prototype fixes
    this by treating pointer constants as boolean values for
    `GetValues(bool*)`. A from-scratch design should unify pointer constant
    representation and avoid implicit numeric conversions.

18. Mental fuzzing found that uniform/varying qualifiers still matter for
    aggregate locals in constexpr functions: `Pair p = {}` defaults to
    varying and cannot be returned from a `constexpr uniform Pair` function.
    This is expected ISPC behavior but worth calling out in docs/examples to
    avoid surprising users when constexpr enters the picture.


Planned adjustments
-------------------

- Keep `__is_compile_time_constant_*` in IR for backend optimization, but
  document and eventually replace stdlib-level "emulation" cases with
  constexpr-friendly APIs that surface immediate parameters directly.

- Replace scale-switch macros (`convert_scale_to_const*`) with front-end
  constexpr immarg enforcement where possible, keeping the switch only for
  truly dynamic scales.

- Consider extending deferred constexpr evaluation to array sizes, vector
  lengths, and template arguments; these contexts currently require
  definitions to be visible before use. A dedicated template-arg parser may be
  needed to relax the `constant_expression_no_gt` workaround.

Systematic test ideas
---------------------

- Differential testing: run constexpr functions both at compile time and
  runtime for a fixed input set, comparing results in a single test file.
  This can be automated by generating a constexpr value and a runtime value
  from the same function and `assert`ing equality in a task/kernel.
- Grammar fuzzing: generate random constant expressions (bounded operators,
  casts, and small aggregates) and ensure the front end either folds them
  identically to runtime evaluation or rejects them with deterministic errors.
- Target-matrix coverage: compile the same constexpr tests across multiple
  targets (SSE/AVX/AVX512/Xe) to ensure target-dependent constant metadata is
  preserved and diagnostics remain stable.
- Negative test corpus: maintain a focused suite of disallowed constructs
  (pointer deref, varying control flow, non-constexpr calls) with strict
  diagnostics to prevent regressions in validation.

Stdlib/builtins hacks inventory
-------------------------------

- Immarg literal constants in `builtins/generic.ispc`: prefetch and masked
  load/store/gather/scatter use immediate operands (prefetch read/write and
  locality, masked load/store/gather/scatter alignment). Historically these
  were hard-coded as `0l/1l/2l/3l` to avoid casts and satisfy LLVM immarg
  requirements. They are now replaced with `constexpr uniform int`
  constants (`kImmArgZero/One/Two/Three`), keeping literal immediates in IR
  while making the intent explicit and constexpr-friendly. The front end also
  rewrites immarg arguments to `ConstExpr` during type checking, so constexpr
  values survive lowering without relying on ad-hoc literal syntax.
- Scale switch macros in `builtins/util.m4` and `builtins/target-avx512-utils.ll`:
  `convert_scale_to_const`, `convert_scale_to_const_gather`, and
  `convert_scale_to_const_scatter` generate `switch`es on `scale` (1/2/4/8)
  and dispatch to the target gather/scatter intrinsics with immediate scale.
  These are used by `__gather_base_offsets*`/`__scatter_base_offsets*` and by
  the pseudo-gather/scatter lowering pipeline. They are a runtime fallback
  for dynamic scales and imply `unreachable` for other values; they also
  inflate code size and obscure the constant requirement. A constexpr-aware
  front end could select the intrinsic directly for constant scales (or error
  on non-constant scales), leaving the switch only for truly dynamic cases.
- IR-level constant probes in `builtins/util.m4`, `builtins/util-xe.m4`, and
  `builtins/target-*.ll`: `__is_compile_time_constant_{uniform_int32,
  varying_int32,mask}` gates fast paths in shuffle/rotate and mask-known
  code (e.g., full/empty mask checks). These are declared in
  `stdlib/include/core.isph` and are meant for LLVM to fold away when
  inputs are constant. This is optimization-only and can coexist with
  constexpr; a future constexpr-first path could bypass these probes in the
  front end and rely on constexpr evaluation to select the fast path.
- Stdlib constants now expressed as constexpr: `programCount` and
  `programIndex` in `stdlib/include/core.isph`, plus math library selectors
  in `stdlib/include/target.isph`, are now `static constexpr` to align with
  the new constexpr semantics instead of relying on `static const`. This
  removes the last stdlib-level "const-but-not-constexpr" workaround.
