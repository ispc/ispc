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
   configuration/capability placeholders used while building shared stdlib
   bitcode deliberately remain ordinary declarations until target linking.

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

14. `builtins/generic.ispc` now uses named constexpr controls and a prefetch
    template in place of `*l` immediate literal workarounds. The alignment
    operands required before LLVM 22 and the zero-input flags of `ctlz`/`cttz`
    are also constexpr constants. Its doubled vector width is computed from
    `programCount` instead of a preprocessor case table. IR-level switches
    for dynamic gather/scatter scales remain necessary for the existing APIs.

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
    aggregate locals in constexpr functions: `Pair p = {0, 0}` defaults to
    varying and cannot be returned from a `constexpr uniform Pair` function.
    This is expected ISPC behavior but worth calling out in docs/examples to
    avoid surprising users when constexpr enters the picture.

19. Switching enum/array size validation from `Expr::GetConstant()` (LLVM
    `ConstantInt`) to `ConstexprEvaluate()` introduced a subtle regression:
    unsigned 32-bit values with the high bit set (e.g. `0x80000000`,
    `0xffffffff`) were rejected by a signed 32-bit representability check.
    The fix was to validate unsigned values via `uint64_t` -> `uint32_t`
    truncation when the expression type is unsigned, preserving the previous
    behavior while still rejecting >32-bit values.


Planned adjustments
-------------------

- Keep `__is_compile_time_constant_*` in IR for backend optimization. Add
  constexpr-friendly stdlib APIs only where the immediate value is surfaced as
  a template argument or direct intrinsic argument, so the value remains visible
  at the call site instead of becoming an ordinary function parameter.

- Keep scale-switch macros (`convert_scale_to_const*`) for the existing dynamic
  gather/scatter wrappers. With the v1 design, constexpr can support parallel
  const-scale template APIs, but it cannot specialize an ordinary `uniform int`
  wrapper parameter after the call crosses the function boundary.

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
  requirements. These sites now use named constexpr constants; prefetch
  wrappers share an NTTP implementation. The front end rewrites their arguments
  to `ConstExpr` during type checking, before IR generation.
- SSE/SSE2 intrinsics headers (`xmmintrin.isph`, `emmintrin.isph`): immediate
  style APIs retain their existing compatibility forms with ordinary immediate
  arguments and add explicit template overloads for shuffles, lane
  extract/insert, byte shifts, bit shifts, and prefetch hints. This is the
  stdlib pattern that works with v1: a user can write a `constexpr` helper that
  returns the control value and pass it as a non-type template argument, so the
  stdlib implementation sees a compile-time value in the function body.
- Scale switch macros in `builtins/util.m4` and `builtins/target-avx512-utils.ll`:
  `convert_scale_to_const`, `convert_scale_to_const_gather`, and
  `convert_scale_to_const_scatter` generate `switch`es on `scale` (1/2/4/8)
  and dispatch to the target gather/scatter intrinsics with immediate scale.
  These are used by `__gather_base_offsets*`/`__scatter_base_offsets*` and by
  the pseudo-gather/scatter lowering pipeline. They are a runtime fallback
  for dynamic scales and imply `unreachable` for other values; they also
  inflate code size and obscure the constant requirement. The current design
  does not remove these switches for ordinary wrapper calls because the `scale`
  parameter is not constexpr inside the wrapper body. A future const-scale
  template API could select the intrinsic directly for constant scales, leaving
  the existing switch wrappers for dynamic scales.
- IR-level constant probes in `builtins/util.m4`, `builtins/util-xe.m4`, and
  `builtins/target-*.ll`: `__is_compile_time_constant_{uniform_int32,
  varying_int32,mask}` gates fast paths in shuffle/rotate and mask-known
  code (e.g., full/empty mask checks). These are declared in
  `stdlib/include/core.isph` and are meant for LLVM to fold away when
  inputs are constant. This is optimization-only and can coexist with
  constexpr; a future constexpr-first path could bypass these probes in the
  front end and rely on constexpr evaluation to select the fast path.


Library validation and follow-up assessment
------------------------------------------

- AMX template overloads call LLVM intrinsics directly. Static assertions
  validate the full integer value before narrowing it to i8; negative and
  greater-than-seven tile IDs are rejected. CPU instruction features are checked
  when a template is instantiated, so including the header does not require
  every AMX instruction set. Tests disable optimization and the legacy AMX
  lowering pass to check that the new path emits literal operands itself.
- Existing argument-taking AMX functions retain the old builtins and lowering
  pass for source compatibility. Backend constant probes and dynamic
  gather/scatter scale switches remain independently useful.
- The AMX matrix-multiply example, its integer/BF16/FP16 runtime tests, and
  SSE/SSE2 benchmarks use the new template overloads. Separate positive and
  negative tests retain coverage of ordinary-call compatibility.
- The existing `_fast_div3`, `_cbrt_newton_iter`, and `_cbrt_halley_iter` math
  helpers are shared header-defined constexpr functions. Production cube-root
  code and constant evaluation use the same arithmetic. Other math routines
  that call runtime builtins are not implicitly constexpr. A constexpr body
  must be visible to the front end; merely marking a declaration whose body
  exists only in precompiled stdlib bitcode is insufficient.
- Public integer `abs` and floating-point `isnan` overloads now share their
  existing arithmetic bodies with constant evaluation. Their prior overflow
  and floating-point rules are unchanged; no intrinsic interpreter is needed.
- The existing bit-conversion functions use the compiler's bitcast operation
  during constant evaluation. This preserves NaN payloads and signed zero and
  enables shared constexpr implementations of floating `abs`, `signbits`,
  `isinf`, and `isfinite`. Signed `saturating_div` also shares its implementation
  with constant evaluation. Existing public signatures and call syntax remain
  available.
- Migrated standard-library definitions retain their existing replacement
  behavior: an ordinary user declaration can select the bitcode fallback or
  provide a replacement body. Earlier runtime calls keep referring to that
  function. This exception is marked on the library definitions; ordinary
  user functions still require consistent constexpr declarations.
- Derived NTTP expressions such as `callee<N + 1>()` and
  `callee<control(N)>()` are retained until substitution. Constexpr local
  values can also feed NTTPs. These are requirements of composable library
  wrappers and are included in the current implementation, rather than
  deferred to v2.
- Rotation shuffle masks are generated by a constexpr loop that fills an
  aggregate, replacing the width-conditioned initializer. The AMX example
  likewise builds its tile descriptor during constant evaluation while
  preserving the existing exported initialization function. These consumers
  exercise nested aggregate writes without permitting writes through pointers.
- SSE/SSE2 template overloads retain portable implementations. Boundary tests
  cover oversized shifts, sign fill, zero extension, and nonidentity shuffles;
  template and ordinary overloads share constexpr implementations while
  preserving runtime controls and the existing public layout.
- SSE float/double and 32-bit integer constructors and shuffles construct
  aggregates directly and are constexpr without changes to their ordinary call
  forms. Narrow integer and 64-bit constructors, word shuffles, extract/insert,
  packed shifts, and representation casts use typed short-vector bitcasts.
  Runtime operations retain their vector representation instead of scalar
  packing arithmetic. The two identical sRGB coefficient tables are shared as
  a single internal constexpr array.
- Numeric folding uses frontend types for precision, signedness, variability,
  and logical element counts. Bitcast symbols retain the source ISPC signature
  even when LLVM represents a uniform short vector and a varying value with
  the same type. APInt/APFloat implement exact bit operations; LLVM type
  reconstruction does not determine constexpr value types. Target layout is
  consulted for byte order, sizes, and address constants.

- Partially initialized uniform short vectors now zero-fill omitted logical
  elements at runtime, matching constant evaluation and the general initializer
  path. Physical storage padding remains poison. The SSE pack implementation
  uses a fully initialized temporary and does not depend on the previous
  runtime omission.
- Integer min/max/clamp, saturating add/subtract, unsigned saturating division,
  and the existing varying average overloads now have visible constexpr
  definitions (96 overloads). Private helpers keep clamp independent of user
  replacements for public min/max. The public signatures and linked fallbacks
  are preserved. Signed averages round negative odd sums in the documented
  direction on every target; the old SSE2 builtin divided toward zero.
- A further 69 SSE/SSE2 value helpers are constexpr: wrapping and saturating
  integer arithmetic, comparisons, min/max, multiply, averages, SAD, packs,
  narrow unpacks, bitwise operations, moves, and movemasks. Integer operations
  use unsigned arithmetic where wrapping is required. Narrow-target literal
  bounds use signed limit constants rather than unsigned negation.
- Saturating arithmetic uses four typed ISPC aliases of native LLVM intrinsics.
  The aliases preserve frontend signedness and uniform short-vector shape,
  including when logical width equals gang width; existing raw LLVM intrinsic
  mapping is unchanged. The evaluator implements their explicit signed or
  unsigned operation using APInt at the ISPC element width. CPU mask packing
  also has a narrow evaluator; Xe movemask uses a constexpr loop because
  SPIR-V cannot bitcast a boolean vector to an integer.
- A late optimization narrows integer comparison masks together with their
  sign extension when upper lanes are proven poison. This avoids AVX-512
  instructions on padding without changing public layout. SSE2 signed 64-bit
  saturation retains its efficient unsigned-bit formula; the existing late
  scalarization pass handles gang-wide 64-bit clamp pairs on SSE2. Standalone
  min/max retain their faster vector lowering.
- The PPC64 runtime path keeps half/integer bitcasts opaque until after
  optimization. Otherwise LLVM can turn bit tests into unsupported half
  floating-point operations. Required late lowering emits the original bit
  operations, including when optional optimization passes are disabled.
- Immediate tests use `--off-phase=pre:last` to disable optional passes.
  `--off-phase=all` is not a supported spelling and does not provide that test.

The follow-up decisions after library adoption are:

| Candidate | Current decision and reason |
| --- | --- |
| Forward definitions in array sizes, vector lengths, and NTTPs | Keep deferred: converted headers can place definitions before uses. This requires deferred type construction and is separate from dependent NTTP substitution. |
| Dedicated template-argument parser | Keep deferred: parentheses handle relational expressions containing `>`; changing parser/lexer context is not required by the converted APIs. |
| Unified aggregate constant folding | Keep deferred: architectural cleanup, not needed for literal scalar intrinsic operands. |
| Local array/struct element mutation | Included for initialized local aggregates and short vectors, including nested assignment and compound operations. Copies remain independent; pointer/reference writes and partial-mask mutation remain unsupported. |
| Static assertions | Included: `static_assert` validates uniform compile-time conditions at file/block scope and template instantiation, independently of runtime assertions. AMX tile ranges and generic prefetch controls use it; gather-scale tests verify values 1/2/4/8 before intrinsic narrowing. Executing runtime `assert` in constexpr functions remains separate. |
| Constexpr evaluation of pure intrinsics | Equal-width numeric bitcasts, integer saturating add/subtract, and CPU mask packing are included. Typed ISPC aliases preserve frontend shape and native runtime instructions. Further operations remain deferred until a real library consumer establishes their semantics and validation needs; floating arithmetic, rounding and transcendental routines need separate treatment. |
| Varying control flow and masked mutation | Keep deferred: expression folding is supported, but writes requiring a partial lane mask are rejected rather than evaluated with incorrect side effects. |
| Constexpr `void` functions and executing runtime `assert` | Keep deferred: current builders return aggregate values and immediate contracts use `static_assert`. Neither feature is required by the migrated APIs. |
| Public constant-scale gather/scatter API | Optional and outside this migration. Existing varying-index memory access remains the public interface; internal constexpr adoption must preserve its behavior without requiring user source changes. |

The remaining library candidates are integer `saturating_mul`, population and
leading/trailing-zero counts, horizontal reductions, and floating-point
min/max, rounding, reciprocal, square root, and transcendental operations.
SSE floating arithmetic/comparisons/conversions are likewise separate from the
integer and bit-preserving value-helper migration. Each family needs explicit
constant-evaluation semantics and runtime parity/code-generation checks;
marking its existing bitcode declaration constexpr would not provide a body.
Loads, stores, atomics, streaming operations, and processor-state access remain
runtime APIs. Their immediate controls can still use the new mechanism.

The average peephole pass still recognizes widened arithmetic in ordinary user
code and uses target-specific average builtins. Integer min/max builtins also
serve reductions. These optimizations are not obsolete because a library
wrapper became constexpr. Unused saturation wrappers remain installed builtin
declarations and are not removed as part of a compatibility-preserving change.
