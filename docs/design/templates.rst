=================
Templates support
=================

Language design
---------------

Please refer to ``Function Template`` section of ``User Guide``.

Implementation design
---------------------

An AST node for template function is `FunctionTemplate`, which is similar to `Function`, but stores type parameters
(represented by `TemplateTypeParmType``) or non-type parameters (represented by `Symbol`).  The body of 
`FunctionTemplate` contains `TemplateTypeParmType`, which implies that some of type checking may not be done until
after the function is instantiated.  This requires redesigning type checking, which needs to be decoupled from AST creation.

Implementation details
----------------------

Type checking is generally not done for function templates, it's only done the instantiations. Nevertheless type
checking might be invoked for selected expressions, as it might be required to implement GetType() and GetValue()
functions. More disciplined approach would do type checking on function templates, but bail in case type dependent
expressions. The reason to do type checking in this manner is to report those errors that are possible to report without
resolving dependent types.

Template argument deduction mechanism. It's implemented as part of FunctionSymbolExpr logic. When function call is
parsed, it's not known if the call is to a function or a function template, so template argument deduction is part of
general overload resolution process.

Possible variants of template argument deduction rules
------------------------------------------------------

While the rules generally follow C++ rules, the existence of `uniform`, `varying`, and `unbound` types, creates a
certain variability in design space for the language. They basically boil down to tree questions that have binary
answers:

1. whether resolved type parameters might be `unbound` or not;
2. whether variability of type parameter `T` might be overwritten by `uniform` or `varying` specifier or not;
3. whether strict matching is required for template type arguments variability with template parameters variability
   or not.

For the (1) there's a temptation to allow `unbound` types, as this is generally how ISPC type system works. I.e. the
types themselves are either `uniform`, `varying`, or `unbound`, while variables might have only `uniform` or `varying`
types. If the variable is declared with `unbound` type (for example `int i`), the default kicks in that treats `unbound`
as `varying`.

The major drawback of allowing `unbound` types as resolved template type parameters is that it creates the possibility
for the user to unconsciously create two function instances instead of one. Consider the following example:

.. code-block:: cpp
    template <typename T> T foo(T t);

    // Two instances should be created, which is very likely not the intended by the user.
    foo<int>(1);
    foo<varying int>(2);

Another drawback is that existence of `unbound` type pushed towards prioritization of `unbound` version of the type
during template type deduction, while it is preferable to prioritize `uniform` as better performing version (if such
possibility exists).

So the decision is to require template type parameters to resolve only to `uniform` and `varying`, but not `unbound`
types.

For the (2), the decision is to allow overwriting of variability of template type parameter `T` with keywords `uniform`
and `varying` to allow better flexibility and expressiveness of the code. For example:

.. code-block:: cpp
    template <typename T> void foo() {
        uniform T ut; // it's always legal to have "uniform T"
        varying T vt; // and "varying T", regardless of T variability.
    };

Nevertheless, the type deduction rules might be different with respect to `uniform` and `varying` keywords, i.e.
consider the following:

.. code-block:: cpp
    template <typename T> void foo1(T t);
    template <typename T> void foo2(uniform T t);
    template <typename T> void foo3(varying T t);

    uniform int ui;
    varying int vi;
    foo1(ui); // T is uniform
    foo1(vi); // T is varying
    foo2(ui); // not clear if "uniform" key word should "cancel" uniform variability of type T
    foo2(vi); // error: varying type cannot be passed to uniform parameter
    foo3(ui); // not clear...
    foo3(vi); // not clear if "varying" key word should "cancel" varying variability of type T

There might be multiple justifications and considerations on "not clear" cases. The proposal is to treat `uniform` and
`varying` keywords as assumption that T has an opposite variability, so the keywords are specified on purpose. i.e.:

    argument |     T t     | uniform T t | varying T t
    --------------------------------------------------
    uniform  | T = uniform | T = varying | T = uniform
    varying  | T = varying | error       | T = uniform

For the (3), C++ philosophy requires strict type matching, i.e. in C++:

.. code-block:: cpp
    template <typename T> void foo(T t1, T t2);

    int i;
    unsigned int ui;
    foo(i, ui); // error: conflict between resolution to T = int and T = unsigned int

But in ISPC it's frequent case when uniform argument is passed to varying parameter assuming default conversion rules
that cause value broadcast.  Disallowing this case for function templates would cause major inconvenience and would be a
blocker for using function templates in libraries that assumed to be drop-in replacement for versions implemented using
function overloads.

.. code-block:: cpp
    template <typename T> T fma(T t1, T t2, T t3);

    varying float v1, v2;
    varying float fma(v1, v2, 2f); // this should be allowed.

So this the decision is to:
1. allow only `uniform` and `varying` as resolved template type parameters values, but not `unbound`;
2. variability of template type parameter `T` might be overwritten by `uniform` and `varying` keywords: `uniform T` and
   `varying T` are always valid;
3. strict matching of template type parameters and template argument variability is not required, `uniform` argument
   might be passed to `varying` parameter.

Open design issues
------------------

AST is implemented as a collection of `Functions`, which means:
- Any manipulation with AST on module level need to be changed synchronously - i.e. covering `FunctionTemplate` in
  addition to `Functions`.  A better design would be to have a single AST root node (`Module`?) that would contain all
  the functions, global variable declarations, typedefs, and templates;
- Symbol table exists as a separate entity and variable declarations on module level do not have AST node.

Also there are some other issues, which require refactoring and redesign:
- Symbols in symbol table do not have explicit information about their definition scope - if they are global or local.
  This needs to be fixed, as it's required to implement correct template instantiation - global symbols should not be
  duplicated.
- `Symbol` and `TemplateSymbol` need to be unified. The `Symbol` class itself is overloaded and needs redesign to
  clearly distinguish between different types of symbols and their properties.
- Error reporting needs to be dramatically improved for templates, this includes better error messages with detailed
  reasons for rejecting specific template function overloads in the process of template argument deduction and
  information about current template instantiation during instantiation process.
- SFINAE-like mechanism for template instantiations. Some of template instantiations may fail to instantiate and this
  should be ok, as other overloads may be valid.
- Template function declarations should be fully supported. Right now they are parsed, but do not work as a valid
  function templates for the purpose of template argument deduction.
- A function symbol may refer to both functions and function templates, right now the candidate list is built from
  either of them, but not both, this needs to be fixed.

C++ template argument deduction algorithm
-----------------------------------------

The template argument deduction algorithm is described in [temp.deduct], [temp.deduct.call], and [temp.deduct.type]
sections of C++ standard. It's useful to read these sections, but no sane human being can convert it to working
algorithm correctly without extra knowledge. The recommended way to learn more is to read and debug clang's function
`DeduceTemplateArgumentsFromCallArgument()`.

Notable not obvious points:
- In C++, the type of an expression is always adjusted so that it will not have reference type (C++ [expr]p6). I.e. the
  standard describes stripping cv-qualifiers from argument type, but not reference, because it's assumed to be already
  stripped before starting the process.
- `const T &` is a reference type, but not a constant type. Stripping cv-qualifiers from this type has no effect, but
  stripping reference type yields `cosnt T`.

Also it might be useful to check [temp.dep.type] and [temp.dep.expr] sections of C++ standard to get better context of
dependent types and expressions.

