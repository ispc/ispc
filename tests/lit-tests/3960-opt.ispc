// Regression test for https://github.com/ispc/ispc/issues/3960
// The debug types of structs declared in function bodies are described in the
// scope of the functions that use them, so removing a function (as unused, or
// during optimization) leaves the debug types of the surviving functions in
// their own scopes, and inlined functions keep theirs.

// RUN: %{ispc} %s --target=host --nowrap -g -O0 --emit-llvm-text -o %t0.ll
// RUN: FileCheck %s --check-prefixes=CHECK,O0 < %t0.ll
// RUN: FileCheck %s --check-prefix=REMOVED < %t0.ll
// RUN: %{ispc} %s --target=host --nowrap -g -O2 --emit-llvm-text -o %t2.ll
// RUN: FileCheck %s --check-prefixes=CHECK,O2 < %t2.ll
// RUN: FileCheck %s --check-prefixes=REMOVED,REMOVED-O2 < %t2.ll

// Only the float instantiation is used.
template <typename T> noinline uniform int early(T t) {
    struct G {
        int e;
    };
    uniform G ge = {(uniform int)t};
    return ge.e;
}

static uniform int unusedEarly() { return early<uniform int>(1); }

// The int instantiation is removed during optimization.
template <typename T> noinline uniform int late(T t) {
    struct G {
        int l;
    };
    uniform G gl = {(uniform int)t};
    return gl.l;
}

// A local struct used by another function only through its signature, with
// the function declaring it used only for float.  The instantiation over the
// local struct has internal linkage, so it has a side effect to be kept at
// -O2.
uniform int asPtrCalls;
template <typename T> noinline T *uniform asPtr(void *uniform p) {
    asPtrCalls++;
    return (T * uniform) p;
}

template <typename T> noinline uniform int foreign(void *uniform p) {
    struct G {
        int f;
    };
    return asPtr<uniform G>(p)->f;
}

static uniform int unusedForeign(void *uniform p) { return foreign<uniform int>(p); }

// Inlined into its caller.
static uniform int inlined(uniform int n) {
    struct G {
        int i;
    };
    uniform G gi = {n};
    return gi.i;
}

export uniform int entry(uniform float f, void *uniform p) {
    uniform int x = 0;
    uniform int r = early<uniform float>(f) + foreign<uniform float>(p) + inlined((uniform int)f);
    if (x)
        r += late<uniform int>(1);
    return r + late<uniform float>(f);
}

// CHECK-DAG: define {{.*}}@early___unf___unf({{.*}} !dbg ![[EARLY_SP:[0-9]+]] {
// CHECK-DAG: ![[EARLY_SP]] = distinct !DISubprogram(name: "early", linkageName: "early___unf___unf",
// CHECK-DAG: !DILocalVariable(name: "ge", {{.*}}type: ![[GE:[0-9]+]]
// CHECK-DAG: ![[GE]] = !DICompositeType(tag: DW_TAG_structure_type, name: "uniform struct G", scope: ![[EARLY_SP]],

// O2-DAG: define {{.*}}@late___unf___unf({{.*}} !dbg ![[LATE_SP:[0-9]+]] {
// O2-DAG: ![[LATE_SP]] = distinct !DISubprogram(name: "late", linkageName: "late___unf___unf",
// O2-DAG: !DILocalVariable(name: "gl", scope: ![[LATE_BLOCK:[0-9]+]], {{.*}}type: ![[GL:[0-9]+]]
// O2-DAG: ![[LATE_BLOCK]] = distinct !DILexicalBlock(scope: ![[LATE_OUTER:[0-9]+]],
// O2-DAG: ![[LATE_OUTER]] = distinct !DILexicalBlock(scope: ![[LATE_SP]],
// O2-DAG: ![[GL]] = !DICompositeType(tag: DW_TAG_structure_type, name: "uniform struct G", scope: ![[LATE_SP]],

// CHECK-DAG: define {{.*}}@asPtr{{.*}} !dbg ![[ASPTR_SP:[0-9]+]] {
// CHECK-DAG: ![[ASPTR_SP]] = distinct !DISubprogram(name: "asPtr", {{.*}}type: ![[ASPTR_TYPE:[0-9]+]],
// CHECK-DAG: ![[ASPTR_TYPE]] = !DISubroutineType({{.*}}types: ![[ASPTR_TYPES:[0-9]+]])
// CHECK-DAG: ![[ASPTR_TYPES]] = !{![[ASPTR_RET:[0-9]+]], !{{[0-9]+}}}
// CHECK-DAG: ![[ASPTR_RET]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[GF:[0-9]+]]
// CHECK-DAG: ![[GF]] = !DICompositeType(tag: DW_TAG_structure_type, name: "uniform struct G", scope: ![[ASPTR_SP]], {{.*}}elements: ![[GF_ELTS:[0-9]+]])
// CHECK-DAG: ![[GF_ELTS]] = !{![[F:[0-9]+]]}
// CHECK-DAG: ![[F]] = !DIDerivedType(tag: DW_TAG_member, name: "f",

// The inlined function has no definition, but its variable is described in
// its caller with its struct in its own scope (at -O2, the variable is
// optimized away).  Before LLVM 21, the declaration is a call to
// llvm.dbg.declare, with the location attached by !dbg.
// O0-DAG: {{#dbg_declare\(|@llvm.dbg.declare\(metadata }}ptr %gi{{[^,]*}}, {{(metadata )?}}![[GI_VAR:[0-9]+]], {{(metadata )?}}!DIExpression(){{(, |\), !dbg )}}![[GI_LOC:[0-9]+]]
// O0-DAG: ![[GI_VAR]] = !DILocalVariable(name: "gi", {{.*}}type: ![[GI:[0-9]+]]
// O0-DAG: ![[GI_LOC]] = !DILocation({{.*}}scope: ![[GI_SCOPE:[0-9]+]], inlinedAt: !{{[0-9]+}})
// O0-DAG: ![[GI_SCOPE]] = distinct !DILexicalBlock(scope: ![[GI_OUTER:[0-9]+]],
// O0-DAG: ![[GI_OUTER]] = distinct !DILexicalBlock(scope: ![[INLINED_SP:[0-9]+]],
// O0-DAG: ![[INLINED_SP]] = distinct !DISubprogram(name: "inlined",
// O0-DAG: ![[GI]] = !DICompositeType(tag: DW_TAG_structure_type, name: "uniform struct G", scope: ![[INLINED_SP]],

// The removed instantiations leave nothing behind.
// REMOVED-NOT: early___uni___uni
// REMOVED-NOT: foreign___uni___uni
// REMOVED-NOT: define {{.*}}@inlined
// REMOVED-O2-NOT: late___uni___uni
