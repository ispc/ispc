// Regression test for https://github.com/ispc/ispc/issues/3960
// Pointers to forward declared structs refer to the complete struct, also if
// created before the definition, and never to another struct with the same
// name.  Structs declared in function bodies are described in the scope of
// each function that uses them.

// RUN: %{ispc} %s --target=host --nostdlib --nowrap --woff -g -O0 --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=host --nostdlib --nowrap --woff -g -O0 --emit-llvm-text -o - | FileCheck %s --check-prefix=REMOVED
// RUN: %{ispc} %s --target=host --nowrap --woff -g -O2 -o %t.o

struct A;
uniform A *uniform global;
struct A {
    int val;
};

struct G;
uniform G *uniform never;

uniform int get() { return global->val; }

// Local structs with the same name in different functions.
uniform int first(void *uniform v) {
    struct L;
    uniform L *uniform p1 = (uniform L * uniform)v;
    struct L {
        int first;
    };
    return p1->first;
}

uniform int second(void *uniform v) {
    struct L;
    uniform L *uniform p2 = (uniform L * uniform)v;
    struct L {
        int second;
    };
    return p2->second;
}

// A local struct with the name of a global struct that is never defined.
uniform int third(void *uniform v) {
    struct G {
        int local;
    };
    uniform G *uniform p3 = (uniform G * uniform)v;
    return p3->local;
}

// Exported only, so there is no internal version of the function.
__attribute__((external_only)) export uniform int fourth(void *uniform v) {
    struct G {
        int exported;
    };
    uniform G *uniform p4 = (uniform G * uniform)v;
    return p4->exported;
}

// Function template instantiations.
template <typename T> uniform int fifth(T t) {
    struct G {
        int templ;
    };
    uniform G g5 = {(uniform int)t};
    return g5.templ;
}

uniform int callFifth() { return fifth<uniform int>(1) + fifth<uniform float>(2); }

// Arrays and members of local structs.
uniform int sixth(uniform int n) {
    struct E {
        int e;
    };
    struct W {
        E inner;
        int w;
    };
    uniform E arr[2] = {{n}, {n + 1}};
    uniform W w6 = {{n}, n};
    return arr[1].e + w6.inner.e;
}

// Also if the struct is forward declared and used through a pointer first.
uniform int seventh(void *uniform v) {
    struct F;
    uniform F *uniform p7 = (uniform F * uniform)v;
    struct F {
        int f;
    };
    uniform F arr7[2] = {{1}, {2}};
    return p7->f + arr7[1].f;
}

// Other functions that use a local struct.
template <typename T> T *uniform asPtr(void *uniform p) { return (T * uniform) p; }

template <typename T> uniform int takeG(T *uniform p) { return p->arg; }

uniform int eighth(void *uniform p) {
    struct G {
        int ret;
        int arg;
    };
    return asPtr<uniform G>(p)->ret + takeG<uniform G>((uniform G * uniform) p);
}

// Structs that get equal debug types in the same function.
// clang-format off
#define BLOCK { struct D { int d; }; struct DW { D inner; }; struct DR { DW inner; }; uniform DW dw; uniform D d; uniform DR dr; }
// clang-format on
void ninth() { BLOCK BLOCK }

// An explicit specialization with its own local struct.
template <typename T> uniform int tenth(T t) {
    struct G {
        int primary;
    };
    uniform G g10 = {1};
    return g10.primary;
}

template <> uniform int tenth<uniform int>(uniform int t) {
    struct G {
        int special;
    };
    uniform G gs = {2};
    return gs.special;
}

uniform int callTenth() { return tenth<uniform int>(1) + tenth<uniform float>(1); }

// A function that is removed as unused.
static void eleventh() {
    struct G {
        int unused;
    };
    uniform G g11;
}

// A task.
task void twelfth(uniform int *uniform out) {
    struct T {
        int tsk;
    };
    uniform T t12 = {1};
    out[0] = t12.tsk;
}

// Self-referential and mutually recursive local structs.
uniform int thirteenth() {
    struct LN {
        LN *next;
        int v;
    };
    struct MA;
    struct MB {
        MA *a;
    };
    struct MA {
        MB *b;
        int m;
    };
    uniform LN ln = {NULL, 1};
    uniform MA ma;
    return ln.v;
}

// Variants of a local struct.
uniform int fourteenth() {
    struct V {
        int v;
    };
    varying V vv;
    soa<4> V sv[1];
    const uniform V cv = {1};
    vv.v = programIndex;
    return cv.v;
}

// A global pointer created before the definition refers to the complete struct.
// CHECK-DAG: !DIGlobalVariable(name: "global", {{.*}}type: ![[GLOBAL_PTR:[0-9]+]]
// CHECK-DAG: ![[GLOBAL_PTR]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[A:[0-9]+]]
// CHECK-DAG: ![[A]] = !DICompositeType(tag: DW_TAG_structure_type, name: "uniform struct A", {{.*}}size: 32, {{.*}}elements: ![[A_ELTS:[0-9]+]])
// CHECK-DAG: ![[A_ELTS]] = !{![[VAL:[0-9]+]]}
// CHECK-DAG: ![[VAL]] = !DIDerivedType(tag: DW_TAG_member, name: "val",

// A struct that is never defined stays a forward declaration in the namespace.
// CHECK-DAG: !DIGlobalVariable(name: "never", {{.*}}type: ![[NEVER_PTR:[0-9]+]]
// CHECK-DAG: ![[NEVER_PTR]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[G:[0-9]+]]
// CHECK-DAG: ![[G]] = !DICompositeType(tag: DW_TAG_structure_type, name: "uniform struct G", scope: ![[NS:[0-9]+]], {{.*}}flags: DIFlagFwdDecl)
// CHECK-DAG: ![[NS]] = !DINamespace(name: "ispc"

// Local structs with the same name are told apart and scoped to their functions.
// CHECK-DAG: ![[FIRST_SP:[0-9]+]] = distinct !DISubprogram(name: "first",
// CHECK-DAG: !DILocalVariable(name: "p1", {{.*}}type: ![[P1_PTR:[0-9]+]]
// CHECK-DAG: ![[P1_PTR]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L1:[0-9]+]]
// CHECK-DAG: ![[L1]] = !DICompositeType(tag: DW_TAG_structure_type, name: "uniform struct L", scope: ![[FIRST_SP]], {{.*}}elements: ![[L1_ELTS:[0-9]+]])
// CHECK-DAG: ![[L1_ELTS]] = !{![[FIRST:[0-9]+]]}
// CHECK-DAG: ![[FIRST]] = !DIDerivedType(tag: DW_TAG_member, name: "first",
// CHECK-DAG: ![[SECOND_SP:[0-9]+]] = distinct !DISubprogram(name: "second",
// CHECK-DAG: !DILocalVariable(name: "p2", {{.*}}type: ![[P2_PTR:[0-9]+]]
// CHECK-DAG: ![[P2_PTR]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L2:[0-9]+]]
// CHECK-DAG: ![[L2]] = !DICompositeType(tag: DW_TAG_structure_type, name: "uniform struct L", scope: ![[SECOND_SP]], {{.*}}elements: ![[L2_ELTS:[0-9]+]])
// CHECK-DAG: ![[L2_ELTS]] = !{![[SECOND:[0-9]+]]}
// CHECK-DAG: ![[SECOND]] = !DIDerivedType(tag: DW_TAG_member, name: "second",
// CHECK-DAG: ![[THIRD_SP:[0-9]+]] = distinct !DISubprogram(name: "third",
// CHECK-DAG: !DILocalVariable(name: "p3", {{.*}}type: ![[P3_PTR:[0-9]+]]
// CHECK-DAG: ![[P3_PTR]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[G3:[0-9]+]]
// CHECK-DAG: ![[G3]] = !DICompositeType(tag: DW_TAG_structure_type, name: "uniform struct G", scope: ![[THIRD_SP]], {{.*}}elements: ![[G3_ELTS:[0-9]+]])
// CHECK-DAG: ![[G3_ELTS]] = !{![[LOCAL:[0-9]+]]}
// CHECK-DAG: ![[LOCAL]] = !DIDerivedType(tag: DW_TAG_member, name: "local",

// CHECK-DAG: ![[FOURTH_SP:[0-9]+]] = distinct !DISubprogram(name: "fourth",
// CHECK-DAG: !DILocalVariable(name: "p4", {{.*}}type: ![[P4_PTR:[0-9]+]]
// CHECK-DAG: ![[P4_PTR]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[G4:[0-9]+]]
// CHECK-DAG: ![[G4]] = !DICompositeType(tag: DW_TAG_structure_type, name: "uniform struct G", scope: ![[FOURTH_SP]],

// Each instantiation of a function template has its own copy.
// CHECK-DAG: ![[FIFTH_INT_SP:[0-9]+]] = distinct !DISubprogram(name: "fifth", linkageName: "fifth___uni___uni",
// CHECK-DAG: ![[FIFTH_FLOAT_SP:[0-9]+]] = distinct !DISubprogram(name: "fifth", linkageName: "fifth___unf___unf",
// CHECK-DAG: !DILocalVariable(name: "g5", scope: ![[FIFTH_INT_BLOCK:[0-9]+]], {{.*}}type: ![[G5_INT:[0-9]+]]
// CHECK-DAG: ![[FIFTH_INT_BLOCK]] = distinct !DILexicalBlock(scope: ![[FIFTH_INT_OUTER:[0-9]+]],
// CHECK-DAG: ![[FIFTH_INT_OUTER]] = distinct !DILexicalBlock(scope: ![[FIFTH_INT_SP]],
// CHECK-DAG: ![[G5_INT]] = !DICompositeType(tag: DW_TAG_structure_type, name: "uniform struct G", scope: ![[FIFTH_INT_SP]],
// CHECK-DAG: !DILocalVariable(name: "g5", scope: ![[FIFTH_FLOAT_BLOCK:[0-9]+]], {{.*}}type: ![[G5_FLOAT:[0-9]+]]
// CHECK-DAG: ![[FIFTH_FLOAT_BLOCK]] = distinct !DILexicalBlock(scope: ![[FIFTH_FLOAT_OUTER:[0-9]+]],
// CHECK-DAG: ![[FIFTH_FLOAT_OUTER]] = distinct !DILexicalBlock(scope: ![[FIFTH_FLOAT_SP]],
// CHECK-DAG: ![[G5_FLOAT]] = !DICompositeType(tag: DW_TAG_structure_type, name: "uniform struct G", scope: ![[FIFTH_FLOAT_SP]],

// Arrays and members of local structs have their sizes.
// CHECK-DAG: !DILocalVariable(name: "arr", {{.*}}type: ![[ARR:[0-9]+]]
// CHECK-DAG: ![[ARR]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[E:[0-9]+]], size: 64, align: 32,
// CHECK-DAG: ![[E]] = !DICompositeType(tag: DW_TAG_structure_type, name: "uniform struct E", {{.*}}size: 32, align: 32,
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "inner", {{.*}}baseType: ![[E]], size: 32, align: 32)
// CHECK-DAG: !DILocalVariable(name: "arr7", {{.*}}type: ![[ARR7:[0-9]+]]
// CHECK-DAG: ![[ARR7]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[F:[0-9]+]], size: 64, align: 32,
// CHECK-DAG: !DILocalVariable(name: "p7", {{.*}}type: ![[P7_PTR:[0-9]+]]
// CHECK-DAG: ![[P7_PTR]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[F]]
// CHECK-DAG: ![[F]] = !DICompositeType(tag: DW_TAG_structure_type, name: "uniform struct F", {{.*}}size: 32, align: 32,

// Other functions that use a local struct have their own copies: here only
// through the return type of asPtr and the parameter of takeG.
// CHECK-DAG: ![[ASPTR_SP:[0-9]+]] = distinct !DISubprogram(name: "asPtr", {{.*}}type: ![[ASPTR_TYPE:[0-9]+]],
// CHECK-DAG: ![[ASPTR_TYPE]] = !DISubroutineType(types: ![[ASPTR_TYPES:[0-9]+]])
// CHECK-DAG: ![[ASPTR_TYPES]] = !{![[ASPTR_RET:[0-9]+]], !{{[0-9]+}}}
// CHECK-DAG: ![[ASPTR_RET]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[G8:[0-9]+]]
// CHECK-DAG: ![[G8]] = !DICompositeType(tag: DW_TAG_structure_type, name: "uniform struct G", scope: ![[ASPTR_SP]], {{.*}}elements: ![[G8_ELTS:[0-9]+]])
// CHECK-DAG: ![[G8_ELTS]] = !{![[RET:[0-9]+]], ![[ARG:[0-9]+]]}
// CHECK-DAG: ![[RET]] = !DIDerivedType(tag: DW_TAG_member, name: "ret",
// CHECK-DAG: ![[TAKEG_SP:[0-9]+]] = distinct !DISubprogram(name: "takeG", {{.*}}type: ![[TAKEG_TYPE:[0-9]+]],
// CHECK-DAG: ![[TAKEG_TYPE]] = !DISubroutineType(types: ![[TAKEG_TYPES:[0-9]+]])
// CHECK-DAG: ![[TAKEG_TYPES]] = !{!{{[0-9]+}}, ![[TAKEG_PARAM:[0-9]+]]}
// CHECK-DAG: !DILocalVariable(name: "p", arg: 1, scope: ![[TAKEG_SP]], {{.*}}type: ![[TAKEG_PARAM]])
// CHECK-DAG: ![[TAKEG_PARAM]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[G8_TAKEG:[0-9]+]]
// CHECK-DAG: ![[G8_TAKEG]] = !DICompositeType(tag: DW_TAG_structure_type, name: "uniform struct G", scope: ![[TAKEG_SP]],

// CHECK-DAG: !DILocalVariable(name: "dr",

// An explicit specialization has its own body and structs.
// CHECK-DAG: ![[TENTH_INT_SP:[0-9]+]] = distinct !DISubprogram(name: "tenth", linkageName: "tenth___uni___uni",
// CHECK-DAG: !DILocalVariable(name: "gs", {{.*}}type: ![[GS:[0-9]+]]
// CHECK-DAG: ![[GS]] = !DICompositeType(tag: DW_TAG_structure_type, name: "uniform struct G", scope: ![[TENTH_INT_SP]], {{.*}}elements: ![[GS_ELTS:[0-9]+]])
// CHECK-DAG: ![[GS_ELTS]] = !{![[SPECIAL:[0-9]+]]}
// CHECK-DAG: ![[SPECIAL]] = !DIDerivedType(tag: DW_TAG_member, name: "special",
// CHECK-DAG: ![[TENTH_FLOAT_SP:[0-9]+]] = distinct !DISubprogram(name: "tenth", linkageName: "tenth___unf___unf",
// CHECK-DAG: !DILocalVariable(name: "g10", {{.*}}type: ![[G10:[0-9]+]]
// CHECK-DAG: ![[G10]] = !DICompositeType(tag: DW_TAG_structure_type, name: "uniform struct G", scope: ![[TENTH_FLOAT_SP]],

// CHECK-DAG: ![[TWELFTH_SP:[0-9]+]] = distinct !DISubprogram(name: "twelfth",
// CHECK-DAG: !DILocalVariable(name: "t12", {{.*}}type: ![[T12:[0-9]+]]
// CHECK-DAG: ![[T12]] = !DICompositeType(tag: DW_TAG_structure_type, name: "uniform struct T", scope: ![[TWELFTH_SP]],

// Self-referential and mutually recursive local structs.
// CHECK-DAG: ![[THIRTEENTH_SP:[0-9]+]] = distinct !DISubprogram(name: "thirteenth",
// CHECK-DAG: !DILocalVariable(name: "ln", {{.*}}type: ![[LN:[0-9]+]]
// CHECK-DAG: ![[LN]] = !DICompositeType(tag: DW_TAG_structure_type, name: "uniform struct LN", scope: ![[THIRTEENTH_SP]], {{.*}}size: 128, {{.*}}elements: ![[LN_ELTS:[0-9]+]])
// CHECK-DAG: ![[LN_ELTS]] = !{![[NEXT:[0-9]+]], !{{[0-9]+}}}
// CHECK-DAG: ![[NEXT]] = !DIDerivedType(tag: DW_TAG_member, name: "next", {{.*}}baseType: ![[LN_PTR:[0-9]+]]
// CHECK-DAG: ![[LN_PTR]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[LN]]
// CHECK-DAG: !DILocalVariable(name: "ma", {{.*}}type: ![[MA:[0-9]+]]
// CHECK-DAG: ![[MA]] = !DICompositeType(tag: DW_TAG_structure_type, name: "uniform struct MA", scope: ![[THIRTEENTH_SP]], {{.*}}elements: ![[MA_ELTS:[0-9]+]])
// CHECK-DAG: ![[MA_ELTS]] = !{![[B:[0-9]+]], !{{[0-9]+}}}
// CHECK-DAG: ![[B]] = !DIDerivedType(tag: DW_TAG_member, name: "b", {{.*}}baseType: ![[MB_PTR:[0-9]+]]
// CHECK-DAG: ![[MB_PTR]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[MB:[0-9]+]]
// CHECK-DAG: ![[MB]] = !DICompositeType(tag: DW_TAG_structure_type, name: "uniform struct MB", scope: ![[THIRTEENTH_SP]], {{.*}}elements: ![[MB_ELTS:[0-9]+]])
// CHECK-DAG: ![[MB_ELTS]] = !{![[A_MEMBER:[0-9]+]]}
// CHECK-DAG: ![[A_MEMBER]] = !DIDerivedType(tag: DW_TAG_member, name: "a", {{.*}}baseType: ![[MA_PTR:[0-9]+]]
// CHECK-DAG: ![[MA_PTR]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[MA]]

// Variants of a local struct.
// CHECK-DAG: ![[FOURTEENTH_SP:[0-9]+]] = distinct !DISubprogram(name: "fourteenth",
// CHECK-DAG: !DILocalVariable(name: "vv", {{.*}}type: ![[VV:[0-9]+]]
// CHECK-DAG: ![[VV]] = !DICompositeType(tag: DW_TAG_structure_type, name: "varying struct V", scope: ![[FOURTEENTH_SP]],
// CHECK-DAG: !DILocalVariable(name: "sv", {{.*}}type: ![[SV_ARR:[0-9]+]]
// CHECK-DAG: ![[SV_ARR]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[SV:[0-9]+]]
// CHECK-DAG: ![[SV]] = !DICompositeType(tag: DW_TAG_structure_type, name: "soa<4> struct V", scope: ![[FOURTEENTH_SP]], {{.*}}size: 128,
// CHECK-DAG: !DILocalVariable(name: "cv", {{.*}}type: ![[CV:[0-9]+]]
// CHECK-DAG: ![[CV]] = !DICompositeType(tag: DW_TAG_structure_type, name: "const uniform struct V", scope: ![[FOURTEENTH_SP]], {{.*}}size: 32,

// The removed function leaves no debug types.
// REMOVED-NOT: name: "unused"

// Enums declared in function bodies have no identifier, through which LLVM
// would merge enums with the same name, e.g. when linking modules.
uniform int enumFirst() {
    enum LE { LE_A = 1 };
    uniform LE le1 = LE_A;
    return le1;
}
uniform int enumSecond() {
    enum LE { LE_B = 2 };
    uniform LE le2 = LE_B;
    return le2;
}
enum GE { GE_A = 3 };
uniform int enumGlobal() {
    uniform GE ge = GE_A;
    return ge;
}
// CHECK-DAG: !DILocalVariable(name: "le1", {{.*}}type: ![[LE1:[0-9]+]]
// CHECK-DAG: ![[LE1]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "uniform enum LE", {{.*}}elements: ![[LE1_ELTS:[0-9]+]]){{$}}
// CHECK-DAG: ![[LE1_ELTS]] = !{![[LE_A:[0-9]+]]}
// CHECK-DAG: ![[LE_A]] = !DIEnumerator(name: "LE_A", value: 1)
// CHECK-DAG: !DILocalVariable(name: "le2", {{.*}}type: ![[LE2:[0-9]+]]
// CHECK-DAG: ![[LE2]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "uniform enum LE", {{.*}}elements: ![[LE2_ELTS:[0-9]+]]){{$}}
// CHECK-DAG: ![[LE2_ELTS]] = !{![[LE_B:[0-9]+]]}
// CHECK-DAG: ![[LE_B]] = !DIEnumerator(name: "LE_B", value: 2)
// CHECK-DAG: !DICompositeType(tag: DW_TAG_enumeration_type, name: "uniform enum GE", {{.*}}identifier: "GE")
