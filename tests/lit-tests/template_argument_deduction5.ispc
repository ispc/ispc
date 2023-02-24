// Check that correct version of the template was instantiated and called.
// RUN: %{ispc} %s --target=host --emit-llvm-text -O0 -o - | FileCheck %s

struct S {
    float V[4];
};

template <typename T> noinline uniform T foo(const uniform T &V1, const uniform T &V2) {
    uniform T Result;
    foreach (i = 0 ... 4) {
        Result.V[i] = max(V1.V[i], V2.V[i]);
    }
    return Result;
}

template <typename T> noinline varying T foo(const varying T &V1, const varying T &V2) {
    varying T Result;
    for (uniform int i = 0; i < 4; i++) {
        Result.V[i] = max(V1.V[i], V2.V[i]);
    }
    return Result;
}

// CHECK-LABEL: define {{.*}} @bar1
// CHECK: call {{.*}} @foo___s_5B_vyS_5D____REFs_5B__c_unS_5D_REFs_5B__c_unS_5D_
unmasked const uniform S bar1(const uniform S Vb, const uniform S Vc) {
    return foo(Vb, Vc);
}

// CHECK-LABEL: define {{.*}} @bar2
// CHECK: call {{.*}} @foo___s_5B_unS_5D____REFs_5B__c_vyS_5D_REFs_5B__c_vyS_5D_
unmasked const varying S bar2(const varying S Vb, const varying S Vc) {
    return foo(Vb, Vc);
}

