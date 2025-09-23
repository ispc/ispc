// This test checks that masked store is promoted to blend stores for structures
// RUN: %{ispc} --target=host --nowrap --debug-phase=280:280 %s -o %t 2>&1 | FileCheck %s

// Original issue
struct Compound {
    int a;
    float<16> value;
};

// CHECK-LABEL: @test
// CHECK-COUNT-16: call void @__masked_store_blend_float
unmasked void test(Compound& data, float test) {
    Compound copy = data;
    if (test <= 0) {
        copy.value += 0.01;
    }
    data = copy;
}

// Deeply nested struct with vector fields
struct InnerStruct {
    uniform int padding;
    int<4> data;
};

struct MiddleStruct {
    float<4> additional_data;
    InnerStruct inner;
    int more_padding;
};

struct OuterStruct {
    MiddleStruct middle;
    double<4> outer_data;
    uniform int final_padding;
};

// CHECK-LABEL: @test_deeply_nested
// CHECK-COUNT-4: call void @__masked_store_blend_i32
// CHECK-COUNT-4: call void @__masked_store_blend_float
// CHECK-COUNT-4: call void @__masked_store_blend_double
unmasked void test_deeply_nested(OuterStruct & compound, float test) {
    OuterStruct copy = compound;
    if (test <= 0) {
        copy.middle.inner.data += 0.01;
        copy.middle.additional_data += 0.02;
        copy.outer_data += 0.03;
    }
    compound = copy;
}

// Struct with multiple vector fields
struct MultiVectorStruct {
    float<4> vector1;
    int<4> vector2;
    int padding1;
    float<4> vector4;
    int padding2;
};

// CHECK-LABEL: @test_multi_vector
// CHECK-COUNT-4: call void @__masked_store_blend_float
// CHECK-COUNT-4: call void @__masked_store_blend_i32
// CHECK-COUNT-4: call void @__masked_store_blend_float
unmasked void test_multi_vector(MultiVectorStruct & data, float test) {
    MultiVectorStruct copy = data;
    if (test > 0) {
        copy.vector1 *= 2.0;
        copy.vector2 += 1;
        copy.vector4 -= 0.5;
    }
    data = copy;
}

// Global structure (should not blend)
// CHECK-LABEL: @test_multi_vector_global
// CHECK-COUNT-4: call void @__masked_store_float
// CHECK-COUNT-4: call void @__masked_store_i32
// CHECK-COUNT-4: call void @__masked_store_float
MultiVectorStruct global_struct;
unmasked void test_multi_vector_global(MultiVectorStruct & data, float test) {
    if (test > 0) {
        global_struct.vector1 *= 2.0;
        global_struct.vector2 += 1;
        global_struct.vector4 -= 0.5;
    }
}



