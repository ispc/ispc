// Check that a warning is issued when a function returns the address of, or a
// reference to, a local variable or a parameter, and that no warning is issued
// when the returned address outlives the function.

// RUN: %{ispc} --target=host --nowrap --nostdlib %s -o %t.o 2>&1 | FileCheck %s --implicit-check-not=Warning

struct S {
    int member;
    int arr[4];
};

struct WithRef {
    uniform int &ref;
};

struct WithNestedRef {
    WithRef inner;
};

uniform int global;
uniform int global_arr[4];

// CHECK: [[@LINE+3]]:12: Warning: Address of stack memory associated with local variable "bad_local" returned.
uniform int *uniform addr_of_local() {
    uniform int bad_local;
    return &bad_local;
}

// CHECK: [[@LINE+3]]:12: Warning: Address of stack memory associated with local variable "bad_array" returned.
uniform int *uniform array_decay() {
    uniform int bad_array[4];
    return bad_array;
}

// CHECK: [[@LINE+2]]:12: Warning: Address of stack memory associated with parameter "bad_param" returned.
uniform int *uniform addr_of_param(uniform int bad_param) {
    return &bad_param;
}

// CHECK: [[@LINE+3]]:12: Warning: Reference to stack memory associated with local variable "bad_ref" returned.
uniform int &ref_to_local() {
    uniform int bad_ref;
    return bad_ref;
}

// CHECK: [[@LINE+2]]:12: Warning: Reference to stack memory associated with parameter "bad_ref_param" returned.
uniform int &ref_to_param(uniform int bad_ref_param) {
    return bad_ref_param;
}

// CHECK: [[@LINE+3]]:12: Warning: Address of stack memory associated with local variable "bad_struct" returned.
uniform int *uniform addr_of_member() {
    uniform S bad_struct;
    return &bad_struct.member;
}

// CHECK: [[@LINE+3]]:12: Warning: Address of stack memory associated with local variable "bad_elt" returned.
uniform int *uniform addr_of_element() {
    uniform int bad_elt[4];
    return &bad_elt[1];
}

// CHECK: [[@LINE+3]]:12: Warning: Address of stack memory associated with local variable "bad_nested" returned.
uniform int *uniform addr_of_nested() {
    uniform S bad_nested[2];
    return &bad_nested[1].arr[2];
}

// CHECK: [[@LINE+3]]:12: Warning: Address of stack memory associated with local variable "bad_cast" returned.
void *uniform addr_with_cast() {
    uniform int bad_cast;
    return (void *uniform)&bad_cast;
}

// CHECK: [[@LINE+3]]:12: Warning: Address of stack memory associated with local variable "bad_varying" returned.
uniform int *varying_addr_of_local(int index) {
    uniform int bad_varying[programCount];
    return &bad_varying[index & (programCount - 1)];
}

// CHECK: [[@LINE+3]]:12: Warning: Reference to stack memory associated with local variable "bad_ref_member" returned.
uniform int &ref_to_member() {
    uniform S bad_ref_member;
    return bad_ref_member.member;
}

// CHECK: [[@LINE+3]]:12: Warning: Address of stack memory associated with local variable "bad_offset" returned.
uniform int *uniform addr_with_offset() {
    uniform int bad_offset[4];
    return &bad_offset[0] + 1;
}

// CHECK: [[@LINE+3]]:12: Warning: Address of stack memory associated with local variable "bad_decay_offset" returned.
uniform int *uniform decay_with_offset(uniform int i) {
    uniform int bad_decay_offset[4];
    return i + bad_decay_offset - 1;
}

// CHECK: [[@LINE+3]]:27: Warning: Reference to stack memory associated with local variable "bad_ref_cast" returned.
uniform int &ref_cast_of_local() {
    uniform int bad_ref_cast;
    return (uniform int &)bad_ref_cast;
}

// CHECK: [[@LINE+2]]:27: Warning: Reference to stack memory associated with parameter "bad_ref_cast_param" returned.
uniform int &ref_cast_of_param(uniform int bad_ref_cast_param) {
    return (uniform int &)bad_ref_cast_param;
}

// CHECK: [[@LINE+3]]:12: Warning: Address of stack memory associated with local variable "bad_select_true" returned.
uniform int *uniform select_true(uniform bool flag) {
    uniform int bad_select_true;
    return flag ? &bad_select_true : &global;
}

// CHECK: [[@LINE+3]]:12: Warning: Address of stack memory associated with local variable "bad_select_false" returned.
uniform int *uniform select_false(uniform bool flag) {
    uniform int bad_select_false;
    return flag ? &global : &bad_select_false;
}

// CHECK: [[@LINE+3]]:12: Warning: Address of stack memory associated with local variable "bad_varying_select" returned.
uniform int *varying varying_select(bool flag) {
    uniform int bad_varying_select;
    return flag ? &global : &bad_varying_select;
}

// CHECK: [[@LINE+3]]:13: Warning: Address of stack memory associated with local variable "bad_comma" returned.
uniform int *uniform comma_operator() {
    uniform int bad_comma;
    return (1, &bad_comma);
}

// CHECK: [[@LINE+3]]:12: Warning: Address of stack memory associated with local variable "bad_interior" returned.
uniform int *uniform interior_of_offset(uniform int i) {
    uniform int bad_interior[4];
    return &(bad_interior + 1)[i];
}

// CHECK: [[@LINE+3]]:12: Warning: Address of stack memory associated with local variable "bad_arrow" returned.
uniform int *uniform arrow_through_address() {
    uniform S bad_arrow;
    return &(&bad_arrow)->member;
}

// CHECK: [[@LINE+3]]:12: Warning: Address of stack memory associated with parameter "bad_assign" returned.
uniform int *uniform assignment(uniform int bad_assign) {
    uniform int *uniform p;
    return p = &bad_assign;
}

// Stack memory that isn't a variable: a temporary that a reference is bound
// to, and memory from alloca().

// CHECK: [[@LINE+2]]:28: Warning: Reference to stack memory returned.
uniform int &ref_to_temporary(uniform int x) {
    return (uniform int &)(x + 1);
}

// CHECK: [[@LINE+3]]:28: Warning: Reference to stack memory returned.
uniform int &ref_to_selected_temporary(uniform bool flag) {
    uniform int x, y;
    return (uniform int &)(flag ? x : y);
}

// CHECK: [[@LINE+2]]:12: Warning: Address of stack memory returned.
void *uniform alloca_memory(uniform int n) {
    return alloca(n);
}

// CHECK: [[@LINE+2]]:12: Warning: Address of stack memory returned.
uniform int *uniform alloca_memory_cast(uniform int n) {
    return (uniform int *uniform)alloca(n);
}

// CHECK: [[@LINE+3]]:12: Warning: Address of stack memory associated with local variable "bad_export" returned.
export uniform int *uniform addr_in_export() {
    uniform int bad_export;
    return &bad_export;
}

// None of the following must be diagnosed.

uniform int *uniform addr_of_static() {
    static uniform int ok_static;
    return &ok_static;
}

uniform int *uniform static_array_decay() {
    static uniform int ok_static_arr[4];
    return ok_static_arr;
}

uniform int *uniform addr_of_global() { return &global; }

uniform int *uniform global_decay() { return global_arr; }

uniform int &ref_to_global() { return global; }

uniform int *uniform pointer_value(uniform int *uniform ok_ptr) { return ok_ptr; }

uniform int *uniform pointer_element(uniform int *uniform ok_ptr_elt) { return &ok_ptr_elt[1]; }

uniform int *uniform pointer_deref(uniform int *uniform ok_ptr_deref) { return &*ok_ptr_deref; }

uniform int *uniform pointer_member(uniform S *uniform ok_ptr_member) { return &ok_ptr_member->member; }

uniform int *uniform pointer_member_decay(uniform S *uniform ok_ptr_decay) { return ok_ptr_decay->arr; }

uniform int *uniform addr_of_ref_param(uniform int &ok_ref_param) { return &ok_ref_param; }

uniform int *uniform ref_member(uniform S &ok_ref_struct) { return &ok_ref_struct.member; }

uniform int &ref_passthrough(uniform int &ok_ref_pass) { return ok_ref_pass; }

uniform int &ref_from_pointer(uniform int *uniform ok_ref_ptr) { return *ok_ref_ptr; }

uniform int &ref_from_ref_struct(uniform S &ok_ref_s) { return ok_ref_s.arr[1]; }

uniform int *uniform local_pointer() {
    uniform int *uniform ok_local_ptr = &global;
    return ok_local_ptr;
}

uniform int *uniform local_pointer_element() {
    uniform int *uniform ok_local_ptr_elt = global_arr;
    return &ok_local_ptr_elt[2];
}

uniform int *uniform pointer_offset(uniform int *uniform ok_ptr_offset) { return ok_ptr_offset + 1; }

uniform int *uniform global_offset() { return global_arr + 1; }

// Globals that are used in a function template are not local to its instances.

template <typename T> noinline uniform int *uniform global_in_template(T value) { return &global; }

template <typename T> noinline uniform int *uniform global_decay_in_template(T value) { return global_arr; }

template <typename T> noinline uniform int &global_ref_in_template(T value) { return global; }

template <typename T> noinline uniform int *uniform static_in_template(T value) {
    static uniform int ok_static_in_template;
    return &ok_static_in_template;
}

uniform int *uniform use_global_in_template() { return global_in_template<uniform int>(1); }

uniform int *uniform use_global_decay_in_template() { return global_decay_in_template<uniform int>(1); }

uniform int &use_global_ref_in_template() { return global_ref_in_template<uniform int>(1); }

uniform int *uniform use_static_in_template() { return static_in_template<uniform int>(1); }

// A reference member refers to storage outside of the struct that holds it.

uniform int *uniform addr_of_ref_member() {
    uniform WithRef ok_ref_member = {global};
    return &ok_ref_member.ref;
}

uniform int *uniform addr_of_ref_member_param(uniform WithRef ok_ref_member_param) { return &ok_ref_member_param.ref; }

uniform int *uniform addr_of_nested_ref_member() {
    uniform WithNestedRef ok_nested_ref_member = {{global}};
    return &ok_nested_ref_member.inner.ref;
}

uniform int &ref_member() {
    uniform WithRef ok_ref_member_ref = {global};
    return ok_ref_member_ref.ref;
}

uniform int *uniform select_globals(uniform bool flag) { return flag ? &global : &global_arr[1]; }

uniform int *uniform comma_global() { return (1, &global); }

uniform int *varying varying_select_globals(bool flag) { return flag ? &global : &global_arr[programIndex & 3]; }

uniform int *uniform pointer_param_offset_element(uniform int *uniform ok_param_base, uniform int i) {
    return &(ok_param_base + 1)[i];
}

uniform int *uniform global_offset_element(uniform int i) { return &(global_arr + 1)[i]; }

uniform int *uniform pointer_param_arrow(uniform S *uniform ok_arrow_param) { return &ok_arrow_param->arr[1]; }

uniform int *uniform assign_global() {
    uniform int *uniform ok_assign_global;
    return ok_assign_global = &global;
}

uniform int *uniform compound_assign(uniform int *uniform ok_compound) { return ok_compound += 1; }

uniform int value_of_local() {
    uniform int ok_value = 1;
    return ok_value;
}

// Function templates are diagnosed when they are instantiated, which happens
// after the rest of the file is processed.

// CHECK: [[@LINE+3]]:12: Warning: Address of stack memory associated with local variable "bad_template" returned.
template <typename T> noinline T *uniform addr_in_template(T value) {
    T bad_template = value;
    return &bad_template;
}

uniform int *uniform use_template() { return addr_in_template<uniform int>(1); }
