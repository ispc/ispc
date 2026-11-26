// Check that returning addresses that are not on the stack of the function
// compiles with --werror: storage reached through a reference member of a
// local struct or a by-value struct parameter, a pointer loaded from a
// by-value struct parameter, and globals used in function templates.

// RUN: %{ispc} --target=host --nowrap --nostdlib --werror %s -o %t.o

struct WithRef {
    uniform int &ref;
};

struct WithNestedRef {
    WithRef inner;
};

uniform int global;
uniform int global_arr[4];

uniform int *uniform addr_of_ref_member() {
    uniform WithRef s = {global};
    return &s.ref;
}

uniform int *uniform addr_of_ref_member_param(uniform WithRef s) { return &s.ref; }

uniform int *uniform addr_of_nested_ref_member() {
    uniform WithNestedRef n = {{global}};
    return &n.inner.ref;
}

uniform int &ref_member() {
    uniform WithRef s = {global};
    return s.ref;
}

template <typename T> noinline uniform int *uniform global_in_template(T value) { return &global; }

template <typename T> noinline uniform int *uniform global_decay_in_template(T value) { return global_arr; }

uniform int *uniform use_global_in_template() { return global_in_template<uniform int>(1); }

uniform int *uniform use_global_decay_in_template() { return global_decay_in_template<uniform int>(1); }

// The implicit conversion to the return type loads the pointer that is stored
// in the parameter, so the address of the parameter itself isn't returned.

typedef uniform int *uniform P;

struct Box {
    P p;
};

P unbox(uniform Box b) { return (P &)(uniform Box &)b; }

P unbox_cast(uniform Box b) { return (P)(P &)(uniform Box &)b; }

void *uniform unbox_void_cast(uniform Box b) { return (void *uniform)(P &)(uniform Box &)b; }
