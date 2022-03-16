=========================
invoke_sycl specification
=========================

Introduction
------------
The ``invoke_sycl`` introduces a mechanism to invoke SPMD function from a SIMD context.

Dependencies
------------
`SYCL_EXT_ONEAPI_UNIFORM`_

.. _SYCL_EXT_ONEAPI_UNIFORM: https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/proposed/sycl_ext_oneapi_uniform.asciidoc

`SYCL_EXT_ONEAPI_INVOKE_SIMD`_

.. _SYCL_EXT_ONEAPI_INVOKE_SIMD: https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/proposed/sycl_ext_oneapi_invoke_simd.asciidoc

Overview
--------
The ``invoke_sycl`` adds support for an execution model in which SPMD functions are called from SIMD context. In
particular, below we will consider calling SYCL functions from ISPC. A similar approach may be applied to call between
other languages like call SYCL from ESIMD or call OpenCL from ISPC. ISPC execution model defines an execution width,
which is mapped to subgroup size in SYCL execution model.  For example, ``gen9-x16`` target has an execution width 16,
so SYCL subgroup size will be 16 to match it. The invocation may be in both convergent and non-convergent contexts.

Defining SYCL function
----------------------
The example below shows a simple SYCL function that multiplies ``va`` by ``factor`` and adds it to ``vb[index]``.

.. code-block:: cpp

    namespace sycl {
        extern "C" SYCL_EXTERNAL void __regcall doVadd(float va, sycl::ext::oneapi::experimental::uniform<float *>vb, int index,
                                                       sycl::ext::oneapi::experimental::uniform<int> factor) {
            vb[index] += va * factor;
        }
    } // namespace sycl

SYCL function must be declared with ``SYCL_EXTERNAL`` specifier. Each function parameter must be an arithmetic type or a
trivially copyable type wrapped in a ``sycl::ext::oneapi::experimental::uniform``. Arguments may not be pointers or
references, but pointers (like any other non-arithmetic type) may be passed if wrapped in a
``sycl::ext::oneapi::experimental::uniform``. Any such pointer value must point to memory that is accessible by all
work-items in the sub-group (`sycl_ext_oneapi_invoke_simd
<https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/proposed/sycl_ext_oneapi_invoke_simd.asciidoc>`_)

There are several specific restrictions caused by ISPC:

* ISPC always uses global memory, so SYCL pointers must point to an allocation in global memory and address space for
  such pointers must be global or generic.  * SYCL function must be defined with ``extern "C"`` to be called from ISPC
  (C-based language).  It is the same approach as is used for C++/ISPC interop on CPU
  (https://ispc.github.io/ispc.html#interoperability-with-the-application).  * In addition SYCL funcion must be defined
  with ``__regcall`` specifier to match ISPC calling convention.

**SYCL LLVM IR**

For the example used above the following LLVM IR declaration is expected:

.. code-block:: llvm

    define dso_local x86_regcallcc void @__regcall3__vmult(float %va, float addrspace(4)* nocapture %vb, i32 %index, i32 %factor.coerce)


**SYCL backend action**

All arguments and return values are vectorized by SYCL backend, in particular by `IGC scalar backend
<https://github.com/intel/intel-graphics-compiler>`_, meaning that uniform values are copied into each lane of the
vector, where the vector width equals the currently compiled SIMD width.  So for example above, the function signature
after the backend transformations is expected to be as below:

Pseudo-code:

.. code-block:: llvm

    spir_func void @__regcall3__dpcpp_func(<16 x float>, <16 x i64>, <16 x i32>, <16 x i32>)


Invoking SYCL function
----------------------
The ``invoke_sycl`` function invokes an SPMD function across the SIMD lane.

The following conditions should be satisfied to be called from ISPC:

* The SPMD function must be declared on ISPC side with ``extern "SYCL"`` qualifier.  * Each function parameter must be a
  uniform or varying arithmetic type or uniform pointer to an allocation in global memory. Address space for such
  pointers must be global or generic.  * The signature of this function should correspond to the **vectorized** version
  of SYCL function. Parameters marked as ``uniform`` on ISPC side should be wrapped into
  ``sycl::ext::oneapi::experimental::uniform`` on SYCL side. Parameters marked as ``varying`` or without explicit
  ``varying`` modifier (varying is the default in ISPC) should be declared as scalars on SYCL side, they will be
  vectorized later by BE.

ISPC broadcasts uniform variables before passing into SYCL callee's uniform parameter, and treats uniform SYCL function
return value as non-uniform (vector) and takes the first element of the returned vector as the call result.

Here is an example of calling SYCL function used in the examples above:

.. code-block:: cpp

    // ispc --target=gen9-x16 simple.ispc
    struct Parameters {
        float *vin;
        float *vout;
        int    count;
    };

    extern "SYCL" __regcall void doVadd(varying float a,
                                        uniform float uniform *vout, int index,
                                        uniform int factor);

    task void simple_ispc(uniform float vin[], uniform float vout[],
                        uniform int count) {
        foreach (index = 0 ... count) {
            // Load the appropriate input value for this program instance.
            float v = vin[index];
            float v_out = vout[index];

            // Do an arbitrary little computation, but at least make the
            // computation dependent on the value being processed
            if (v < 3.)
            v = v * v;
            else
            v = sqrt(v);

            invoke_sycl(doVadd, v, vout, index, 2);
        }
    }

**Control flow**

``invoke_sycl`` can be called in divergent and convergent contexts.  In case when ``invoke_sycl`` is called in divergent
CF, HW mask is set before calling to SYCL function. There is no need for users to pass the mask explicitly.

Pseudo-code (LLVM IR):

.. code-block:: llvm

    %v = call i1 @llvm.genx.simdcf.any(<16 x i1> %mask)
    br i1 %v, label %funcall_call, label %funcall_done

    funcall_call:
    call spir_func void @funcobj(<16 x i32> sg_id, <16 x float> %a, float addrspace(4)* %res, i32 %factor)
    br label %funcall_done

    funcall_done:
    br label %exit


Note: SIMD control flow is managed differently for SYCL (based on scalar IGC) and ISPC (based on vector IGC). Until the
approach is unified between the backends, only convergent CF is supported.

**Using SYCL classes**

There is no way to create and use SYCL objects (e.g. item, nd-item, group, sub-group) inside of ISPC, so in order to
call arbitrary SYCL functionality, ``invoke_sycl`` allows to pass a special type, so the users can specify when they
call ``invoke_sycl``, to request a specific SYCL handle (e.g. a sycl::sub_group) to be created and passed into the
function as an argument.

For example:

.. code-block:: cpp

    // SYCL function needs a handle to the sub-group, which represents a set of work-items executing together (typically in SIMD style)
    extern “C” [[sycl::reqd_sub_group_size(VL)]] SYCL_EXTERNAL float my_reduce(sycl::sub_group sg, float x) {
        return sycl::reduce(sg, x, sycl::plus<>());
    }

    task void ispc_caller() {
        varying float x = foo();
        invoke_sycl(my_reduce, ispc_sub_group_placeholder{}, x);
    }

User may specify limited number of SYCL objects that can be extended in the future: ``ispc_sub_group_placeholder()``,
``ispc_group_placeholder()``, ``ispc_nd_item_placeholder()``, ``ispc_item_placeholder()``.
