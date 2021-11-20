=========================
invoke_sycl specification
=========================

Introduction
------------
The ``invoke_sycl`` introduces a mechanism to invoke SPMD function from a SIMD context.

Dependencies
------------
`SYCL_EXT_ONEAPI_UNIFORM`_

.. _SYCL_EXT_ONEAPI_UNIFORM: https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/Uniform/Uniform.asciidoc

`SYCL_EXT_ONEAPI_INVOKE_SIMD`_

.. _SYCL_EXT_ONEAPI_INVOKE_SIMD: https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/InvokeSIMD/InvokeSIMD.asciidoc#invoking-a-simd-function

Overview
--------
The ``invoke_sycl`` adds support for an execution model in which SPMD functions are called from SIMD context. In particular, below we will consider calling SYCL functions from ISPC. A similar approach may be applied to call between other languages like call SYCL from ESIMD or call OpenCL from ISPC.
ISPC execution model defines an execution width, which is mapped to subgroup size in SYCL execution model. For example, ``gen9-x16`` target has an execution width 16, so SYCL subgroup size will be 16 to match it. The invocation may be in both convergent and non-convergent contexts.

Defining SYCL function
----------------------
The example below shows a simple SYCL function that multiplies ``in`` by ``factor`` and saves it to ``out[sg_id]``.

.. code-block:: cpp

    namespace sycl {
    #define VL 16
    extern "C" [[intel::reqd_sub_group_size(VL)]] SYCL_EXTERNAL void vmult(int sg_id, float in, sycl::ext::oneapi::experimental::uniform<float *>out,
                                                                           sycl::ext::oneapi::experimental::uniform<float> factor) {
        out[sg_id] = in * factor;
    }
    } // namespace sycl

SYCL function must be declared with ``SYCL_EXTERNAL`` specifier and have ``[[intel::reqd_sub_group_size(VL)]]`` attribute  where VL equals to ISPC SIMD width.
Each function parameter must be an arithmetic type or a trivially copyable type wrapped in a ``sycl::ext::oneapi::experimental::uniform``. Arguments may not be pointers or references, but pointers (like any other non-arithmetic type) may be passed if wrapped in a ``sycl::ext::oneapi::experimental::uniform``. Any such pointer value must point to memory that is accessible by all work-items in the sub-group.

There are several specific restrictions caused by ISPC:
- ISPC always uses global memory, so SYCL pointers must point to an allocation in global memory and address space for such pointers must be global or generic.
- SYCL function must be defined with `extern "C"` to be called from ISPC (C-based language). It is the same approach as is used for C++/ISPC interop on CPU (https://ispc.github.io/ispc.html#interoperability-with-the-application).

**DPC++ LLVM IR**

For the example used above the following LLVM IR declaration is expected:

.. code-block:: llvm

    define dso_local spir_func void @vmult(i32 %sg_id, float %in, %"class.sycl::ext::oneapi::experimental::uniform"* nocapture readonly
    byval(%"class.sycl::ext::oneapi::experimental::uniform") align 8 %out, %"class.sycl::ext::oneapi::experimental::uniform.0"* nocapture readonly
    byval(%"class.sycl::ext::oneapi::experimental::uniform.0") align 4 %factor) local_unnamed_addr #0 !intel_reqd_sub_group_size !5
    !5 = !{i32 16}


**Assumed IGC scalar backend action**

For such function we expect it to be vectorized for subgroup width VL at vISA level according to `SYCL_EXT_ONEAPI_INVOKE_SIMD`_.

.. _SYCL_EXT_ONEAPI_INVOKE_SIMD: https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/InvokeSIMD/InvokeSIMD.asciidoc#invoking-a-simd-function

"All arithmetic arguments of type T are converted to type ``sycl::ext::oneapi::experimental::simd<T, N>``, where N is the sub-group size of the calling kernel. Element i of the SIMD type represents the value from the work-item with sub-group local ID i.
Arguments of type ``sycl::ext::oneapi::experimental::uniform<T>`` are converted to type T. Conversion follows the same rules as the implicit conversion operator T() from the ``sycl::ext::oneapi::experimental::uniform<T>`` class; if the return value of operator T() would be undefined, the value of the scalar variable passed to the SIMD function is undefined."

Pseudo-code:

.. code-block:: llvm

    @vmult(<16 x i32> %sg_id, <16 x float> %in, float* %out, float %factor)


Invoking SYCL function
----------------------
The ``invoke_sycl`` function invokes an SPMD function across the SIMD lane.

The following conditions should be satisfied to be called from ISPC:

* The SPMD function must be declared on ISPC side with ``extern "C"`` qualifier.
* Each function parameter must be an uniform or varying arithmetic type or uniform pointer to an allocation in global memory. Address space for such pointers must be global or generic.
* The signature of this function should correspond to the **vectorized** version of SYCL function. Parameters marked as ``uniform`` on ISPC side should be wrapped into ``sycl::ext::oneapi::experimental::uniform`` on SYCL side. Parameters marked as ``varying`` or without explicit ``varying`` modifier (varying is the default in ISPC) should be declared as scalars on DPC++ side, they will be vectorized later by BE.

Here is an example of calling SYCL function used in the examples above:

.. code-block:: cpp

    // ispc --target=gen9-x16 simple.ispc
    struct Parameters {
        float *vin;
        float *vout;
        int    count;
    };

    extern "C" void vmult(varying int sg, varying float in, uniform float * uniform out, uniform int factor);

    typedef void (*FuncType)(varying int sg_id, varying float a, uniform float * uniform res, uniform int factor);

    task void simple_ispc(void *uniform _p) {
        Parameters *uniform p = (Parameters * uniform) _p;
        FuncType *uniform funcobj = vmult;
        foreach (index = 0 ... p->count) {
            varying float v = p->vin[index];

            // Example of invoke_sycl call
            invoke_sycl(funcobj, programIndex, v, p->vout, 2);

            if (v < 3.)
                // Example of invoke_sycl call in SIMD control flow
                invoke_sycl(funcobj, programIndex, v, p->vout, 2);
            else
                v = sqrt(v);
        }
    }


``invoke_sycl`` can be called in divergent and convergent contexts.
In case when ``invoke_sycl`` is called in divergent CF, HW mask is set before calling to SYCL function. There is no need for users to pass the mask explicitly.

Pseudo-code (LLVM IR):

.. code-block:: llvm

    %v = call i1 @llvm.genx.simdcf.any(<16 x i1> %mask)
    br i1 %v, label %funcall_call, label %funcall_done

    funcall_call:
    call spir_func void @funcobj(<16 x i32> sg_id, <16 x float> %a, float addrspace(4)* %res, i32 %factor)
    br label %funcall_done

    funcall_done:
    br label %exit


**Using SYCL classes**

There is no way to create and use SYCL objects (e.g. item, nd-item, group, sub-group) inside of ISPC, so in order to call arbitrary SYCL functionality, ``invoke_sycl`` allows to pass a special type, so the users can specify when they call ``invoke_sycl``, to request a specific SYCL handle (e.g. a sycl::sub_group) to be created and passed into the function as an argument.
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

User may specify limited number of SYCL objects that can be extended in the future: ``ispc_sub_group_placeholder()``, ``ispc_group_placeholder()``, ``ispc_nd_item_placeholder()``, ``ispc_item_placeholder()``.