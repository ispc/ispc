#define __ZMM64BIT__
#include "knc-i1x8.h"

/* the following tests fails because on KNC native vec8_i32 and vec8_float are 512 and not 256 bit in size.
 *
 *  Using test compiler: Intel(r) SPMD Program Compiler (ispc), 1.4.5dev (build commit d68dbbc7bce74803 @ 20130919, LLVM 3.3)
 *  Using C/C++ compiler: icpc (ICC) 14.0.0 20130728
 *
 */

/* knc-i1x8unsafe_fast.h fails: 
 * ----------------------------
1 / 1206 tests FAILED compilation:
	./tests/ptr-assign-lhs-math-1.ispc
33 / 1206 tests FAILED execution:
	./tests/array-gather-simple.ispc
	./tests/array-gather-vary.ispc
	./tests/array-multidim-gather-scatter.ispc
	./tests/array-scatter-vary.ispc
	./tests/atomics-5.ispc
	./tests/atomics-swap.ispc
	./tests/cfor-array-gather-vary.ispc
	./tests/cfor-gs-improve-varying-1.ispc
	./tests/cfor-struct-gather-2.ispc
	./tests/cfor-struct-gather-3.ispc
	./tests/cfor-struct-gather.ispc
	./tests/gather-struct-vector.ispc
	./tests/global-array-4.ispc
	./tests/gs-improve-varying-1.ispc
	./tests/half-1.ispc
	./tests/half-3.ispc
	./tests/half.ispc
	./tests/launch-3.ispc
	./tests/launch-4.ispc
	./tests/masked-scatter-vector.ispc
	./tests/masked-struct-scatter-varying.ispc
	./tests/new-delete-6.ispc
	./tests/ptr-24.ispc
	./tests/ptr-25.ispc
	./tests/short-vec-15.ispc
	./tests/struct-gather-2.ispc
	./tests/struct-gather-3.ispc
	./tests/struct-gather.ispc
	./tests/struct-ref-lvalue.ispc
	./tests/struct-test-118.ispc
	./tests/struct-vary-index-expr.ispc
	./tests/typedef-2.ispc
	./tests/vector-varying-scatter.ispc
*/

/* knc-i1x8.h fails: 
 * ----------------------------
1 / 1206 tests FAILED compilation:
	./tests/ptr-assign-lhs-math-1.ispc
3 / 1206 tests FAILED execution:
	./tests/half-1.ispc
	./tests/half-3.ispc
	./tests/half.ispc
*/

/* knc-i1x8.h fails: 
 * ----------------------------
1 / 1206 tests FAILED compilation:
        ./tests/ptr-assign-lhs-math-1.ispc
4 / 1206 tests FAILED execution:
        ./tests/half-1.ispc
        ./tests/half-3.ispc
        ./tests/half.ispc
        ./tests/test-141.ispc
*/

/* generic-16.h fails: (from these knc-i1x8.h & knc-i1x16.h are derived 
 * ----------------------------
1 / 1206 tests FAILED compilation:
        ./tests/ptr-assign-lhs-math-1.ispc
6 / 1206 tests FAILED execution:
        ./tests/func-overload-max.ispc
        ./tests/half-1.ispc
        ./tests/half-3.ispc
        ./tests/half.ispc
        ./tests/test-141.ispc
        ./tests/test-143.ispc
*/



