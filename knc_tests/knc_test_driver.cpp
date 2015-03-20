#include "knc_test_driver_core.h"

int main () {
    
    test_insert_extract();
    test_load_store();
    test_smear();
    test_setzero();
    test_select();
    test_broadcast();
    test_rotate();
    test_shift();
    test_shuffle();
    test_cast();
    test_cmp();
    test_cast_bits();
    test_reduce();
    test_popcnt();
    test_count_zeros();
    test_other();
    test_masked_load_store();
    test_binary_op();
    test_gather_scatter();

    return 0;
}
