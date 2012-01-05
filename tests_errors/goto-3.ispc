// "goto" statements are only legal under "uniform" control flow

void func(int x) {
    cif (x < 0)
        goto label;

 label:
        ;
}


