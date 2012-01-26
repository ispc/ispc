// "goto" statements are only legal under "uniform" control flow

void func(int x) {
    if (x < 0)
        goto label;

 label:
        ;
}


