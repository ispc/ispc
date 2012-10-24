// "goto" statements are only legal under "uniform" control flow

void func(int x) {
 label:

    for(int i =0 ;i<x;)
        goto label;
}


