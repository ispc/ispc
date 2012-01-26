// "case" label illegal outside of "switch" statement

void foo(float f) {
    switch (f) {
    case 1:
        ++f;
    case 2:
        f = 0;
    }
 case 3:
     --f;
}

