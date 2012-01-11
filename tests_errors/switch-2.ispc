// Duplicate case value "1"

void foo(float f) {
    switch (f) {
    case 1:
        ++f;
    case 2:
    case 1:
        f = 0;
    }
}

