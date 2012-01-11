// Case statement value must be a compile-time integer constant

void foo(float f) {
    switch (f) {
    case 1.5:
        ++f;
    }
}

