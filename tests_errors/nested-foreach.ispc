// Nested "foreach" statements are currently illegal

void func();

void foo(uniform int a, uniform int b) {
    foreach (i = 0 ... a) {
        foreach (j = 0 ... b) {
            func();
        }
    }
}
