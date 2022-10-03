// RUN: %{ispc} %s > %t 2>&1

extern "C" int32 foo(int8* ptr) {
    int i = 0;
    while(*ptr != 0){
        i++;
    }
    return i;
}
