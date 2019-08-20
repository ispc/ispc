// RUN: %{ispc} %s -O0 --emit-llvm-text -o - | FileCheck %s

bool uniform_less( ) {
    uniform float f = sqrt(-1.0f);
    // CHECK: fcmp olt
    if (f < 0.0f)
        return false;
}

bool uniform_great( ) {
    uniform float f = sqrt(-1.0f);
    // CHECK: fcmp ogt
    if (f > 0.0f)
        return false;
}

bool uniform_LE( ) {
    uniform float f = sqrt(-1.0f);
    // CHECK: fcmp ole
    if (f <= 0.0f)
        return false;
}

bool uniform_GE( ) {
    uniform float f = sqrt(-1.0f);
    // CHECK: fcmp oge
    if (f >= 0.0f)
        return false;
}

bool uniform_EQ( ) {
    uniform float f = sqrt(-1.0f);
    // CHECK: fcmp oeq
    if (f == 0.0f)
        return false;
}

bool uniform_NE( ) {
    uniform float f = sqrt(-1.0f);
    // CHECK: fcmp une
    if (f != 0.0f)
        return false;
}

bool VAR_less( ) {
    float f = sqrt(-1.0f);
    // CHECK: fcmp olt
    if (f < 0.0f)
        return false;
}

bool VAR_great( ) {
    float f = sqrt(-1.0f);
    // CHECK: fcmp ogt
    if (f > 0.0f)
        return false;
}

bool VAR_LE( ) {
    float f = sqrt(-1.0f);
    // CHECK: fcmp ole
    if (f <= 0.0f)
        return false;
}

bool VAR_GE( ) {
    float f = sqrt(-1.0f);
    // CHECK: fcmp oge
    if (f >= 0.0f)
        return false;
}

bool VAR_EQ( ) {
    float f = sqrt(-1.0f);
    // CHECK: fcmp oeq
    if (f == 0.0f)
        return false;
}

bool VAR_NE( ) {
    float f = sqrt(-1.0f);
    // CHECK: fcmp une
    if (f != 0.0f)
        return false;
}
