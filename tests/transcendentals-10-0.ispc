// rule: run on OS=linux
// sincos() function is available only on Linux.

export uniform int width() { return programCount; }

export void f_du(uniform float RET[], uniform double aFOO[], uniform double b) {
    double a = aFOO[programIndex];
    uniform double sin_a[] = {0.8414709848078965, 0.9092974268256817, 0.1411200080598672, -0.7568024953079282, -0.9589242746631385,
                             -0.27941549819892586, 0.6569865987187891, 0.9893582466233818, 0.4121184852417566, -0.5440211108893698,
                             -0.9999902065507035, -0.5365729180004349, 0.4201670368266409, 0.9906073556948704, 0.6502878401571168,
                             -0.2879033166650653};
    uniform double cos_a[] = {0.5403023058681398, -0.4161468365471424, -0.9899924966004454, -0.6536436208636119, 0.28366218546322625,
                              0.960170286650366, 0.7539022543433046, -0.14550003380861354, -0.9111302618846769, -0.8390715290764524,
                              0.004425697988050785, 0.8438539587324921, 0.9074467814501962, 0.1367372182078336, -0.7596879128588213,
                             -0.9576594803233847};

    RET[programIndex]  = abs(sin(a) - sin_a[programIndex]) < 1d-7 ? 0 : 1;
    RET[programIndex] += abs(cos(a) - cos_a[programIndex]) < 1d-7 ? 0 : 1;

    double sincos_s, sincos_c;
    sincos(a, &sincos_s, &sincos_c);

    RET[programIndex]  = abs(sincos_s - sin_a[programIndex]) < 1d-7 ? 0 : 1;
    RET[programIndex] += abs(sincos_c - cos_a[programIndex]) < 1d-7 ? 0 : 1;
}

export void result(uniform float RET[]) {
    RET[programCount] = 0;
}
