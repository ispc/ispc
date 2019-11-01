struct S {
  float x, y;
  int i;
  bool b;
};

uniform S us;
varying S vs;

export void f_v(uniform float RET[]) {
    uniform S uus = {1.0, 2.0, 5, true};
    us = uus;
    vs = uus;
    RET[programIndex] = vs.x + vs.y + (vs.b ? vs.i : 0);
}


export void result(uniform float RET[]) {
    RET[programIndex] = 8;
}
