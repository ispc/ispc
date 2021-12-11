//; RUN: %{ispc} %s --target=host -o %t.o --nostdlib

// Make sure it compiles and doesn't fail during const folding in the front-end

bool f1() {
  return (programIndex % 2) ? true : false;
}

int8 f2() {
  return (programIndex % 2) ? (int8) -1 : (int8) -2;
}

uint8 f3() {
  return (programIndex % 2) ? (uint8) 3 : (uint8) 4;
}

int16 f4() {
  return (programIndex % 2) ? (int16) -5 : (int16) -6;
}

uint16 f5() {
  return (programIndex % 2) ? (uint16) 7 : (uint16) 8;
}

int32 f6() {
  return (programIndex % 2) ? -9 : -10;
}

uint32 f7() {
  return (programIndex % 2) ? (uint32) 11 : (uint32) 12;
}

int64 f9() {
  return (programIndex % 2) ? -13LL : -14LL;
}

uint64 f10() {
  return (programIndex % 2) ? 15ULL : 16ULL;
}

float16 f11() {
  return (programIndex % 2) ? 1.23f16 : 4.56f16;
}

float f12() {
  return (programIndex % 2) ? 1.23e+30f : 4.56e+30f;
}

double f13() {
  return (programIndex % 2) ? 1.23e+300d : 4.56e+300d;
}
