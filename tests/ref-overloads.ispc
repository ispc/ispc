#include "../test_static.isph"
// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2
/////////////////////////VARYING
//int A011(varying float f){return 0;}  int A011(varying float& f){return 1;}       int A011(varying const float& f){return 2;}
//int A011(varying double f){return 3;} int A011(varying int f){return 4;}          int A011(uniform float f){return 5;}
//int A011(uniform float& f){return 6;} int A011(uniform const float& f){return 7;} int A011(uniform double f){return 8;}         int A011(uniform int f){return 9;}

int A013(varying const float& f){return 2;} int A013(varying double f){return 3;}  int A013(varying int f){return 4;}
int A013(uniform float f){return 5;} int A013(uniform float& f){return 6;}  int A013(uniform const float& f){return 7;}
int A013(uniform double f){return 8;} int A013(uniform int f){return 9;}

int A014(varying double f){return 3;}  int A014(varying int f){return 4;} 
int A014(uniform float f){return 5;} int A014(uniform float& f){return 6;}  int A014(uniform const float& f){return 7;} 
int A014(uniform double f){return 8;} int A014(uniform int f){return 9;}

int A015(varying int f){return 4;} int A015(uniform float f){return 5;}
int A015(uniform float& f){return 6;}  int A015(uniform const float& f){return 7;} 
int A015(uniform double f){return 8;} int A015(uniform int f){return 9;}

int A032(varying float& f){return 1;} int A032(varying double f){return 3;} int A032(varying int f){return 4;} 
int A032(uniform float f){return 5;}  int A032(uniform float& f){return 6;} int A032(uniform const float& f){return 7;}
int A032(uniform double f){return 8;} int A032(uniform int f){return 9;}

int A033(varying int f){return 4;}
int A033(uniform float f){return 5;}  int A033(uniform float& f){return 6;} int A033(uniform const float& f){return 7;}
int A033(uniform double f){return 8;} int A033(uniform int f){return 9;}

/////////////////////////UNIFORM

int A051(varying float f){return 0;}        int A051(varying float& f){return 1;} int A051(varying const float& f){return 2;}
int A051(varying double f){return 3;}       int A051(varying int f){return 4;}
int A051(uniform const float& f){return 7;} int A051(uniform double f){return 8;} int A051(uniform int f){return 9;}

int A052(varying float f){return 0;}        int A052(varying float& f){return 1;} int A052(varying const float& f){return 2;}
int A052(varying double f){return 3;}       int A052(varying int f){return 4;}
int A052(uniform double f){return 8;}       int A052(uniform int f){return 9;}

int A053(varying float f){return 0;}        int A053(varying float& f){return 1;} int A053(varying const float& f){return 2;}
int A053(varying double f){return 3;}       int A053(varying int f){return 4;}    int A053(uniform int f){return 9;} 

int A054(varying float& f){return 1;}       int A054(varying const float& f){return 2;}
int A054(varying double f){return 3;}       int A054(varying int f){return 4;}  int A054(uniform int f){return 9;}

int A055(varying float& f){return 1;}       int A055(varying const float& f){return 2;}
int A055(varying double f){return 3;}       int A055(varying int f){return 4;}

int A056(varying float& f){return 1;}       int A056(varying const float& f){return 2;} int A056(varying int f){return 4;}





task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    varying float vf = 5;
    varying const float vcf = 5;
    varying float& vfr = vf;
    varying const float& vcfr = vf;
    uniform float uf = 5;
    uniform const float ucf = 5;
    uniform float& ufr = uf;
    uniform const float& ucfr = uf;

    if (A013(vf) != 2) {RET[programIndex] = 1; return;}
    if (A014(vf) != 3) {RET[programIndex] = 2; return;}
    if (A015(vf) != 4) {RET[programIndex] = 3; return;}

    if (A013(vfr) != 2) {RET[programIndex] = 4; return;}
    if (A014(vfr) != 3) {RET[programIndex] = 5; return;}
    if (A015(vfr) != 4) {RET[programIndex] = 6; return;}

    if (A032(vcf) != 3) {RET[programIndex] = 7; return;}
    if (A033(vcf) != 4) {RET[programIndex] = 8; return;}

    if (A032(vcfr) != 3) {RET[programIndex] = 9; return;} 
    if (A033(vcfr) != 4) {RET[programIndex] = 10; return;} 

    if (A051(uf) != 7) {RET[programIndex] = 11; return;}
    if (A052(uf) != 8) {RET[programIndex] = 12; return;}
    if (A053(uf) != 0) {RET[programIndex] = 13; return;}
    if (A054(uf) != 9) {RET[programIndex] = 14; return;}
    if (A055(uf) != 3) {RET[programIndex] = 15; return;}
    if (A056(uf) != 4) {RET[programIndex] = 16; return;}

    if (A051(ufr) != 7) {RET[programIndex] = 17; return;}
    if (A052(ufr) != 8) {RET[programIndex] = 18; return;}
    if (A053(ufr) != 0) {RET[programIndex] = 19; return;}
    if (A054(ufr)  != 9) {RET[programIndex] = 20; return;}
    if (A055(ufr)  != 3) {RET[programIndex] = 21; return;}
    if (A056(ufr)  != 4) {RET[programIndex] = 22; return;}

   RET[programIndex] = 0;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
