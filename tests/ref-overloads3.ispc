#include "../test_static.isph"
int A01(varying float f){return 0;} int A01(varying float& f){return 1;} int A01(varying const float& f){return 2;} int A01(varying double f){return 3;} int A01(varying int f){return 4;}
                                    int A02(varying float& f){return 1;} int A02(varying const float& f){return 2;} int A02(varying double f){return 3;} int A02(varying int f){return 4;}
int A03(varying float f){return 0;}                                      int A03(varying const float& f){return 2;} int A03(varying double f){return 3;} int A03(varying int f){return 4;}
int A04(varying float f){return 0;} int A04(varying float& f){return 1;}                                            int A04(varying double f){return 3;} int A04(varying int f){return 4;}

int A07(varying float f){return 0;} int A07(varying float& f){return 1;} int A07(varying const float& f){return 2;} int A07(varying double f){return 3;} int A07(varying int f){return 4;}
int A08(varying float f){return 0;} int A08(varying float& f){return 1;} int A08(varying const float& f){return 2;} int A08(varying double f){return 3;} int A08(varying int f){return 4;}
int A09(varying float f){return 0;} int A09(varying float& f){return 1;} int A09(varying const float& f){return 2;} int A09(varying double f){return 3;} int A09(varying int f){return 4;}

int A01(uniform float f){return 5;} int A01(uniform float& f){return 6;} int A01(uniform const float& f){return 7;} int A01(uniform double f){return 8;} int A01(uniform int f){return 9;}
int A02(uniform float f){return 5;} int A02(uniform float& f){return 6;} int A02(uniform const float& f){return 7;} int A02(uniform double f){return 8;} int A02(uniform int f){return 9;}
int A03(uniform float f){return 5;} int A03(uniform float& f){return 6;} int A03(uniform const float& f){return 7;} int A03(uniform double f){return 8;} int A03(uniform int f){return 9;}
int A04(uniform float f){return 5;} int A04(uniform float& f){return 6;} int A04(uniform const float& f){return 7;} int A04(uniform double f){return 8;} int A04(uniform int f){return 9;}

                                    int A07(uniform float& f){return 6;} int A07(uniform const float& f){return 7;} int A07(uniform double f){return 8;} int A07(uniform int f){return 9;}
int A08(uniform float f){return 5;}                                      int A08(uniform const float& f){return 7;} int A08(uniform double f){return 8;} int A08(uniform int f){return 9;}
int A09(uniform float f){return 5;} int A09(uniform float& f){return 6;}                                            int A09(uniform double f){return 8;} int A09(uniform int f){return 9;}




task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    varying float vf = 5;
    varying const float vcf = 5;
    varying float& vfr = vf;
    varying const float& vcfr = vf;
    uniform float uf = 5;
    uniform const float ucf = 5;
    uniform float& ufr = uf;
    uniform const float& ucfr = uf;

    if (A01(vf)  != 0) {RET[programIndex] = 1; return;}
    if (A01(vcf) != 0) {RET[programIndex] = 2; return;}
    if (A01(vfr) != 1) {RET[programIndex] = 3; return;}
    if (A01(vcfr)!= 2) {RET[programIndex] = 4; return;}
    if (A01(uf)  != 5) {RET[programIndex] = 5; return;}
    if (A01(ucf) != 5) {RET[programIndex] = 6; return;}
    if (A01(ufr) != 6) {RET[programIndex] = 7; return;}
    if (A01(ucfr)!= 7) {RET[programIndex] = 8; return;}

    if (A02(vf)  != 1) {RET[programIndex] = 9; return;}
    if (A02(vcf) != 2) {RET[programIndex] = 10; return;}
    if (A02(vfr) != 1) {RET[programIndex] = 11; return;}
    if (A02(vcfr)!= 2) {RET[programIndex] = 12; return;}
    if (A02(uf)  != 5) {RET[programIndex] = 13; return;}
    if (A02(ucf) != 5) {RET[programIndex] = 14; return;}
    if (A02(ufr) != 6) {RET[programIndex] = 15; return;}
    if (A02(ucfr)!= 7) {RET[programIndex] = 16; return;}

    if (A03(vf)  != 0) {RET[programIndex] = 17; return;}
    if (A03(vcf) != 0) {RET[programIndex] = 18; return;}
    if (A03(vfr) != 0) {RET[programIndex] = 19; return;}
    if (A03(vcfr)!= 2) {RET[programIndex] = 20; return;}
    if (A03(uf)  != 5) {RET[programIndex] = 21; return;}
    if (A03(ucf) != 5) {RET[programIndex] = 22; return;}
    if (A03(ufr) != 6) {RET[programIndex] = 23; return;}
    if (A03(ucfr)!= 7) {RET[programIndex] = 24; return;}

    if (A04(vf)  != 0) {RET[programIndex] = 25; return;}
    if (A04(vcf) != 0) {RET[programIndex] = 26; return;}
    if (A04(vfr) != 1) {RET[programIndex] = 27; return;}
    if (A04(vcfr)!= 0) {RET[programIndex] = 28; return;}
    if (A04(uf)  != 5) {RET[programIndex] = 29; return;}
    if (A04(ucf) != 5) {RET[programIndex] = 30; return;}
    if (A04(ufr) != 6) {RET[programIndex] = 31; return;}
    if (A04(ucfr)!= 7) {RET[programIndex] = 32; return;}

    if (A07(vf)  != 0) {RET[programIndex] = 33; return;}
    if (A07(vcf) != 0) {RET[programIndex] = 34; return;}
    if (A07(vfr) != 1) {RET[programIndex] = 35; return;}
    if (A07(vcfr)!= 2) {RET[programIndex] = 36; return;}
    if (A07(uf)  != 6) {RET[programIndex] = 37; return;}
    if (A07(ucf) != 7) {RET[programIndex] = 38; return;}
    if (A07(ufr) != 6) {RET[programIndex] = 39; return;}
    if (A07(ucfr)!= 7) {RET[programIndex] = 40; return;}

    if (A08(vf)  != 0) {RET[programIndex] = 41; return;}
    if (A08(vcf) != 0) {RET[programIndex] = 42; return;}
    if (A08(vfr) != 1) {RET[programIndex] = 43; return;}
    if (A08(vcfr)!= 2) {RET[programIndex] = 44; return;}
    if (A08(uf)  != 5) {RET[programIndex] = 45; return;}
    if (A08(ucf) != 5) {RET[programIndex] = 46; return;}
    if (A08(ufr) != 5) {RET[programIndex] = 47; return;}
    if (A08(ucfr)!= 7) {RET[programIndex] = 48; return;}

    if (A09(vf)  != 0) {RET[programIndex] = 49; return;}
    if (A09(vcf) != 0) {RET[programIndex] = 50; return;}
    if (A09(vfr) != 1) {RET[programIndex] = 51; return;}
    if (A09(vcfr)!= 2) {RET[programIndex] = 52; return;}
    if (A09(uf)  != 5) {RET[programIndex] = 53; return;}
    if (A09(ucf) != 5) {RET[programIndex] = 54; return;}
    if (A09(ufr) != 6) {RET[programIndex] = 55; return;}
    if (A09(ucfr)!= 5) {RET[programIndex] = 56; return;}

    RET[programIndex] = 0;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
