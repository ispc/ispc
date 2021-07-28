#include "../test_static.isph"
void A01(varying float f){} void A01(varying float& f){} void A01(varying const float& f){} void A01(varying double f){} void A01(varying int f){}
                            void A02(varying float& f){} void A02(varying const float& f){} void A02(varying double f){} void A02(varying int f){}
void A03(varying float f){}                              void A03(varying const float& f){} void A03(varying double f){} void A03(varying int f){}
void A04(varying float f){} void A04(varying float& f){}                                    void A04(varying double f){} void A04(varying int f){}
void A05(varying float f){} void A05(varying float& f){} void A05(varying const float& f){}                              void A05(varying int f){}
void A06(varying float f){} void A06(varying float& f){} void A06(varying const float& f){} void A06(varying double f){}

void A07(varying float f){} void A07(varying float& f){} void A07(varying const float& f){} void A07(varying double f){} void A07(varying int f){}
void A08(varying float f){} void A08(varying float& f){} void A08(varying const float& f){} void A08(varying double f){} void A08(varying int f){}
void A09(varying float f){} void A09(varying float& f){} void A09(varying const float& f){} void A09(varying double f){} void A09(varying int f){}
void A10(varying float f){} void A10(varying float& f){} void A10(varying const float& f){} void A10(varying double f){} void A10(varying int f){}
void A11(varying float f){} void A11(varying float& f){} void A11(varying const float& f){} void A11(varying double f){} void A11(varying int f){}

void A01(uniform float f){} void A01(uniform float& f){} void A01(uniform const float& f){} void A01(uniform double f){} void A01(uniform int f){}
void A02(uniform float f){} void A02(uniform float& f){} void A02(uniform const float& f){} void A02(uniform double f){} void A02(uniform int f){}
void A03(uniform float f){} void A03(uniform float& f){} void A03(uniform const float& f){} void A03(uniform double f){} void A03(uniform int f){}
void A04(uniform float f){} void A04(uniform float& f){} void A04(uniform const float& f){} void A04(uniform double f){} void A04(uniform int f){}
void A05(uniform float f){} void A05(uniform float& f){} void A05(uniform const float& f){} void A05(uniform double f){} void A05(uniform int f){}
void A06(uniform float f){} void A06(uniform float& f){} void A06(uniform const float& f){} void A06(uniform double f){} void A06(uniform int f){}

                            void A07(uniform float& f){} void A07(uniform const float& f){} void A07(uniform double f){} void A07(uniform int f){}
void A08(uniform float f){}                              void A08(uniform const float& f){} void A08(uniform double f){} void A08(uniform int f){}
void A09(uniform float f){} void A09(uniform float& f){}                                    void A09(uniform double f){} void A09(uniform int f){}
void A10(uniform float f){} void A10(uniform float& f){} void A10(uniform const float& f){}                              void A10(uniform int f){}
void A11(uniform float f){} void A11(uniform float& f){} void A11(uniform const float& f){} void A11(uniform double f){}




task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    varying float vf = 5;
    varying const float vcf = 5;
    varying float& vfr = vf;
    varying const float& vcfr = vf;
    uniform float uf = 5;
    uniform const float ucf = 5;
    uniform float& ufr = uf;
    uniform const float& ucfr = uf;

    A01(vf); A01(vcf); A01(vfr); A01(vcfr); A01(uf); A01(ucf); A01(ufr); A01(ucfr);
    A02(vf); A02(vcf); A02(vfr); A02(vcfr); A02(uf); A02(ucf); A02(ufr); A02(ucfr);
    A03(vf); A03(vcf); A03(vfr); A03(vcfr); A03(uf); A03(ucf); A03(ufr); A03(ucfr);
    A04(vf); A04(vcf); A04(vfr); A04(vcfr); A04(uf); A04(ucf); A04(ufr); A04(ucfr);
    A05(vf); A05(vcf); A05(vfr); A05(vcfr); A05(uf); A05(ucf); A05(ufr); A05(ucfr);
    A06(vf); A06(vcf); A06(vfr); A06(vcfr); A06(uf); A06(ucf); A06(ufr); A06(ucfr);
    A07(vf); A07(vcf); A07(vfr); A07(vcfr); A07(uf); A07(ucf); A07(ufr); A07(ucfr);
    A08(vf); A08(vcf); A08(vfr); A08(vcfr); A08(uf); A08(ucf); A08(ufr); A08(ucfr);
    A09(vf); A09(vcf); A09(vfr); A09(vcfr); A09(uf); A09(ucf); A09(ufr); A09(ucfr);
    A10(vf); A10(vcf); A10(vfr); A10(vcfr); A10(uf); A10(ucf); A10(ufr); A10(ucfr);
    RET[programIndex] = 1;
}

task void result(uniform float RET[]) { RET[programIndex] = 1; }
