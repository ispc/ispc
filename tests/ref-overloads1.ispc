void A1(float f)               {}
void A2(float& f)              {}
void A3(const float& f)        {} 
void A4(double f)              {}
void A5(int f)                 {}
void A6(uniform float f)       {}
void A7(uniform float& f)      {}
void A8(uniform const float& f){}
void A9(uniform double f)      {}
void A0(uniform int f)         {}






export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    varying float vf = 5;
    varying const float vcf = 5;
    varying float& vfr = vf;
    varying const float& vcfr = vf;
    uniform float uf = 5;
    uniform const float ucf = 5;
    uniform float& ufr = uf;
    uniform const float& ucfr = uf;

    A1(vf); A1(vcf); A1(vfr); A1(vcfr); A1(uf); A1(ucf); A1(ufr); A1(ucfr);
    A2(vf);          A2(vfr);
    A3(vf); A3(vcf); A3(vfr); A3(vcfr);
    A4(vf); A4(vcf); A4(vfr); A4(vcfr); A4(uf); A4(ucf); A4(ufr); A4(ucfr);
    A5(vf); A5(vcf); A5(vfr); A5(vcfr); A5(uf); A5(ucf); A5(ufr); A5(ucfr);
                                        A6(uf); A6(ucf); A6(ufr); A6(ucfr);
                                        A7(uf);          A7(ufr);
                                        A8(uf); A8(ucf); A8(ufr); A8(ucfr);
                                        A9(uf); A9(ucf); A9(ufr); A9(ucfr);
                                        A0(uf); A0(ucf); A0(ufr); A0(ucfr); 
    RET[programIndex] = 1;
}

export void result(uniform float RET[]) { RET[programIndex] = 1; }
