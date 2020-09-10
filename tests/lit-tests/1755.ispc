//; RUN: %{ispc} %s -o %t.o --nowrap --target=avx2-i32x8 2>&1
void a(uniform bool aFOO) 
{ 
    bool t = {aFOO, false, false, false, false, false, false, false};
    
}
