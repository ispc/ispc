// RUN: %{ispc} --target=avx2-i64x4,sse4 --nostdlib --nowrap -o %t.ll --emit-llvm-text %s 2>&1
// REQUIRES: X86_ENABLED

// CHECK-NOT: FATAL ERROR:
void MyFunc(uniform int x) {
    print("Hello %\n", x);
}

struct MyObject {
    void (* MyFuncPtr)(uniform int);
};

export void TestMain()
{
    MyObject obj;
    obj.MyFuncPtr = MyFunc;

    uniform int x = 60;
    obj.MyFuncPtr(x);
}

export void TestMainBroken()
{
    uniform MyObject obj; // The only difference
    obj.MyFuncPtr = MyFunc; 

    uniform int x = 60;
    obj.MyFuncPtr(x);
}
