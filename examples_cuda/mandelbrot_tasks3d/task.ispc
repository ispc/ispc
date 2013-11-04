extern uniform int foo1();
extern uniform int data[1024];


void foo()
{
  data[foo1()] = 0; //taskIndex;
}
