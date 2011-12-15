// Left hand side of assignment statement can't be assigned to

int foo() {return 2;}

int bar()
{
    foo() = 2;
}
