// Left hand side of assignment expression can't be assigned to

int foo() {return 2;}

int bar()
{
    foo() = 2;
}
