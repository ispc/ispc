// Illegal to overload function by return type only 

float foo() {
    int x = { 2 };
}

int y = { 2 };

void foo() {
//CO    while (true)
//CO        ;
//CO    for (;;)
//CO        ;
    do ; while(1);
}

