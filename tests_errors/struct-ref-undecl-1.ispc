// Member operator "." can't be applied to declared but not defined struct type

struct Foo;

int bar(Foo & foo) { return foo.x; }
