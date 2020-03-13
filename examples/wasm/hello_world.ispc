
export void hello_from_ispc(uniform int a[], uniform int b[], uniform int c[], uniform int N) {
  print ("hello world from ISPC!\n");
  {
    foreach (i = 0 ... N) {
      if (i % 2 == 0)
        print ("a[%] = %\n", i, a[i]);
    }
  }
}