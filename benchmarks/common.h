#include <iostream>
#include <string>

// Set maximum alignment for existing ISPC targets.
#define ALIGNMENT 64

class Docs {
  public:
    Docs(std::string message) { std::cout << message << "\n"; }
};
