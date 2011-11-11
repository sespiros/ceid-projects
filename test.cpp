#include "organism.h"
#include <iostream>

int main() {
    Gtp g(2, 2);
    Clown c(5, 5);

    std::cout << Organism::global_count << std::endl;

    return 0;
}
