#include "organism.h"
#include "Ocean.h"
#include <iostream>

int main() {
    Gtp g(2, 2);
    Eel c(5, 5);

    Ocean::create(&g);
    Ocean::create(&c);
    Ocean::info();

    std::cout << "Deleting a fish...\n";

    Ocean::kill(0);
    Ocean::info();

    return 0;
}
