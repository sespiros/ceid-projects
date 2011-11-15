#include "organism.h"
#include "Ocean.h"
#include <iostream>

using namespace std;

int main() {
    Gtp g(2, 2);
    Eel c(5, 5);

    Ocean::add(&g);
    Ocean::add(&c);
    Ocean::info();

	Ocean::update();
	Ocean::info();

    std::cout << "Deleting a fish...\n";

    Ocean::kill(0);
	Ocean::info();
    return 0;
}
