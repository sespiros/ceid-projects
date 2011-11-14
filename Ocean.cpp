#include "Ocean.h"
#include "organism.h"
#include <vector>
#include <iostream>

/*
* static declarations
*/
int Ocean::count = 0;
std::vector<Organism *> Ocean::fish;

Ocean::Ocean() {

}

void Ocean::create(Organism *toCreate) {
    Ocean::fish.push_back(toCreate);
    count++;
}

void Ocean::kill(int idx) {
    Ocean::fish.erase(Ocean::fish.begin()+idx);
    count--;
}

void Ocean::info() {
    std::cout << "***************\n";
    std::cout << Ocean::count << " fish.\n";
    for (unsigned int i = 0; i < fish.size(); i++)
	std::cout << "Fish " << i << " is of type " << (fish.at(i))->getType() << "\n";
    std::cout << "***************\n";
}
