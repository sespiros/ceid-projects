#include "Ocean.h"
#include "organism.h"
#include "ClassRegistry.h"
#include <vector>
#include <iostream>
#include <cstdlib>
#include <ctime>

#define MAX_WIDTH 15
#define MAX_HEIGHT 26

/*
 * static declarations
 */
int Ocean::count = 0;
std::vector<Organism *> Ocean::fish;

Ocean::Ocean() {
}

void Ocean::add(Organism *toAdd) {
    Ocean::fish.push_back(toAdd);
    count++;
}

void Ocean::kill(int idx) {
    Ocean::fish.erase(Ocean::fish.begin()+idx);
    count--;
}

void Ocean::createAndAddFish(int t, int x, int y) {
    organism_creator f = ClassRegistry::getConstructor(t);

    Ocean::add(f(x, y));
}

void Ocean::move(Organism& fish) {
    int curX = fish.getX();
    int curY = fish.getY();
    std::cout<<curX<<" "<<curY<<std::endl;

    int dx, dy;

    do {
        dx = rand()%3 - 1;
        dy = rand()%3 - 1;
    } while (((curX + dx < 0) || (curX + dx >= MAX_WIDTH)) || ((curY + dy < 0) || (curY + dy >= MAX_HEIGHT)));

    fish.setX(curX + dx);
    fish.setY(curY + dy);
}

void Ocean::update() {
    srand(time(0));
    for (unsigned int i = 0; i < fish.size(); i++) {
        move(*fish.at(i));
    }
}

void Ocean::info() {
    std::cout << "***************\n";
    std::cout << Ocean::count << " fish.\n";
    for (unsigned int i = 0; i < fish.size(); i++){
        std::cout << "Fish " << i << " is of type " << (fish.at(i))->getType() << std::endl;
        std::cout << "Position: " << (fish.at(i))->getX() << " " << (fish.at(i))->getY() << std::endl;
    }
    std::cout << "***************\n";
}
