#include "Ocean.h"
#include "organism.h"

#include <vector>
#include <iostream>
#include <cstdlib>
#include <ctime>

#define MAX_WIDTH 32
#define MAX_HEIGHT 32

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

void Ocean::init(){


}

void Ocean::kill(int idx) {
    Ocean::fish.erase(Ocean::fish.begin()+idx);
    count--;
}

void Ocean::update() {
    srand(time(0));
    for (unsigned int i = 0; i < fish.size(); i++) {
        int dx, dy;
        int curX = (fish.at(i))->getX(), curY = (fish.at(i))->getY();
        do {
            dx = rand()%3 - 1;
            dy = rand()%3 - 1;
        } while ((curX + dx < 0) && (curX + dx >= MAX_WIDTH) && (curY + dy < 0) && (curY + dy >= MAX_HEIGHT));
        (fish.at(i))->setX(curX + dx);
        (fish.at(i))->setY(curY + dy);
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
