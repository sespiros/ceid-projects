#include "Ocean.h"
#include "organism.h"
#include "ClassRegistry.h"
#include <vector>
#include <iostream>
#include <cstdlib>
#include <ctime>

/*
 * static declarations
 */
int Ocean::count = 0;
std::vector<Organism *> Ocean::fish;

void Ocean::init() {
	ClassRegistry::registerClasses();
	Ocean::populate();
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

void Ocean::populate() {
	for (int i = 0; i < Ocean::MAX_COUNT; i++) {
		int x = rand()%(Ocean::MAX_X - 1);
		int y = rand()%(Ocean::MAX_Y - 1);
		int id = rand()%10;
		Ocean::createAndAddFish(id, x, y);
	}
}

void Ocean::move(Organism& fish) {
    int curX = fish.getX();
	int curY = fish.getY();

    int dx, dy;

    do {
        dx = rand()%3 - 1;
        dy = rand()%3 - 1;
	} while (((curX + dx < 0) || (curX + dx >= Ocean::MAX_X)) || ((curY + dy < 0) || (curY + dy >= Ocean::MAX_Y)));

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
