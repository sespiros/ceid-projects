#include "Ocean.h"
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
    Ocean::fishMap[toAdd->getX()][toAdd->getY()] = toAdd;
    count++;
}

void Ocean::kill(int idx) {
    Ocean::fishMap[(Ocean::fish[idx])->getX()][(Ocean::fish[idx])->getY()] = 0;
    Ocean::fish.erase(Ocean::fish.begin()+idx);
    count--;
}

void Ocean::createAndAddFish(int t, int x, int y) {
    organism_creator f = ClassRegistry::getConstructor(t);

    Ocean::add(f(x, y));
}

void Ocean::createAndAddRandFish(int x, int y) {
	organism_creator f = ClassRegistry::getConstructor(genRandType());

	Ocean::add(f(x, y));
}

void Ocean::populate() {
	for (int i = 0; i < Ocean::MAX_COUNT; i++) {
		int x = rand()%(Ocean::MAX_X - 1);
		int y = rand()%(Ocean::MAX_Y - 1);
		Ocean::createAndAddRandFish(x, y);
	}
}

Organism::fishtype Ocean::genRandType() {
	std::map<Organism::fishtype, int>::const_iterator it;
	int sum = 0;
	for (it = Organism::weightMap.begin(); it != Organism::weightMap.end(); it++) {
		sum += it->second;
	}

	int rnd = rand()%sum;
	for (it = Organism::weightMap.begin(); it != Organism::weightMap.end(); it++) {
		rnd -= it->second;
		if (rnd < 0)
			return it->first;
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

Organism*** Ocean::fishMap = Ocean::initMap(1024, 600, 32, 32);

Organism*** Ocean::initMap(const int x, const int y, const int w, const int h){
    int jtiles = x/(w+5);
    int itiles = y/(h+5);
    Organism*** tempMap;


    //buffer allocation
    tempMap = new Organism**[itiles];
    for (int i=0;i<itiles;++i)
        tempMap[i] = new Organism*[jtiles];

    for(int i; i<itiles;i++){
        for(int j; j<jtiles;j++){
            tempMap[i][j] = 0;
        }
    }

    return tempMap;
}
