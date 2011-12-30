#include "Ocean.h"
#include "ClassRegistry.h"
#include <vector>
#include <iostream>
#include <cstdlib>
#include <ctime>

using  std::map;
using  std::pair;
/*
 * static declarations
 */
int Ocean::count = 0;
std::map<int, Organism*> Ocean::fishMap;

void Ocean::init() {
    ClassRegistry::registerClasses();
    Ocean::populate();
}

void Ocean::add(Organism *toAdd) {
    int hash = toAdd->getX() + toAdd->getY() * MAX_X;
    fishMap.insert(pair<int, Organism*>(hash, toAdd));
    count++;
}

void Ocean::kill(int key) {
    fishMap.erase(key);
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
    return (Organism::fishtype)0;
}

void Ocean::move(int key, int x, int y) {
    int hash = x + y * MAX_X;

    fishMap[key]->setX(x);
    fishMap[key]->setY(y);

    fishMap.insert(pair<int, Organism*>(hash, fishMap[key]));
    //not Ocean::kill because in Ocean::we move an object and we don't want count--
    fishMap.erase(key);
}

void Ocean::collide(int key){
    int curX = fishMap[key]->getX();
    int curY = fishMap[key]->getY();

    int dx, dy, x, y, hash;
    bool hasMoved = false;
    bool hasEat = false;
    bool hasBred = false;

    do {
        //optimize rand selection-------------------------------------------------------------------------------------------//
        do {                                                                                                                //
            dx = rand()%3 - 1;                                                                                              //
            dy = rand()%3 - 1;                                                                                              //
        } while (((curX + dx < 0) || (curX + dx >= Ocean::MAX_X)) || ((curY + dy < 0) || (curY + dy >= Ocean::MAX_Y)));     //
        //------------------------------------------------------------------------------------------------------------------//
        x = curX + dx;
        y = curY + dy;

        hash = x + y * MAX_X;
        if(fishMap.find(hash) != fishMap.end()){
            if(fishMap[hash]->getType() < fishMap[key]->getType()){
                hasEat = true;
            }else if(fishMap[hash]->getType() == fishMap[key]->getType()){
                hasBred = true;
            }
        }else{
            hasMoved = true;
        }
    }while(!hasEat && !hasBred && !hasMoved);

    if(hasEat){
        kill(hash);
        move(key, x, y);
        //add stats print
    }else if(hasBred){
        //to be added
        //add stats print
    }else{
        move(key, x, y);
        //add stats print
    }
}

void Ocean::update() {
    srand(time(0));
    map<int, Organism*>::const_iterator it;
    for (it = fishMap.begin(); it != fishMap.end(); it++) {
        collide(it->first);
    }
}

void Ocean::info() {
    std::cout << "***************\n";
    std::cout << Ocean::count << " fish.\n";
    int i=0;
    std::map<int, Organism*>::iterator it;
    for (it = fishMap.begin(); it != fishMap.end(); it++){
        std::cout << "Fish " << i++ << " is of type " << it->second->getType() << std::endl;
        std::cout << "Position: " << it->second->getX() << " " << it->second->getY() << std::endl;
    }
    std::cout << "***************\n";
}
