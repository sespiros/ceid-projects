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
Organism*** Ocean::fishMap;

void Ocean::init() {
    ClassRegistry::registerClasses();
    fishMap = Ocean::initMap(1024, 600, 32, 32);
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

//temporary move
void Ocean::move(Organism& fish, int x, int y) {
    fish.setX(x);
    fish.setY(y);
}

void Ocean::collide(int idx){
    int curX = Ocean::fish.at(idx)->getX();
    int curY = Ocean::fish.at(idx)->getY();

    int dx, dy, x, y;
    bool hasMoved = false;
    bool hasDied = false;
    bool hasEat = false;
    bool hasBred = false;

    do {
        //optimize rand selection
        do {
            dx = rand()%3 - 1;
            dy = rand()%3 - 1;
        } while (((curX + dx < 0) || (curX + dx >= Ocean::MAX_X)) || ((curY + dy < 0) || (curY + dy >= Ocean::MAX_Y)));

        x = curX + dx;
        y = curY + dy;

        if(fishMap[x][y] != (Organism*)0 && fishMap[curX][curY] != (Organism*)0){   //after && temporary fix cause of the vector has deleted entries
            if(fishMap[x][y]->getType() < fishMap[curX][curY]->getType()){
                hasEat = true;
            }else if(fishMap[x][y]->getType() > fishMap[curX][curY]->getType()){
                hasDied = true;
            }else{
                hasBred = true;
            }
        }else{
            hasMoved = true;
        }
    }while(!hasEat && !hasDied && !hasBred && !hasMoved);

    if(hasEat){
        move(*fish.at(idx), x, y); //change move first so only sets x and y and not generating
        //find a way to kill Ocean::fishMap[x][y] from Ocean::fish
        fishMap[x][y] = fish.at(idx);
        //add stats print
    }else if(hasDied){
        fishMap[curX][curY] = 0;
        kill(idx);
        //add stats print
    }
    else if(hasBred){

        //add stats print
    }else{
        move(*fish.at(idx), x, y); //change move first so only sets x and y and not generating
        fishMap[x][y] = fish.at(idx);
        //add stats print
    }

}

void Ocean::update() {
    srand(time(0));
    for (unsigned int i = 0; i < fish.size(); i++) {
        //move(*fish.at(i));
        collide(i);
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

Organism*** Ocean::initMap(const int x, const int y, const int w, const int h){
    int jtiles = x/(w+5);
    int itiles = y/(h+5);
    Organism*** tempMap;

    //buffer allocation
    tempMap = new Organism**[itiles];
    for (int i=0;i<itiles;++i)
        tempMap[i] = new Organism*[jtiles];

    for(int i = 0; i<itiles;i++){
        for(int j = 0; j<jtiles;j++){
            tempMap[i][j] = 0;
        }
    }

    return tempMap;
}
