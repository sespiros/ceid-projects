#ifndef OCEAN_H
#define OCEAN_H

#include <vector>
#include "organism.h"

class Ocean {
public:
    static const int MAX_COUNT = 100;
    static const int MAX_X = 16;
    static const int MAX_Y = 26;

    static int count;

    static void add(Organism *toAdd);
    static void kill(int key);
    static void move(int key, int x, int y);
    static void createAndAddFish(int t, int x, int y);

    static void createAndAddRandFish(int x, int y);
    static void populate();
    static void collide(int key);
    static Organism::fishtype genRandType();

    static void init();
    static void update();
    static void info();

    Ocean();

    static std::map<int, Organism*> fishMap;
};

#endif // OCEAN_H
