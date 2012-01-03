#ifndef OCEAN_H
#define OCEAN_H

#include <vector>

#include "organism.h"
#include "pollution.h"

class Ocean {
public:
    static const int MAX_COUNT = 100;
    static const int MAX_X = 16;
    static const int MAX_Y = 26;

    static int count;

    static void add(Organism *toAdd);
    static void kill(int key);
	static std::map<int, Organism*>::iterator move(int key, int x, int y);
    static void createAndAddFish(int t, int x, int y);

	static void pollute(int, int, int);

    static void createAndAddRandFish(int x, int y);
    static void populate();
	static std::map<int, Organism*>::iterator collide(int key);
    static Organism::fishtype genRandType();

    static void init();
	static void tickPollution();
    static void update();
    static void info();
	static bool isValid(int, int);

    Ocean();

    static std::map<int, Organism*> fishMap;
private:
	static std::vector<Pollution*> pollution;
};

#endif // OCEAN_H
