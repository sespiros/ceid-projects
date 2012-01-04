#ifndef OCEAN_H
#define OCEAN_H

#include <vector>

#include "organism.h"
#include "pollution.h"

typedef std::map<int, Organism*>::iterator mapIter;

class Ocean {
public:
    static const int MAX_COUNT = 100;
    static const int MAX_X = 16;
    static const int MAX_Y = 26;

    static int count;
    static int deaths;
    static int turns;

    static void add(Organism *toAdd);
    static void kill(int key);
	static mapIter move(int key, int x, int y);
    static void createAndAddFish(int t, int x, int y);

	static void pollute(int, int, int);

    static void createAndAddRandFish(int x, int y);
    static void populate();
	static mapIter collide(int key);
    static Organism::fishtype genRandType();

    static void init();
	static void tickPollution();
    static void update();
    static void info();
	static bool isValid(int, int);

    static void stats();
    static void drawStats(sf::RenderWindow*, bool , bool);

    Ocean();

	static std::map<int, Organism*> fishMap;
	static std::vector<Pollution*> pollution;
    static float averageCategorySize[10];
    static float averageConsumptionWeek[10];
    static float averageDeathRate[10];
    static float averageAge[10];
    static sf::String categories[10];
    static sf::Sprite Images[10];

};

#endif // OCEAN_H
