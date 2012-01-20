#ifndef OCEAN_H
#define OCEAN_H

#include <vector>

#include "organism.h"
#include "pollution.h"
#include <cstring>
#include <sstream>

typedef std::map<int, Organism*>::iterator mapIter;

class Ocean {
public:
    static int MAX_COUNT;
    static int MAX_X;
    static int MAX_Y;
    static int count;
    static int deaths;
    static int births;
    static int turns;
    static int choiceHash;

    static bool worldIsBig;
    static bool easterEggs;
    static bool choice;

    static void add(Organism *toAdd);
    static void kill(int key);
    static void pollute(int, int, int, int = 8);
    static void createAndAddFish(int t, int x, int y);
    static void createAndAddRandFish(int x, int y);
    static void populate();
    static void init(bool big);
    static void tickPollution();
    static void update();
    static void info();
    static void stats();
    static void drawStats(sf::RenderWindow*, bool , bool);
    static void regLog(std::string);
    static void setup();
    static void reset();

    static mapIter move(int key, int x, int y);
    static mapIter breed(int key1, int key2);
    static mapIter collide(int key);

    static Organism::fishtype genRandType();

    static bool isValid(int, int);

    Ocean();

    static std::map<int, Organism*> fishMap;
    static std::vector<Pollution*> pollution;
    static float averageCategorySize[10];
    static float averageConsumptionWeek[10];
    static float averageDeathRate[10];
    static float averageAge[10];
    static sf::String categories[10];
    static sf::Sprite Images[10];
    static sf::Font GlobalFont;

    static std::stringstream log;

};

#endif // OCEAN_H
