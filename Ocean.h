#ifndef OCEAN_H
#define OCEAN_H

#include <vector>
#include "organism.h"

class Ocean {
public:
	static const int MAX_COUNT = 100;
	static const int MAX_X = 15;
	static const int MAX_Y = 26;

    static int count;

    static void add(Organism *toAdd);
    static void kill(int idx);
    static void createAndAddFish(int t, int x, int y);
	static void createAndAddRandFish(int x, int y);
    static void populate();

	static void init();
    static void update();
    static void move(Organism&);
    static void info();

	static Organism::fishtype genRandType();
    static Organism*** initMap(const int x, const int y, const int w, const int h);
    Ocean();

//private:
    /* Can't use Organism as the vector type;
     * the vector assignment process tries to
     * write on our const members
     */
    static std::vector<Organism *> fish;
    static Organism*** fishMap;
};

#endif // OCEAN_H
