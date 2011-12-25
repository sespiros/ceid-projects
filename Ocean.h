#ifndef OCEAN_H
#define OCEAN_H

#include <vector>
#include "organism.h"

class Ocean {
public:
	static const int MAX_COUNT = 20;
	static const int MAX_X = 15;
	static const int MAX_Y = 26;

    static int count;

    static void add(Organism *toAdd);
    static void kill(int idx);
    static void createAndAddFish(int t, int x, int y);
    static void populate();

	static void init();
    static void update();
    static void move(Organism&);
    static void info();

    Ocean();

//private:
    /* Can't use Organism as the vector type;
     * the vector assignment process tries to
     * write on our const members
     */
    static std::vector<Organism *> fish;
};

#endif // OCEAN_H
