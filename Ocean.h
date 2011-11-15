#ifndef OCEAN_H
#define OCEAN_H

#include <vector>
#include "organism.h"

class Ocean {
public:
    static int count;

	static void add(Organism *toAdd);
    static void kill(int idx);
    static void info();

    Ocean();

private:
    /* Can't use Organism as the vector type;
     * the vector assignment process tries to
     * write on our const members
     */
    static std::vector<Organism *> fish;
};

#endif // OCEAN_H
