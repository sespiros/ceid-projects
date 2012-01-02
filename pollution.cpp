#include "pollution.h"

bool isDone(Pollution* p)
{
	bool ret = false;
	if (p->roundsRun >= p->lifespan) {
		delete p;
		p = 0;
		ret = true;
	}
	return ret;
}

Pollution::Pollution(int r, int x, int y, int ls)
{
	radius = r;
	Pollution::x = x;
	Pollution::y = y;
	lifespan = ls;
	roundsRun = 0;
}

void Pollution::tick()
{
	roundsRun++;
}

int Pollution::getRadius() const
{
	return radius;
}

int Pollution::getX() const
{
	return x;
}

int Pollution::getY() const
{
	return y;
}
