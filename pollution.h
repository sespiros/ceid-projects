#ifndef POLLUTION_H
#define POLLUTION_H

class Pollution
{
public:
	Pollution();
	Pollution(int, int, int, int = 3);

	void tick();

	int getRadius() const;
	int getX() const;
	int getY() const;

	friend bool isDone(Pollution *);
private:
	int radius;
	int x, y;
	int roundsRun;
	int lifespan;
};

// Utility function as a predicate for the erase-remove idiom
bool isDone(Pollution *);

#endif // POLLUTION_H
