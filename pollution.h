#ifndef POLLUTION_H
#define POLLUTION_H

#include "SFML/Graphics.hpp"

class Pollution
{
public:
	Pollution();
	Pollution(int, int, int, int = 3);

	static void bind(sf::RenderWindow*);

	void tick();
	void draw();

	int getRadius() const;
	int getX() const;
	int getY() const;

	friend bool isDone(Pollution *);
private:
	int radius;
	int x, y;
	int roundsRun;
	int lifespan;
	sf::Image img;
	sf::Sprite sprite;
	static sf::RenderWindow* window;
};

// Utility function as a predicate for the erase-remove idiom
bool isDone(Pollution *);

#endif // POLLUTION_H
