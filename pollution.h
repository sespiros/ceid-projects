#ifndef POLLUTION_H
#define POLLUTION_H

#include "SFML/Graphics.hpp"

class Pollution
{
public:
	Pollution();
	Pollution(int, int, int, int = 8, int = 2);

	static void bind(sf::RenderWindow*, bool = false);

	void tick();
	void draw();

	// Utility function as a predicate for the erase-remove idiom
	static bool isDone(Pollution *);

	int getRadius() const;
	int getX() const;
	int getY() const;
private:
	int radius;
	const int maxRadius;
	int width;
	int x, y;
	int roundsRun;
	int lifespan;
	sf::Image img;
	sf::Sprite sprite;
	static sf::RenderWindow* window;
	static bool isPaused_;
};

#endif // POLLUTION_H
