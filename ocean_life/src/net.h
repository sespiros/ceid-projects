#ifndef NET_H
#define NET_H

#include "SFML/Graphics.hpp"

class Net
{
public:
	Net();
    Net(int, int, int, int = 6);

    static void bind(sf::RenderWindow*);

	void tick();
	void draw();

	// Utility function as a predicate for the erase-remove idiom
	static bool isDone(Net *);

	int getRadius() const;
	int getX() const;
	int getY() const;
    std::string getCount() const;
private:
    int radius;
	const int maxRadius;
	int x, y;
	int roundsRun;
	int lifespan;
    int count;
	sf::Image img;
	sf::Sprite sprite;
	static sf::RenderWindow* window;
	static bool isPaused_;
};

#endif // NET_H
