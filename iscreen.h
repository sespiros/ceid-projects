#ifndef SCREEN_H
#define SCREEN_H
#include "SFML/Graphics.hpp"

/**
 * A Screen Interface.
 */
class IScreen
{
public:
    virtual int Run (sf::RenderWindow &App) = 0;
};
#endif // SCREEN_H
