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
    static bool logChoice;
};
bool IScreen::logChoice = false;
#endif // SCREEN_H
