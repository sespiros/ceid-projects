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
    static float speed;
    static bool info;
};
bool IScreen::logChoice = false;
bool IScreen::info = false;
float IScreen::speed;
#endif // SCREEN_H
