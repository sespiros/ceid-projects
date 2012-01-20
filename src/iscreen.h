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
    static bool helpChoice;
    static float speed;
    static bool info;
    static int actionChoice;
};
bool IScreen::logChoice = false;
bool IScreen::helpChoice = false;
bool IScreen::info = false;
float IScreen::speed;
int IScreen::actionChoice = 0;
#endif // SCREEN_H
