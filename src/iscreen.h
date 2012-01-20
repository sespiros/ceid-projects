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

    //show log or not between states
    static bool logChoice;

    //show help or not between states
    static bool helpChoice;

    //show info or not
    static bool info;

    //keeps slider speed between states
    static float speed;

    //to chosee between pollution or nets
    static int actionChoice;
};
//static variables declaration
bool IScreen::logChoice = false;
bool IScreen::helpChoice = false;
bool IScreen::info = false;
float IScreen::speed;
int IScreen::actionChoice = 0;
#endif // SCREEN_H
