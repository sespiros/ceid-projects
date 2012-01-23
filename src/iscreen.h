#ifndef SCREEN_H
#define SCREEN_H
#include "SFML/Graphics.hpp"
#include "SFML/Audio.hpp"
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

    static sf::Music introMusic;
};
//static variables declaration
bool IScreen::logChoice = false;
bool IScreen::helpChoice = false;
bool IScreen::info = false;
float IScreen::speed;
int IScreen::actionChoice = 0;
sf::Music IScreen::introMusic;

#endif // SCREEN_H
