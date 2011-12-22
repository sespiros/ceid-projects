#ifndef SCREEN_H
#define SCREEN_H
#include "SFML/Graphics.hpp"
class cScreen
{
public:
    virtual int Run (sf::RenderWindow &App) = 0;
};
#endif // SCREEN_H
