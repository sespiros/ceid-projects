#ifndef ABOUTSCREEN_H
#define ABOUTSCREEN_H

#include <iostream>
#include "iscreen.h"
#include "organism.h"
#include "Ocean.h"

class AboutScreen : public IScreen
{
private:
    bool playing;
public:
	AboutScreen (void) {playing = false;}
    virtual int Run (sf::RenderWindow &App);
};

#endif
