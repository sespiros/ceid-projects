#ifndef RUNSCREEN_H
#define RUNSCREEN_H

#include <iostream>
#include "iscreen.h"
#include "organism.h"
#include "Ocean.h"

class RunScreen : public IScreen
{
private:
	bool playing;
public:
	RunScreen (void) {playing = false;}
	virtual int Run (sf::RenderWindow &App);
};

#endif
