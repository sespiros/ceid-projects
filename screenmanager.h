#ifndef SCREENMANAGER_H
#define SCREENMANAGER_H

#include "aboutscreen.h"
#include "runscreen.h"

class ScreenManager {
public:
	static void init();
	static void run();
private:
	static std::vector<IScreen *> Screens;
	static sf::RenderWindow App;
};

#endif // SCREENMANAGER_H
