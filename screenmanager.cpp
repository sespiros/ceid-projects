#include "screenmanager.h"

sf::RenderWindow ScreenManager::App(sf::VideoMode(1024, 600, 32), "Ocean Life v0.1");
std::vector<IScreen *> ScreenManager::Screens;

void ScreenManager::init() {
	AboutScreen about;
	RunScreen run;

	Screens.push_back(&about);
	Screens.push_back(&run);
}

void ScreenManager::run() {
	int curScreen = 0;

	while (curScreen >= 0)
		curScreen = Screens[curScreen]->Run(App);
}
