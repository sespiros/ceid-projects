#include "helper.h"
#include "screen.h"
#include "about.h"
#include "run.h"

using namespace sf;

int main(int argc, char** argv)
{
	Ocean::init();

	sf::RenderWindow App(sf::VideoMode(1024, 600, 32), "Ocean Life v0.1");

	std::vector<cScreen *> Screens;
	about s0;
	Screens.push_back (&s0);
	run s1;
	Screens.push_back (&s1);
	//Screen Manager main loop
	int screen = 0 ;
    while (screen >= 0)
    {
		screen = Screens[screen]->Run(App);
    }

    return EXIT_SUCCESS;
}
