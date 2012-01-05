#include "helper.h"
#include "iscreen.h"
#include "pause.h"
#include "run.h"

int main()
{
    // Ocean Initialization select true for BigWorld
	Ocean::init(false);

	sf::RenderWindow App(sf::VideoMode(1024, 600, 32), "Ocean Life v0.2");
    std::vector<IScreen *> Screens;

    PauseScreen pause;
    RunScreen run;

    Screens.push_back(&pause);
    Screens.push_back(&run);

    int curScreen = 0;

    while (curScreen >= 0)
        curScreen = Screens[curScreen]->Run(App);

	Helper::cleanup();

    return EXIT_SUCCESS;
}
