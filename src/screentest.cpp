#include "helper.h"
#include "iscreen.h"
#include "pause.h"
#include "run.h"
#include "menu.h"

int main()
{
    // Ocean Initialization select true for BigWorld
    Ocean::setup();

    sf::RenderWindow App(sf::VideoMode(1024, 600, 32), "Ocean Life v0.2");
    std::vector<IScreen *> Screens;

    PauseScreen pause;
    RunScreen run;
    MenuScreen menu;
    Screens.push_back(&pause);
    Screens.push_back(&run);
    Screens.push_back(&menu);

    int curScreen = 2;

    while (curScreen >= 0)
        curScreen = Screens[curScreen]->Run(App);

    Helper::cleanup();
    Helper::worldToPixel = NULL;

    return EXIT_SUCCESS;
}
