#include "helper.h"
#include "iscreen.h"
#include "pause.h"
#include "run.h"

int main()
{
    // Ocean Initialization
    Ocean::init();

    sf::RenderWindow App(sf::VideoMode(1024, 600, 32), "Ocean Life v0.1");
    std::vector<IScreen *> Screens;

    PauseScreen pause;
    RunScreen run;

    Screens.push_back(&pause);
    Screens.push_back(&run);

    int curScreen = 0;

    while (curScreen >= 0)
        curScreen = Screens[curScreen]->Run(App);

    return EXIT_SUCCESS;
}
