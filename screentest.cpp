#include "helper.h"
#include "iscreen.h"
#include "about.h"
#include "run.h"

int main(int argc, char** argv)
{
    // Ocean Initialization
    Ocean::init();

    sf::RenderWindow App(sf::VideoMode(1024, 600, 32), "Ocean Life v0.1");
    std::vector<IScreen *> Screens;

    AboutScreen about;
    RunScreen run;

    Screens.push_back(&about);
    Screens.push_back(&run);

    int curScreen = 0;

    while (curScreen >= 0)
        curScreen = Screens[curScreen]->Run(App);

    return EXIT_SUCCESS;
}
