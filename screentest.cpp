#include "helper.h" //includes everything
#include "screens.h"

using namespace sf;

int main(int argc, char** argv)
{
    //Window creation
    sf::RenderWindow App(sf::VideoMode(1024, 600, 32), "SFML Demo 3");

    //Register Classses in registry
    ClassRegistry::registerClasses();

    //Create 4 magikarps
    srand (time(NULL));
    for(int i=0;i<20;i++){
        int x = rand()%16;      //x is the j coordinate of the worldToPixel equals 16 places
        int y = rand()%27;      //y is the i coordinate of the worldToPixel equals 27 places
        int num = rand()%10;
        Ocean::createAndAddFish(num,x,y);
    }

    //Screen Manager Initialization
    int screen = 0 ;
    std::vector<cScreen*> Screens;
    about s0;
    Screens.push_back (&s0);
    run s1;
    Screens.push_back (&s1);

    //Screen Manager main loop
    while (screen >= 0)
    {
        screen = Screens[screen]->Run(App);
    }

    return EXIT_SUCCESS;
}
