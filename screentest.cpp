#include "helpers.h" //includes everything


int main(int argc, char** argv)
{
    //Applications variables
    std::vector<cScreen*> Screens;
    int screen = 0;

    //Window creation
    sf::RenderWindow App(sf::VideoMode(1024, 600, 32), "SFML Demo 3");

    //Screens preparations
    about s0;
    Screens.push_back (&s0);
    run s1;
    Screens.push_back (&s1);

    srand ( time(NULL) );

    Vector2f **asd;
    asd=ArrayCon(1024,600,32,32);

    ClassRegistry::associate<ZPlankton>(Organism::ZPL);
    ClassRegistry::associate<PPlankton>(Organism::PPL);
    ClassRegistry::associate<Shrimp>(Organism::SHRIMP);
    ClassRegistry::associate<Jelly>(Organism::JELLY);
    ClassRegistry::associate<Eel>(Organism::EEL);
    ClassRegistry::associate<Balloon>(Organism::BALLOON);
    ClassRegistry::associate<Clown>(Organism::CLOWN);
    ClassRegistry::associate<Gtp>(Organism::GTP);
    ClassRegistry::associate<Magikarp>(Organism::MAGIKARP);
    ClassRegistry::associate<Narwhal>(Organism::NARWHAL);

    for(int i=0;i<160;i++){
        int b = rand()%16;
        int c = rand()%27;
        Organism* a = ClassRegistry::getConstructor(rand()%10)(asd[b][c].x,asd[b][c].y);
        Ocean::add(a);
    }

    Ocean::info();

    Ocean::update();
    Ocean::info();

    std::cout << "Deleting a fish...\n"<<std::endl;

    Ocean::kill(0);
    Ocean::info();



    //Main loop
    while (screen >= 0)
    {
        screen = Screens[screen]->Run(App);
    }

    return EXIT_SUCCESS;
}
