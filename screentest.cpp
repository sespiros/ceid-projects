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

    ZPlankton     a(15, 10);
    PPlankton     b(47+5, 10);
    Eel           c(81+10, 10);
    Shrimp        d(113+15, 10);
    Jelly         e(145+20, 10);
    Balloon       f(177+25, 10);
    Clown         g(209+30, 10);
    Gtp           h(241+35, 10);
    Magikarp      i(273+40, 10);
    Narwhal         j(780, 10);
    ZPlankton     k(15, 480);
    PPlankton     l(47+5, 480);
    Eel           m(81+10, 480);
    Shrimp        n(113+15, 480);
    Jelly         o(145+20, 480);
    Balloon       p(177+25, 480);
    Clown         q(209+30, 480);
    Gtp           r(241+35, 480);
    Magikarp      s(273+40, 480);
    Narwhal         t(780, 480);

    Ocean::add(&a);
    Ocean::add(&b);
    Ocean::add(&c);
    Ocean::add(&d);
    Ocean::add(&e);
    Ocean::add(&f);
    Ocean::add(&g);
    Ocean::add(&h);
    Ocean::add(&i);
    Ocean::add(&j);
    Ocean::add(&k);
    Ocean::add(&l);
    Ocean::add(&m);
    Ocean::add(&n);
    Ocean::add(&o);
    Ocean::add(&p);
    Ocean::add(&q);
    Ocean::add(&r);
    Ocean::add(&s);
    Ocean::add(&t);



    Ocean::info();

    Ocean::update();
    Ocean::info();

    // std::cout << "Deleting a fish...\n"<<std::endl;

    //Ocean::kill(0);
    // Ocean::info();

    //Vector2f **asd;
    //asd=ArrayCon(800,600,35,35);

    //Main loop
    while (screen >= 0)
    {
        screen = Screens[screen]->Run(App);
    }

    return EXIT_SUCCESS;
}
