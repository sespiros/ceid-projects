#include <iostream>
#include "screen.h"
#include "organism.h"
#include "Ocean.h"

class run : public cScreen
{
private:
    bool playing;
public:
    run (void);
    virtual int Run (sf::RenderWindow &App);
};

run::run(void)
{
    playing = false;
}

int run::Run(sf::RenderWindow &App)
{
    sf::Event Event;
    bool Running = true;
    sf::Image runImage;
    sf::Image back;
    sf::Sprite runSprite;
    sf::Sprite backSprite;

    sf::View view(sf::FloatRect(10, 5, 817, 517));


    /* Drawable area (15,10)----------(780,10)
                        |                 |
                        |                 |
                        |                 |
                     (15,480)----------(780,480)
    */
    //ZPlankton asdf(15,10);

    if (!runImage.LoadFromFile("C:/Users/Spiros/Qt Projects/ocean_life/artwork/run.png")){
        std::cerr<<"Error loading background image"<<std::endl;
        return(-1);
    }
    runSprite.SetImage(runImage);

    if (!back.LoadFromFile("C:/Users/Spiros/Qt Projects/ocean_life/artwork/test.jpg")){
        std::cerr<<"Error loading background image"<<std::endl;
        return(-1);
    }
    backSprite.SetImage(back);

    //App.Clear();

    while (Running)
    {
        while(App.GetEvent(Event))
        {
            if (Event.Type == sf::Event::Closed)
            {
                return (-1);
            }
            if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::Escape){
                playing = true;
                return(0);
            }
            if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::Down){
                Ocean::fish.at(0)->sprite.SetY((Ocean::fish.at(0))->sprite.GetPosition().y+37.0f);
            }
            if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::Up){
                Ocean::fish.at(0)->sprite.SetY((Ocean::fish.at(0))->sprite.GetPosition().y-37.0f);

            }
            if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::Left){
                Ocean::fish.at(0)->sprite.SetX((Ocean::fish.at(0))->sprite.GetPosition().x-37.0f);
            }
            if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::Right){
                Ocean::fish.at(0)->sprite.SetX((Ocean::fish.at(0))->sprite.GetPosition().x+37.0f);
            }
            if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::W){
                view.Zoom(1.050f);
            }
            if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::S){
                view.Zoom(0.950f);
            }
        }

        App.SetView(view);

        App.Draw(backSprite);
        for(unsigned int i=0;i<Ocean::fish.size();i++){
            App.Draw((Ocean::fish.at(i))->sprite);
        }

        App.SetView(App.GetDefaultView());
        App.Draw(runSprite);

        App.Display();
        //asdf.sprite.SetY(asdf.sprite.GetPosition().y+32.0f);

        App.Clear();
    }
    return(-1);
}
