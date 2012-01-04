#include <iostream>
#include "iscreen.h"
#include "organism.h"
#include "Ocean.h"

class PauseScreen : public IScreen
{
private:
    bool playing;
public:
    PauseScreen (void);
    virtual int Run (sf::RenderWindow &App);
};


PauseScreen::PauseScreen(void){
    playing=false;
}

int PauseScreen::Run(sf::RenderWindow &App)
{
    sf::Event Event;
    bool Running = true;
    sf::Image pauseImage;
    sf::Sprite pauseSprite;
    sf::Image back;
    sf::Sprite backSprite;
    sf::Image runImage;
    sf::Sprite runSprite;
    sf::Vector2f MousePos;
    int zoom = 0;

    sf::View view;
    if(Ocean::worldIsBig){
        view.SetFromRect(sf::FloatRect(0,0,1200,700));
        view.Zoom(0.5);
        view.Move(600.0f,320.0f);
        if (!back.LoadFromFile("artwork/big.jpg")){
            std::cerr<<"Error loading background image"<<std::endl;
            return(-1);
        }

        backSprite.SetImage(back);

    }else{
        view.SetFromRect(sf::FloatRect(2,-5.66833,1190.67,690.782));
        view.Zoom(0.97);
        if (!back.LoadFromFile("artwork/small.jpg")){
            std::cerr<<"Error loading background image"<<std::endl;
            return(-1);
        }

        backSprite.SetImage(back);

    }

    if (!pauseImage.LoadFromFile("artwork/pause.png")){
        std::cerr<<"Error loading background image"<<std::endl;
        return(-1);
    }

    pauseSprite.SetImage(pauseImage);

    if (!runImage.LoadFromFile("artwork/run.png")){
        std::cerr<<"Error loading background image"<<std::endl;
        return(-1);
    }
    runSprite.SetImage(runImage);

    while (Running)
    {
        while(App.GetEvent(Event))
        {
            MousePos = App.ConvertCoords(App.GetInput().GetMouseX(), App.GetInput().GetMouseY());
            if (Event.Type == sf::Event::Closed)
            {
                return (-1);
            }
            if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::Return){
                playing = true;
                return(1);
            }
            if (Event.Type == sf::Event::MouseButtonPressed && Event.MouseButton.Button == sf::Mouse::Left){
                if(MousePos.x > 15 && MousePos.x < 818 && MousePos.y > 15 && MousePos.y < 512){
                    playing = true;
                    return(1);
                }
            }
            if (Event.Type == sf::Event::MouseButtonPressed && Event.MouseButton.Button == sf::Mouse::Left){
                std::cout << App.GetInput().GetMouseX()<<" "<<App.GetInput().GetMouseY()<<std::endl;
                if(MousePos.x > 826 && MousePos.x < 1015 && MousePos.y > 558 && MousePos.y < 594){
                    IScreen::logChoice = !IScreen::logChoice;
                }
            }
            //////////////////////////////////////////////////
            if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::W){
                if(zoom < 2){
                    view.Zoom(2.0f);
                    zoom++;
                }

            }
            if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::S){
                if(zoom >= 0){//temporary for debugging
                    view.Zoom(0.5f);
                    zoom--;
                }
            }

            if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::Up){
                view.Move(0, 10.0f);
            }
            if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::Down){
                view.Move(0, -10.0f);
            }
            if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::Left){
                view.Move(-10.0f, 0);
            }
            if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::Right){
                view.Move(10.0f, 0);
            }

            //////////////////////////////////////////////////
        }

        App.SetView(view);

        App.Draw(backSprite);
        std::map<int, Organism*>::iterator it;
        for(it = Ocean::fishMap.begin();it != Ocean::fishMap.end(); it++){
            App.Draw(it->second->sprite);
        }

        Pollution::bind(&App);
        std::for_each(Ocean::pollution.begin(), Ocean::pollution.end(), std::mem_fun(&Pollution::draw));
        Pollution::bind(0);

        App.SetView(App.GetDefaultView());
        App.Draw(runSprite);
        App.Draw(pauseSprite);

        //Draw stats box
        Ocean::drawStats(&App, IScreen::logChoice, 0);

        App.Display();
        App.Clear();
    }
    return(-1);
}
