#include <iostream>
#include "iscreen.h"
#include "organism.h"
#include "Ocean.h"
#include "helper.h"

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
    sf::Vector2f MousePosView;
    sf::Vector2i local;

    int zoom = 0;

    sf::View view;
    if(Ocean::worldIsBig){
        view.SetFromRect(sf::FloatRect(-5,-5,800,500));
        view.Zoom(0.35f);
        view.Move(722.0f,438.0f);
        if (!back.LoadFromFile("artwork/big.jpg")){
            std::cerr<<"Error loading background image"<<std::endl;
            return(-1);
        }
        backSprite.SetImage(back);
    }else{
        view.SetFromRect(sf::FloatRect(-5,-5,800,500));
        view.Zoom(0.81f);
        view.Move(87.f,49.f);
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
            MousePosView = App.ConvertCoords(App.GetInput().GetMouseX(),App.GetInput().GetMouseY(),&view);
            if (Event.Type == sf::Event::Closed)
            {
                return (-1);
            }
            if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::Return){
                playing = true;
                return(1);
            }
            if (Event.Type == sf::Event::MouseButtonPressed && Event.MouseButton.Button == sf::Mouse::Left){
                local = Helper::getLocalCoords(MousePosView.x,MousePosView.y);
                //debugging
                std::cout << local.x << " " << local.y << std::endl;

            }
            if (Event.Type == sf::Event::MouseButtonPressed && Event.MouseButton.Button == sf::Mouse::Left){
                if(MousePos.x > 826 && MousePos.x < 1015 && MousePos.y > 558 && MousePos.y < 594){
                    IScreen::logChoice = !IScreen::logChoice;
                }
            }
            //////////////////////////////////////////////////
            if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::W){
                if(zoom < 2){
                    view.Zoom(1.25f);
                    zoom++;
                }

            }
            if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::S){
                if(zoom > 0){
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
                view.Move(10.0f, 0);
            }
            if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::Right){
                view.Move(-10.0f, 0);
            }

            //////////////////////////////////////////////////
        }

        App.SetView(view);

        App.Draw(backSprite);
        std::map<int, Organism*>::iterator it;
        for(it = Ocean::fishMap.begin();it != Ocean::fishMap.end(); it++){
            App.Draw(it->second->sprite);
        }

        Pollution::bind(&App, true);
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
    App.SetFramerateLimit(0);
    return(-1);
}
