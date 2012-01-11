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
int Ocean::MAX_X;
int Ocean::MAX_Y;

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
    float OffsetX, CenterX;
    float OffsetY, CenterY;

    int zoom = 0;

    sf::View view;
    if(Ocean::worldIsBig){
        view.SetFromRect(sf::FloatRect(-5,-5,800,500));
        view.Zoom(0.35f);
        //OffsetX = 722.0f;
        //OffsetY = 438.0f;
        //view.Move(OffsetX, OffsetY);
        CenterX = 1119.5;
        CenterY = 685.5;
        view.SetCenter(CenterX, CenterY);
    }else{
        view.SetFromRect(sf::FloatRect(-5,-5,800,500));
        view.Zoom(0.78f);
        //OffsetX = 97.0f;
        //OffsetY = 57.0f;
        //view.Move(OffsetX, OffsetY);
        CenterX = 500.5;
        CenterY = 300.5;
        view.SetCenter(CenterX, CenterY);
    }

    if (!back.LoadFromFile("artwork/back.jpg")){
        std::cerr<<"Error loading background image"<<std::endl;
        return(-1);
    }
    backSprite.SetImage(back);

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
                if(MousePos.x >= 14 && MousePos.x <= 818 && MousePos.y >= 14 && MousePos.y <= 512){
                    local = Helper::getLocalCoords(MousePosView.x,MousePosView.y);

                    int hash = local.x + local.y * Ocean::MAX_X;

                    if(Ocean::fishMap.find(hash) == Ocean::fishMap.end())
                    {
                        Ocean::choice = false;
                        Ocean::choiceHash = 0;
                    }else{
                        Ocean::choice = true;
                        Ocean::choiceHash = hash;
                    }
                }

            }
            if (Event.Type == sf::Event::MouseButtonPressed && Event.MouseButton.Button == sf::Mouse::Left){
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

        App.SetView(App.GetDefaultView());
        App.Draw(backSprite);

        App.SetView(view);

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
        Ocean::drawStats(&App, IScreen::logChoice, Ocean::choice);

        App.Display();
        App.Clear();
    }
    App.SetFramerateLimit(0);
    return(-1);
}
