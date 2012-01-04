#include <iostream>
#include <functional>
#include "iscreen.h"
#include "organism.h"
#include "Ocean.h"

class RunScreen : public IScreen
{
private:
    bool playing;
public:
    RunScreen (void);
    virtual int Run (sf::RenderWindow &App);
    //static int a;
    //static int b;
};

RunScreen::RunScreen(void){
    playing = false;
}

//int RunScreen::a = 0;
//int RunScreen::b = 0;

int RunScreen::Run(sf::RenderWindow &App)
{
    sf::Event Event;
    bool Running = true;
    sf::Image runImage;
    sf::Image back;
    sf::Sprite runSprite;
    sf::Sprite backSprite;
    sf::Clock runner;
    sf::Vector2f MousePos;

    //adjusted view
    sf::View view(sf::FloatRect(-11.7823,-5.66833,1190.67,690.782));

    //617 350 center
    //635 372 halfsize

    /* Drawable area (15,10)----------(780,10)
                        |                 |
                        |                 |
                        |                 |
                     (15,480)----------(780,480)
    */

    if (!runImage.LoadFromFile("artwork/run.png")){
        std::cerr<<"Error loading background image"<<std::endl;
        return(-1);
    }
    runSprite.SetImage(runImage);

    if (!back.LoadFromFile("artwork/test.jpg")){
        std::cerr<<"Error loading background image"<<std::endl;
        return(-1);
    }

    backSprite.SetImage(back);

    //App.Clear();

    float dt=1.f/5.f;          //change game rate
    float accumulator = 0.f;
    bool drawn= false;

    while (Running)
    {
        accumulator+=runner.GetElapsedTime();
        runner.Reset();

        while(accumulator>=dt){

            while(App.GetEvent(Event))
            {
                MousePos = App.ConvertCoords(App.GetInput().GetMouseX(), App.GetInput().GetMouseY());
                if (Event.Type == sf::Event::Closed)
                {
                    return (-1);
                }
                if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::Escape){
                    playing = true;
                    return(0);
                }
                if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::Space) {
                    Ocean::pollute(rand()%4 + 1, rand()%Ocean::MAX_X, rand()%Ocean::MAX_Y);
                }
                if (Event.Type == sf::Event::MouseButtonPressed && Event.MouseButton.Button == sf::Mouse::Left){
                    //std::cout << App.GetInput().GetMouseX()<<" "<<App.GetInput().GetMouseY()<<std::endl;
                    if(MousePos.x > 826 && MousePos.x < 1015 && MousePos.y > 558 && MousePos.y < 594){
                        IScreen::logChoice = !IScreen::logChoice;
                    }
                }
            }

            Ocean::update();

            accumulator-=dt;
            drawn=false;
            //b++;
            //std::cout<<"update= "<<b<<std::endl;
        }

        if(drawn){
            sf::Sleep(0.01);
        }
        else{
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

            //Draw stats box
            Ocean::drawStats(&App, IScreen::logChoice, 0);

            App.Display();
            App.Clear();

            drawn=true;
            //a++;

            //debug for placing view
            //std::cout<<view.GetRect().Right<<","<<view.GetRect().Top<<" "<<view.GetRect().Left<<","<<view.GetRect().Bottom<<std::endl;
            //std::cout <<"render= "<<a<<std::endl<<std::endl;
        }


    }

    return(-1);
}

