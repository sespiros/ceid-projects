#include <iostream>
#include <functional>
#include "iscreen.h"
#include "organism.h"
#include "Ocean.h"
#include "helper.h"

class RunScreen : public IScreen
{
private:
    bool playing;
public:
    RunScreen (void);
    virtual int Run (sf::RenderWindow &App);
    const float TICKS_PER_SECOND;
    const float SKIP_TICKS;
    const int MAX_FRAMESKIP;
};

RunScreen::RunScreen(void) : TICKS_PER_SECOND(1.2f), SKIP_TICKS(1.0f / TICKS_PER_SECOND), MAX_FRAMESKIP(10){
    playing = false;
}


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
    sf::Vector2f MousePosView;
    sf::Vector2i local;
    mapIter it;
    float OffsetX, moX, CenterX;
    float OffsetY, moY, CenterY;
    bool che = false;
    bool debug = true;

    int zoom = 0;
    int loops = 0;
    float next_game_tick = runner.GetElapsedTime();
    float curTime, lastTime, fps = 0;
    int frames = 0;

    sf::String lock;
    lock.SetColor(sf::Color::White);
    lock.SetSize(12);
    lock.SetFont(Ocean::GlobalFont);

    sf::Sprite tooltip;
    sf::Image   _tooltip;

    tooltip.SetColor(sf::Color::Black);

    sf::String tip;
    tip.SetColor(sf::Color::White);
    tip.SetSize(11);
    tip.SetFont(Ocean::GlobalFont);

    sf::Rect<float> addNew                  (831,39,875,560);
    sf::Rect<float> averageCategorySize     (871,39,900,560);
    sf::Rect<float> averageConsumptionWeek  (904,39,935,560);
    sf::Rect<float> averageDeathRate        (937,39,973,560);
    sf::Rect<float> averageAge              (976,39,1013,560);

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
        view.Zoom(0.79f);
        //OffsetX = 97.0f;
        //OffsetY = 57.0f;
        //view.Move(OffsetX, OffsetY);
        CenterX = 494.5;
        CenterY = 304.5;
        view.SetCenter(CenterX, CenterY);
    }

    if (!back.LoadFromFile("artwork/back.jpg")){
        std::cerr<<"Error loading background image"<<std::endl;
        return(-1);
    }
    backSprite.SetImage(back);

    if (!runImage.LoadFromFile("artwork/run.png")){
        std::cerr<<"Error loading background image"<<std::endl;
        return(-1);
    }
    runSprite.SetImage(runImage);

    lastTime = runner.GetElapsedTime();

    while (Running)
    {

        while(App.GetEvent(Event))
        {
            MousePos = App.ConvertCoords(App.GetInput().GetMouseX(), App.GetInput().GetMouseY());
            MousePosView = App.ConvertCoords(App.GetInput().GetMouseX(),App.GetInput().GetMouseY(),&view);
            sf::Rect<float> mouseRect(MousePos.x, MousePos.y, MousePos.x + 0.001, MousePos.y + 0.001);

            if (Event.Type == sf::Event::Closed)
            {
                return (-1); // window close
            }
            if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::Escape){
                playing = true;
                return(0); // state switch
            }
            if (Event.Type == sf::Event::MouseButtonPressed && Event.MouseButton.Button == sf::Mouse::Left){
                if(MousePos.x >= 14 && MousePos.x <= 818 && MousePos.y >= 14 && MousePos.y <= 565){
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
            if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::Space) {
                Ocean::pollute(rand()%5 + 1, rand()%Ocean::MAX_X, rand()%Ocean::MAX_Y, rand()%8 + 3);
            }
            if (Event.Type == sf::Event::MouseButtonPressed && Event.MouseButton.Button == sf::Mouse::Left){
                if(MousePos.x > 826 && MousePos.x < 1015 && MousePos.y > 576 && MousePos.y < 589){
                    IScreen::logChoice = !IScreen::logChoice;
                }
            }
            /////////////////////////////////////////////////////
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
            if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::C){
                che = !che;
                if (!che){
                    view.SetCenter(CenterX, CenterY);
                }
            }
            if (Event.Type == sf::Event::MouseButtonPressed && Event.MouseButton.Button == sf::Mouse::Left){
                std::cout<<MousePos.x<<" "<<MousePos.y<<std::endl;
            }
            if(addNew.Intersects(mouseRect)){
                tip.SetText("Drag organism to add");
                tip.SetPosition(MousePos.x - 140, MousePos.y - 10);
                tooltip.SetSubRect(sf::Rect<int>(tip.GetRect().Left - 5,tip.GetRect().Top - 5,tip.GetRect().Right + 5,tip.GetRect().Bottom));
                tooltip.SetPosition(MousePos.x - 145, MousePos.y - 10);
            }else{
                tip.SetText("");
                tooltip.SetPosition(1024, 600);//PATENTA
            }
            if(averageCategorySize.Intersects(mouseRect)){
                tip.SetText("average category size");
                tip.SetPosition(MousePos.x - 145, MousePos.y - 10);
                tooltip.SetSubRect(sf::Rect<int>(tip.GetRect().Left - 5,tip.GetRect().Top - 5,tip.GetRect().Right + 5,tip.GetRect().Bottom));
                tooltip.SetPosition(MousePos.x - 150, MousePos.y - 10);
            }
            if(averageConsumptionWeek.Intersects(mouseRect)){
                tip.SetText("consumption/week");
                tip.SetPosition(MousePos.x - 110, MousePos.y - 10);
                tooltip.SetSubRect(sf::Rect<int>(tip.GetRect().Left - 5,tip.GetRect().Top - 5,tip.GetRect().Right + 5,tip.GetRect().Bottom));
                tooltip.SetPosition(MousePos.x - 115, MousePos.y - 10);
            }
            if(averageDeathRate.Intersects(mouseRect)){
                tip.SetText("death rate");
                tip.SetPosition(MousePos.x - 70, MousePos.y - 10);
                tooltip.SetSubRect(sf::Rect<int>(tip.GetRect().Left - 5,tip.GetRect().Top - 5,tip.GetRect().Right + 5,tip.GetRect().Bottom));
                tooltip.SetPosition(MousePos.x - 75, MousePos.y - 10);
            }
            if(averageAge.Intersects(mouseRect)){
                tip.SetText("average age");
                tip.SetPosition(MousePos.x - 80, MousePos.y - 10);
                tooltip.SetSubRect(sf::Rect<int>(tip.GetRect().Left - 5,tip.GetRect().Top - 5,tip.GetRect().Right + 5,tip.GetRect().Bottom));
                tooltip.SetPosition(MousePos.x - 85, MousePos.y - 10);
            }

            if(MousePos.x >= 14 && MousePos.x <= 818 && MousePos.y >= 14 && MousePos.y <= 565 && debug){
                std::stringstream ss;
                local = Helper::getLocalCoords(MousePosView.x,MousePosView.y);
                ss << local.x <<", "<<local.y;
                tip.SetText(ss.str());
                tip.SetPosition(MousePos.x - 10, MousePos.y - 10);
            }

            ////////////////////////////////////////////////////
        }

        loops = 0;

        while(runner.GetElapsedTime() > next_game_tick && loops < MAX_FRAMESKIP){
            Ocean::update();

            next_game_tick += SKIP_TICKS;
            loops++;
        }

        curTime = runner.GetElapsedTime();
        ++frames;

        if (curTime - lastTime >= 1.0f) {
            fps = frames/(curTime - lastTime);
            frames = 0;
            lastTime = curTime;
        }

        float interpolation = float( runner.GetElapsedTime() + SKIP_TICKS - next_game_tick ) / float(SKIP_TICKS );

        App.SetView(App.GetDefaultView());
        App.Draw(backSprite);


        //////////////////////////  EXPERIMENTAL CAMERA LOCK   //////////////////////////////

        if(Ocean::choice && che && zoom >= 1){
            moX = Helper::worldToPixel[Ocean::fishMap[Ocean::choiceHash]->getX()][Ocean::fishMap[Ocean::choiceHash]->getY()].x - view.GetCenter().x;//caused by zooming
            moY = Helper::worldToPixel[Ocean::fishMap[Ocean::choiceHash]->getX()][Ocean::fishMap[Ocean::choiceHash]->getY()].y - view.GetCenter().y;

            view.Move(moX * interpolation, moY * interpolation);

        }
        if(zoom == 0){
            view.SetCenter(CenterX, CenterY);
        }

        //////////////////////////////////////////////////////////////////////////////////


        App.SetView(view);

        for(it = Ocean::fishMap.begin();it != Ocean::fishMap.end(); it++){

            float ElapsedTime = interpolation;
            float movX = (Helper::worldToPixel[it->second->getX()][it->second->getY()].x - it->second->sprite.GetPosition().x);
            float movY = (Helper::worldToPixel[it->second->getX()][it->second->getY()].y - it->second->sprite.GetPosition().y);

            it->second->sprite.Move(movX * ElapsedTime, movY * ElapsedTime);
        }

        for(it = Ocean::fishMap.begin();it != Ocean::fishMap.end(); it++){
            App.Draw(it->second->sprite);
        }


        Pollution::bind(&App);
        std::for_each(Ocean::pollution.begin(), Ocean::pollution.end(), std::mem_fun(&Pollution::draw));
        Pollution::bind(0);

        App.SetView(App.GetDefaultView());
        App.Draw(runSprite);

        if(che){
            lock.SetText("Press C to disable followcam");
            lock.SetPosition(610,550);
        }else{
            lock.SetText("Press C to enable followcam");
            lock.SetPosition(620,550);
        }

        App.Draw(lock);
        //Draw stats box
        Ocean::drawStats(&App, IScreen::logChoice, Ocean::choice);

        App.Draw(tooltip);
        App.Draw(tip);

        App.Display();
        App.Clear();


    }

    return(-1);
}

