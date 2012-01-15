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
    float SKIP_TICKS;
    const int MAX_FRAMESKIP;
};

RunScreen::RunScreen(void) : TICKS_PER_SECOND(1.0f), SKIP_TICKS(1.0f / TICKS_PER_SECOND), MAX_FRAMESKIP(10){
    playing = false;
}


int RunScreen::Run(sf::RenderWindow &App)
{
    App.SetFramerateLimit(0);
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
    float moX, CenterX;
    float moY, CenterY;
    bool che = false;
    bool debug = true;
    bool pollute = false;

    int zoom = 0;
    int loops = 0;
    float next_game_tick = runner.GetElapsedTime();
    float curTime, lastTime, fps = 0;
    int frames = 0;

    sf::Rect<float> catSelect[10];
    int top = 38;
    int bottom = top + 40;
    for(int i = 0; i < 10; i++){
        catSelect[i] = sf::Rect<float>(833,top,870,bottom);
        top = bottom + 13;
        bottom = top + 40;
    }
    sf::Sprite drag;
    int drop = 0;
    drag.SetScale(0.25,0.25);

    sf::String lock;
    lock.SetColor(sf::Color::White);
    lock.SetSize(15);
    lock.SetFont(Ocean::GlobalFont);

    sf::String action;
    action.SetColor(sf::Color::White);
    action.SetSize(15);
    action.SetFont(Ocean::GlobalFont);

    sf::Image he_lp;
    sf::Sprite help;
    if (!he_lp.LoadFromFile("artwork/help.png")){
        std::cerr<<"Error loading resources"<<std::endl;
        return(-1);
    }
    help.SetImage(he_lp);

    sf::Sprite tooltip;

    sf::Sprite d_vertical;
    sf::Sprite d_horizontal;
    sf::String d_tip;

    sf::Sprite selection;

    sf::Image vertical;
    sf::Image horizontal;
    sf::Image img_selection;

    if (!vertical.LoadFromFile("artwork/grid_vertical.png")){
        std::cerr<<"Error loading resources"<<std::endl;
        return(-1);
    }
    d_vertical.SetImage(vertical);

    if (!horizontal.LoadFromFile("artwork/grid_horizontal.png")){
        std::cerr<<"Error loading resources"<<std::endl;
        return(-1);
    }
    d_horizontal.SetImage(horizontal);

    if (!img_selection.LoadFromFile("artwork/selection.png")){
        std::cerr<<"Error loading resources"<<std::endl;
        return(-1);
    }
    selection.SetImage(img_selection);
    selection.SetScale(0.25f,0.25f);

    tooltip.SetColor(sf::Color::Black);

    sf::String tip;
    tip.SetColor(sf::Color::White);
    tip.SetSize(13);
    tip.SetFont(Ocean::GlobalFont);

    d_tip.SetColor(sf::Color::White);
    d_tip.SetSize(13);
    d_tip.SetFont(Ocean::GlobalFont);

    sf::Rect<float> addNew                  (831,39,875,560);
    sf::Rect<float> averageCategorySize     (871,39,900,560);
    sf::Rect<float> averageConsumptionWeek  (910,39,935,560);
    sf::Rect<float> averageDeathRate        (947,39,973,560);
    sf::Rect<float> averageAge              (986,39,1013,560);

    sf::Image st_b;
    sf::Image in_b;
    sf::Image  se_sl;
    sf::Sprite stop;
    sf::Sprite info;
    sf::Sprite slider;
    bool slide = false;

    if (!st_b.LoadFromFile("artwork/stop.png")){
        std::cerr<<"Error loading resources"<<std::endl;
        return(-1);
    }
    if (!in_b.LoadFromFile("artwork/info.png")){
        std::cerr<<"Error loading resources"<<std::endl;
        return(-1);
    }
    if (!se_sl.LoadFromFile("artwork/speed.png")){
        std::cerr<<"Error loading resources"<<std::endl;
        return(-1);
    }

    stop.SetImage(st_b);
    info.SetImage(in_b);
    slider.SetImage(se_sl);

    stop.SetPosition(756,573);
    info.SetPosition(787,573);
    slider.SetPosition(583,573);

    if(IScreen::speed == 15.0){
        slider.SetPosition(714,slider.GetPosition().y);
    }else if (IScreen::speed == 6.0){
        slider.SetPosition(668,slider.GetPosition().y);
    }else if(IScreen::speed == 3.0){
        slider.SetPosition(625,slider.GetPosition().y);
    }

    sf::Rect<float> stopRect, infoRect;
    stopRect = sf::FloatRect(764,581,786,597);
    infoRect = sf::FloatRect(795,581,817,597);

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
        std::cerr<<"Error loading resources"<<std::endl;
        return(-1);
    }
    backSprite.SetImage(back);

    if (!runImage.LoadFromFile("artwork/run.png")){
        std::cerr<<"Error loading resources"<<std::endl;
        return(-1);
    }
    runSprite.SetImage(runImage);

    lastTime = runner.GetElapsedTime();
    /************************************************************************* LOOP ***************************/
    while (Running)
    {
        SKIP_TICKS = 1.0f/IScreen::speed;
        /************************************************************************* EVENTS ***************************/
        MousePos = App.ConvertCoords(App.GetInput().GetMouseX(), App.GetInput().GetMouseY());
        MousePosView = App.ConvertCoords(App.GetInput().GetMouseX(),App.GetInput().GetMouseY(),&view);
        sf::Rect<float> mouseRect(MousePos.x, MousePos.y, MousePos.x + 0.001, MousePos.y + 0.001);
        sf::Rect<float> sliderRect(slider.GetPosition().x,slider.GetPosition().y + 5,slider.GetPosition().x + 37,slider.GetPosition().y + 20);

        while(App.GetEvent(Event))
        {
            if (Event.Type == sf::Event::Closed)
            {
                return (-1); // window close
            }
            if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::Return){
                playing = true;
                return(0); // state switch
            }
            if (Event.Type == sf::Event::MouseButtonPressed && Event.MouseButton.Button == sf::Mouse::Left){
                if(MousePos.x >= 14 && MousePos.x <= 818 && MousePos.y >= 14 && MousePos.y <= 565){
                    if(IScreen::actionChoice == 1){
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
                    }else if (IScreen::actionChoice == 2){
                        Ocean::pollute(rand()%5 + 1, Helper::getLocalCoords(MousePosView.x, MousePosView.y).x,Helper::getLocalCoords(MousePosView.x, MousePosView.y).y , rand()%8 + 3);
                    }else{
                        //throw nets
                    }

                }
            }

            if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::Space) {
                IScreen::actionChoice++;
                if (IScreen::actionChoice > 3)IScreen::actionChoice = 1;
            }
            if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::F1) {
                IScreen::helpChoice = !IScreen::helpChoice;
            }
            if (Event.Type == sf::Event::MouseButtonPressed && Event.MouseButton.Button == sf::Mouse::Left){
                if(MousePos.x > 826 && MousePos.x < 1015 && MousePos.y > 576 && MousePos.y < 589){
                    IScreen::logChoice = !IScreen::logChoice;
                }
            }
            /////////////////////////////////////////////////////TO ADD IN PAUSE
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
            if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::D){
                debug = !debug;
            }
            if (Event.Type == sf::Event::MouseButtonPressed && Event.MouseButton.Button == sf::Mouse::Left){
                //std::cout<<MousePos.x<<" "<<MousePos.y<<std::endl;
                if(stopRect.Intersects(mouseRect)){        //restart function
                    std::cout<<"in stop"<<std::endl;
                    return 2;
                }
                if(infoRect.Intersects(mouseRect)){        //info sprite display in pause screen
                    IScreen::info = true;
                    playing = true;
                    return(0);
                }

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
                d_tip.SetText(ss.str());
                d_tip.SetPosition(MousePos.x + 5, MousePos.y - 15);

                d_vertical.SetPosition(MousePos.x, 15);
                d_horizontal.SetPosition(13, MousePos.y);
            }else{
                d_vertical.SetPosition(0,0);
                d_horizontal.SetPosition(0,0);
                d_tip.SetPosition(1024, 600); //sta diala
            }

            ////////////////////////////////////////////////////TO ADD IN PAUSE
        }

        if(App.GetInput().IsMouseButtonDown(sf::Mouse::Left)){
            for(int i = 0; i < 10;i++){
                if(catSelect[i].Intersects(mouseRect)){
                    drop = i+1;
                    drag.SetImage(ClassRegistry::assocMapImages[i]);
                }
            }

            if(drop){
                drag.SetPosition(MousePos.x,MousePos.y);
            }else{
                drag.SetPosition(1024,600);
            }

            if(sliderRect.Intersects(mouseRect)){
                slide = true;
            }

            if(slide){
                if(MousePos.x > 600 && MousePos.x < 754 - 15){
                    slider.SetPosition(MousePos.x - 20, slider.GetPosition().y);
                }
            }

            //            if(IScreen::actionChoice == 2){
            //                pollute = true;
            //            }
            //            if(pollute){
            //                Ocean::pollute(rand()%5 + 1, Helper::getLocalCoords(MousePosView.x, MousePosView.y).x,Helper::getLocalCoords(MousePosView.x, MousePosView.y).y , rand()%8 + 3);
            //                App.
            //                pollute = false;
            //            }
        }else{
            if(drop){
                sf::Vector2i local = Helper::getLocalCoords(MousePosView.x ,MousePosView.y);
                int hash = local.x + local.y * Ocean::MAX_X;

                if(MousePos.x >= 14 && MousePos.x <= 818 && MousePos.y >= 14 && MousePos.y <= 565){
                    if(Ocean::fishMap.find(hash) == Ocean::fishMap.end()){
                        Ocean::createAndAddFish(drop-1, local.x, local.y) ;
                        std::stringstream ss;
                        ss << "A "<<ClassRegistry::assocMapNames[drop - 1]<<" was added by user";
                        Ocean::regLog(ss.str());
                    }
                }
            }
            drop = 0;
            slide = false;
            //            if(pollute){
            //                Ocean::pollute(rand()%5 + 1, Helper::getLocalCoords(MousePosView.x, MousePosView.y).x,Helper::getLocalCoords(MousePosView.x, MousePosView.y).y , rand()%8 + 3);
            //            }
            //            pollute = false;

            if(slider.GetPosition().x > 700){
                slider.SetPosition(714,slider.GetPosition().y);
                IScreen::speed = 15.0f;
            }else if(slider.GetPosition().x > 653){
                slider.SetPosition(668,slider.GetPosition().y);
                IScreen::speed = 6.0f;
            }else if(slider.GetPosition().x > 610){
                slider.SetPosition(625,slider.GetPosition().y);
                IScreen::speed = 3.0f;
            }else{
                slider.SetPosition(583,slider.GetPosition().y);
                IScreen::speed = 1.0f;
            }
            SKIP_TICKS = 1.0f/IScreen::speed;
        }

        /************************************************************************* UPDATE ***************************/
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

        /************************************************************************* RENDER ***************************/
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

            float movX = (Helper::worldToPixel[it->second->getX()][it->second->getY()].x - it->second->sprite.GetPosition().x);
            float movY = (Helper::worldToPixel[it->second->getX()][it->second->getY()].y - it->second->sprite.GetPosition().y);

            //if(Ocean::choice && Ocean::choiceHash == it->first){
            //std::cout<<loops<<"           "<<movX<<"         "<<movY<<std::endl;
            //}

            it->second->sprite.Move(movX * interpolation, movY * interpolation);
        }

        for(it = Ocean::fishMap.begin();it != Ocean::fishMap.end(); it++){
            if(Ocean::choice && Ocean::choiceHash == it->first){
                selection.SetPosition(Ocean::fishMap[Ocean::choiceHash]->sprite.GetPosition().x-5, Ocean::fishMap[Ocean::choiceHash]->sprite.GetPosition().y-5);
                App.Draw(selection);
            }
            App.Draw(it->second->sprite);
        }


        Pollution::bind(&App);
        std::for_each(Ocean::pollution.begin(), Ocean::pollution.end(), std::mem_fun(&Pollution::draw));
        Pollution::bind(0);

        App.SetView(App.GetDefaultView());

        if(debug){
            App.Draw(d_horizontal);
            App.Draw(d_vertical);
            App.Draw(d_tip);
        }

        App.Draw(runSprite);

        if(che){
            lock.SetText("Press C to disable followcam");
            lock.SetPosition(618,545);
        }else{
            lock.SetText("Press C to enable followcam");
            lock.SetPosition(625,545);
        }

        if(IScreen::actionChoice == 1){
            action.SetText("Click to choose organism");
            action.SetPosition(645,530);
        }else if(IScreen::actionChoice == 2){
            action.SetText("Click to add pollution");
            action.SetPosition(660,530);
        }else{
            action.SetText("Click to add nets");
            action.SetPosition(695,530);
        }
        if(debug){
            App.Draw(action);
            App.Draw(lock);
        }
        //Draw stats box
        Ocean::drawStats(&App, IScreen::logChoice, Ocean::choice);

        if(!Ocean::choice){
            App.Draw(tooltip);
            App.Draw(tip);
        }
        if(drop)App.Draw(drag);

        App.Draw(stop);
        App.Draw(info);
        App.Draw(slider);

        if(IScreen::helpChoice)
            App.Draw(help);

        App.Display();
        App.Clear();


    }

    return(-1);
}

