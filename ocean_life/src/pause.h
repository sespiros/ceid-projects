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
    App.SetFramerateLimit(23);
    IScreen::introMusic.SetVolume(5);
    if (mute) introMusic.SetVolume(0);

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
    float CenterX;
    float CenterY;
    bool debug = true;

    sf::Sprite about;
    sf::Image ab_im;
    if (!ab_im.LoadFromFile("artwork/about.png")){
        std::cerr<<"Error loading resources"<<std::endl;
        return(-1);
    }
    about.SetImage(ab_im);

    int zoom = 0;

    sf::Image he_lp;
    sf::Sprite help;
    if (!he_lp.LoadFromFile("artwork/help.png")){
        std::cerr<<"Error loading resources"<<std::endl;
        return(-1);
    }
    help.SetImage(he_lp);

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

    sf::Sprite tooltip;

    sf::Sprite d_vertical;
    sf::Sprite d_horizontal;
    sf::String d_tip;

    sf::Sprite selection;

    sf::Image vertical;
    sf::Image horizontal;
    sf::Image img_selection;

    if (!vertical.LoadFromFile("artwork/grid_vertical.png")){
        std::cerr<<"Error loading background image"<<std::endl;
        return(-1);
    }
    d_vertical.SetImage(vertical);

    if (!horizontal.LoadFromFile("artwork/grid_horizontal.png")){
        std::cerr<<"Error loading background image"<<std::endl;
        return(-1);
    }
    d_horizontal.SetImage(horizontal);

    if (!img_selection.LoadFromFile("artwork/selection.png")){
        std::cerr<<"Error loading background image"<<std::endl;
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
        std::cerr<<"Error loading background image"<<std::endl;
        return(-1);
    }
    if (!in_b.LoadFromFile("artwork/info.png")){
        std::cerr<<"Error loading background image"<<std::endl;
        return(-1);
    }
    if (!se_sl.LoadFromFile("artwork/speed.png")){
        std::cerr<<"Error loading background image"<<std::endl;
        return(-1);
    }

    stop.SetImage(st_b);
    info.SetImage(in_b);
    slider.SetImage(se_sl);

    stop.SetPosition(756,573);
    info.SetPosition(787,573);
    slider.SetPosition(583,573);

    if(IScreen::speed == 7.0){
        slider.SetPosition(714,slider.GetPosition().y);
    }else if (IScreen::speed == 5.0){
        slider.SetPosition(668,slider.GetPosition().y);
    }else if(IScreen::speed == 3.0){
        slider.SetPosition(583,slider.GetPosition().y);
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
        MousePos = App.ConvertCoords(App.GetInput().GetMouseX(), App.GetInput().GetMouseY());
        MousePosView = App.ConvertCoords(App.GetInput().GetMouseX(),App.GetInput().GetMouseY(),&view);
        sf::Rect<float> mouseRect(MousePos.x, MousePos.y, MousePos.x + 0.001, MousePos.y + 0.001);
        sf::Rect<float> sliderRect(slider.GetPosition().x,slider.GetPosition().y + 5,slider.GetPosition().x + 37,slider.GetPosition().y + 20);

        while(App.GetEvent(Event))
        {
            switch (Event.Type) {
            case sf::Event::Closed:
                return -1;
                break;
            case sf::Event::KeyPressed:
                if (Event.Key.Code == sf::Key::M){
                    mute = !mute;
                    if (mute)
                        introMusic.SetVolume(0);
                    else
                        introMusic.SetVolume(5);
                }
                break;
            default:
                break;
            }
            if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::Escape){
                IScreen::info = false;
            }
            if (Event.Type == sf::Event::Closed)
            {
                return (-1);
            }
            if(!IScreen::info){

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
                if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::F1) {
                    IScreen::helpChoice = !IScreen::helpChoice;
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
                        if(zoom == 0){
                            view.SetCenter(CenterX, CenterY);
                        }
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
                if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::D){
                    debug = !debug;
                }
                if (Event.Type == sf::Event::MouseButtonPressed && Event.MouseButton.Button == sf::Mouse::Left){
                    if(stopRect.Intersects(mouseRect)){        //restart function
                        return 2;
                        std::cout<<"in stop"<<std::endl;
                    }
                    if(infoRect.Intersects(mouseRect)){        //info sprite display in pause screen
                        IScreen::info = true;
                        std::cout<<"in info"<<std::endl;
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
                    d_tip.SetPosition(1024,600); //STA DIALA
                }
            }

        }
        if(!IScreen::info){
            if(App.GetInput().IsMouseButtonDown(sf::Mouse::Left)){
                for(int i = 0; i < 10;i++){
                    if(catSelect[i].Intersects(mouseRect) && !drop){
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

                if(slider.GetPosition().x > 700){
                    slider.SetPosition(714,slider.GetPosition().y);
                    IScreen::speed = 7.0f;
                }else if(slider.GetPosition().x > 653){
                    slider.SetPosition(668,slider.GetPosition().y);
                    IScreen::speed = 5.0f;
                }else if(slider.GetPosition().x > 610){
                    slider.SetPosition(625,slider.GetPosition().y);
                    IScreen::speed = 3.0f;
                }else{
                    slider.SetPosition(583,slider.GetPosition().y);
                    IScreen::speed = 1.0f;
                }
            }
        }

        App.SetView(App.GetDefaultView());
        App.Draw(backSprite);

        App.SetView(view);

        std::map<int, Organism*>::iterator it;
        for(it = Ocean::fishMap.begin();it != Ocean::fishMap.end(); it++){
            if(Ocean::choice && Ocean::choiceHash == it->first){
                selection.SetPosition(Ocean::fishMap[Ocean::choiceHash]->sprite.GetPosition().x-5, Ocean::fishMap[Ocean::choiceHash]->sprite.GetPosition().y-5);
                App.Draw(selection);
            }
            App.Draw(it->second->sprite);
        }

        Pollution::bind(&App, true);
        std::for_each(Ocean::pollution.begin(), Ocean::pollution.end(), std::mem_fun(&Pollution::draw));
        Pollution::bind(0);

        Net::bind(&App);
        std::for_each(Ocean::nets.begin(), Ocean::nets.end(), std::mem_fun(&Net::draw));
        Net::bind(0);

        App.SetView(App.GetDefaultView());

        if(debug){
            App.Draw(d_horizontal);
            App.Draw(d_vertical);
            App.Draw(d_tip);
        }

        App.Draw(runSprite);
        App.Draw(pauseSprite);

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

        if(IScreen::info){
            App.Draw(about);
        }

        App.Display();
        App.Clear();
    }
    return(-1);
}
