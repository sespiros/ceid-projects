#include <iostream>
#include "iscreen.h"
#include "organism.h"
#include "Ocean.h"

class AboutScreen : public IScreen
{
private:
    bool playing;
public:
    AboutScreen (void);
    virtual int Run (sf::RenderWindow &App);
};


AboutScreen::AboutScreen(void){
    playing=false;
}

int AboutScreen::Run(sf::RenderWindow &App)
{
    sf::Event Event;
    bool Running = true;
    sf::Image aboutImage;
    sf::Sprite aboutSprite;
    sf::Image runImage;
    sf::Sprite runSprite;

    sf::String Names;
    sf::View view(sf::FloatRect(10, 5, 817, 517));

    if (!aboutImage.LoadFromFile("artwork/about.png")){
        std::cerr<<"Error loading background image"<<std::endl;
        return(-1);
    }
    aboutSprite.SetImage(aboutImage);

    if (!runImage.LoadFromFile("artwork/run2.png")){
        std::cerr<<"Error loading background image"<<std::endl;
        return(-1);
    }
    runSprite.SetImage(runImage);

    Names.SetSize(40);
    Names.SetText("Spyros Seimenis");
    Names.SetX(100);
    Names.SetY(500);

    while (Running)
    {
        while(App.GetEvent(Event))
        {
            if (Event.Type == sf::Event::Closed)
            {
                return (-1);
            }
            if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::Return){
                playing = true;
                return(1);
            }
        }

        //App.SetView(view);
        std::map<int, Organism*>::iterator it;
        for(it = Ocean::fishMap.begin();it != Ocean::fishMap.end(); it++){
            App.Draw(it->second->sprite);
        }

		Pollution::bind(&App);
		std::for_each(Ocean::pollution.begin(), Ocean::pollution.end(), std::mem_fun(&Pollution::draw));
		Pollution::bind(0);

        App.SetView(App.GetDefaultView());
        App.Draw(runSprite);
        App.Draw(aboutSprite);

        App.Display();
        App.Clear();
    }
    return(-1);
}
