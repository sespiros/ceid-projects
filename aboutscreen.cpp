#include "aboutscreen.h"

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
		for(int i=0;i<Ocean::count;i++){
			App.Draw((Ocean::fish.at(i))->sprite);
		}
		App.SetView(App.GetDefaultView());
		App.Draw(runSprite);
		App.Draw(aboutSprite);

		App.Display();
		App.Clear();
	}
	return(-1);
}
