#include <iostream>
#include <functional>
#include "iscreen.h"
#include "organism.h"
#include "Ocean.h"

class ExpScreen : public IScreen
{
private:
	bool playing;
public:
	ExpScreen (void);
	virtual int Run (sf::RenderWindow &App);
	const float TICKS_PER_SECOND;
	const float SKIP_TICKS;
	const int MAX_FRAMESKIP;
	//static int a;
	//static int b;
};

ExpScreen::ExpScreen(void) : TICKS_PER_SECOND(1.2f), SKIP_TICKS(1.0f / TICKS_PER_SECOND), MAX_FRAMESKIP(10){
	playing = false;
}

//int RunScreen::a = 0;
//int RunScreen::b = 0;

int ExpScreen::Run(sf::RenderWindow &App)
{
	sf::Event Event;
	bool Running = true;
	sf::Image runImage;
	sf::Image back;
	sf::Sprite runSprite;
	sf::Sprite backSprite;
	sf::Clock runner;
	sf::Vector2f MousePos;
	int zoom = 0;
	int loops = 0;
	float next_game_tick = runner.GetElapsedTime();
	float curTime, lastTime, fps = 0;
	int frames = 0;

	//adjusted view
	//sf::View view(sf::FloatRect(-11.7823,-5.66833,1190.67,690.782));
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
		view.Zoom(0.97f);
		if (!back.LoadFromFile("artwork/small.jpg")){
			std::cerr<<"Error loading background image"<<std::endl;
			return(-1);
		}

		backSprite.SetImage(back);

	}


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

	lastTime = runner.GetElapsedTime();

	while (Running)
	{

		while(App.GetEvent(Event))
		{
			MousePos = App.ConvertCoords(App.GetInput().GetMouseX(), App.GetInput().GetMouseY());
			if (Event.Type == sf::Event::Closed)
			{
				return (-1); // window close
			}
			if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::Escape){
				playing = true;
				return(0); // state switch
			}
			if (Event.Type == sf::Event::MouseButtonPressed && Event.MouseButton.Button == sf::Mouse::Left){
				if(MousePos.x > 15 && MousePos.x < 818 && MousePos.y > 15 && MousePos.y < 512){
					playing = true;
					return(0); // state switch
				}
			}
			if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::Space) {
				Ocean::pollute(rand()%5 + 1, rand()%Ocean::MAX_X, rand()%Ocean::MAX_Y, rand()%8 + 3);
			}
			if (Event.Type == sf::Event::MouseButtonPressed && Event.MouseButton.Button == sf::Mouse::Left){
				//std::cout << App.GetInput().GetMouseX()<<" "<<App.GetInput().GetMouseY()<<std::endl;
				if(MousePos.x > 826 && MousePos.x < 1015 && MousePos.y > 558 && MousePos.y < 594){
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
				view.Move(10.0f, 0);
			}
			if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::Right){
				view.Move(-10.0f, 0);
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


	}

	return(-1);
}

