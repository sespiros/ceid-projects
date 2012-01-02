#include "runscreen.h"

int RunScreen::Run(sf::RenderWindow &App)
{
    sf::Event Event;
    bool Running = true;
    sf::Image runImage;
    sf::Image back;
    sf::Sprite runSprite;
    sf::Sprite backSprite;
    sf::Clock runner;

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

    if (!runImage.LoadFromFile("artwork/run2.png")){
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
                if (Event.Type == sf::Event::Closed)
                {
                    return (-1);
                }
                if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::Escape){
                    playing = true;
                    return(0);
                }
                //FIX THE ZOOM AND MOVE CONTROLS-------------------------------------------------------
                if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::Q){   //view.Zoom(1.050f);

                }
                if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::E){   //view.Zoom(0.950f);

                }

                if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::A){   //view.Move(5.0f,0);

                }
                if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::D){   //view.Move(-5.0f,0);

                }
                if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::W){   //view.Move(0,5.0f);

                }
                if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::S){   //view.Move(0,-5.0f);

                }
                //--------------------------------------------------------------------------------------
				if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::Space) {
					Ocean::pollute(rand()%4 + 1, rand()%Ocean::MAX_X, rand()%Ocean::MAX_Y);
				}
			}

            Ocean::update();

            accumulator-=dt;
            drawn=false;

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

            App.SetView(App.GetDefaultView());
            App.Draw(runSprite);

            App.Display();
            App.Clear();

            drawn=true;
            //debug for placing view
            //std::cout<<view.GetRect().Right<<","<<view.GetRect().Top<<" "<<view.GetRect().Left<<","<<view.GetRect().Bottom<<std::endl;
        }


    }

    return(-1);
}

