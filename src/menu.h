#ifndef MENU_H
#define MENU_H

#include "iscreen.h"

class MenuScreen : public IScreen {
public:
    MenuScreen();
    virtual int Run(sf::RenderWindow &App);
private:
    sf::Image bgImg;
    sf::Sprite bgSprite;
};

MenuScreen::MenuScreen()
{
    if(!bgImg.LoadFromFile("artwork/about.jpg")) {
        std::cerr << "Error loading background image (about)!" << std::endl;
    }
    bgSprite.SetImage(bgImg);
}

int MenuScreen::Run(sf::RenderWindow &App)
{
    App.SetFramerateLimit(60);

    bool Running = true;

    float mouseX, mouseY;

    sf::Image playBigImg, playSmallImg;
    sf::Sprite playBigSprite, playSmallSprite;

    sf::Clock timer;

    if (!playBigImg.LoadFromFile("artwork/playb.png") || !playSmallImg.LoadFromFile("artwork/plays.png"))
        std::cerr << "Error loading text images!" << std::endl;
    playBigSprite.SetImage(playBigImg);
    playSmallSprite.SetImage(playSmallImg);

    float spacing = 50.0f;
    float totalSize = playBigSprite.GetSize().x + playSmallSprite.GetSize().x + spacing;

    playBigSprite.SetCenter(playBigSprite.GetSize().x/2.0f, playBigSprite.GetSize().y/2.0f);
    playBigSprite.SetPosition((App.GetWidth() - totalSize + playBigSprite.GetSize().x)/2.0f, 250);
    playSmallSprite.SetCenter(playSmallSprite.GetSize().x/2.0f, playSmallSprite.GetSize().y/2.0f);
    playSmallSprite.SetPosition((App.GetWidth() - totalSize + playSmallSprite.GetSize().x)/2.0f + playBigSprite.GetSize().x + spacing, 250);

    sf::FloatRect mouseRect;

    sf::Event e;

    while (Running) {
        while (App.GetEvent(e)) {
            switch (e.Type) {
            case sf::Event::Closed:
                return -1;
                break;
            default:
                break;
            }
        }

        mouseX = App.GetInput().GetMouseX();
        mouseY = App.GetInput().GetMouseY();
        mouseRect.Left = mouseX;
        mouseRect.Top = mouseY;
        mouseRect.Right = mouseX + 0.1f;
        mouseRect.Bottom = mouseY + 0.1f;

        if (mouseX >= playSmallSprite.GetPosition().x - playSmallSprite.GetCenter().x && mouseX <= playSmallSprite.GetPosition().x - playSmallSprite.GetCenter().x + playSmallSprite.GetSize().x) {
            playSmallSprite.SetScale(1.0f, 0.6f);
        }
        else
            playSmallSprite.SetScale(1, 1);

        playBigSprite.SetScale(0.02f * cos(1.5f * timer.GetElapsedTime()) + 1, 0.02f * cos(1.5f * timer.GetElapsedTime()) + 1);

        App.Draw(bgSprite);
        App.Draw(playBigSprite);
        App.Draw(playSmallSprite);
        App.Display();
        App.Clear();
    }
}

#endif // MENU_H
