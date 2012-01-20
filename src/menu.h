#ifndef MENU_H
#define MENU_H

#include <iostream>

#include "iscreen.h"
#include "Ocean.h"

/** Menu state class
  *
  */
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
    sf::String help;
    help.SetColor(sf::Color::White);
    help.SetFont(Ocean::GlobalFont);
    help.SetSize(15);
    help.SetPosition(5,5);
    help.SetText("In Game press F1 for help");

    sf::Clock timer;

    if (!playBigImg.LoadFromFile("artwork/playb.png") || !playSmallImg.LoadFromFile("artwork/plays.png"))
        std::cerr << "Error loading text images!" << std::endl;
    playBigSprite.SetImage(playBigImg);
    playSmallSprite.SetImage(playSmallImg);

    float spacing = 50.0f, offsetY = 0, offsetX = 0;
    float totalSize = playBigSprite.GetSize().x + playSmallSprite.GetSize().x + spacing;

    playBigSprite.SetCenter(playBigSprite.GetSize().x/2.0f, playBigSprite.GetSize().y/2.0f);
    playBigSprite.SetPosition((App.ConvertCoords(App.GetWidth(), 0).x - totalSize + playBigSprite.GetSize().x)/2.0f, 250);
    playSmallSprite.SetCenter(playSmallSprite.GetSize().x/2.0f, playSmallSprite.GetSize().y/2.0f);
    playSmallSprite.SetPosition((App.ConvertCoords(App.GetWidth(), 0).x - totalSize + playSmallSprite.GetSize().x)/2.0f + playBigSprite.GetSize().x + spacing, 250);

    sf::FloatRect mouseRect, playRect;
    sf::Vector2f orig(0, 0), end;

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

        mouseX = App.ConvertCoords(App.GetInput().GetMouseX(), App.GetInput().GetMouseY()).x;
        mouseY = App.ConvertCoords(App.GetInput().GetMouseX(), App.GetInput().GetMouseY()).y;
        mouseRect.Left = mouseX;
        mouseRect.Top = mouseY;
        mouseRect.Right = mouseX + 0.1f;
        mouseRect.Bottom = mouseY + 0.1f;

        end.x = playSmallImg.GetWidth();
        end.y = playSmallImg.GetHeight();
        playRect.Left = playSmallSprite.TransformToGlobal(orig).x - offsetX/2;
        playRect.Top = playSmallSprite.TransformToGlobal(orig).y - offsetY/2;
        playRect.Right = playSmallSprite.TransformToGlobal(end).x + offsetX/2;
        playRect.Bottom = playSmallSprite.TransformToGlobal(end).y + offsetY/2;

        if (mouseRect.Intersects(playRect)) {
            playSmallSprite.SetScale(0.95f, 0.95f);
            // normal - reduced size
            offsetY = playSmallImg.GetHeight() * (1 - playSmallSprite.GetScale().x);
            offsetX = playSmallImg.GetWidth() * (1 - playSmallSprite.GetScale().y);
            if (App.GetInput().IsMouseButtonDown(sf::Mouse::Left)) {
                Ocean::init(false);
                return 0;
            }
        }
        else{
            playSmallSprite.SetScale(1, 1);
            offsetY = 0;
            offsetX = 0;
        }

        end.x = playBigImg.GetWidth();
        end.y = playBigImg.GetHeight();
        playRect.Left = playBigSprite.TransformToGlobal(orig).x;
        playRect.Top = playBigSprite.TransformToGlobal(orig).y;
        playRect.Right = playBigSprite.TransformToGlobal(end).x;
        playRect.Bottom = playBigSprite.TransformToGlobal(end).y;

        if (mouseRect.Intersects(playRect)) {
            playBigSprite.SetScale(1.2f, 1.2f);
            if (App.GetInput().IsMouseButtonDown(sf::Mouse::Left)) {
                Ocean::init(true);
                return 0;
            }
        }
        else
            playBigSprite.SetScale(1, 1);

        App.Draw(bgSprite);
        App.Draw(playBigSprite);
        App.Draw(playSmallSprite);
        App.Draw(help);
        App.Display();
        App.Clear();
    }

    return 0;
}

#endif // MENU_H
