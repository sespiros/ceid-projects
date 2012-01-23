#ifndef MENU_H
#define MENU_H

#include <iostream>

#include "iscreen.h"
#include "Ocean.h"

/**
 * Menu state class
 */
class MenuScreen : public IScreen {
public:
    MenuScreen();
    virtual int Run(sf::RenderWindow &App);
private:
    sf::Image bgImg;
    sf::Sprite bgSprite;
    bool mute;
};

MenuScreen::MenuScreen()
{
    if(!bgImg.LoadFromFile("artwork/about.jpg")) {
        std::cerr << "Error loading background image (about)!" << std::endl;
    }
    bgSprite.SetImage(bgImg);
    if (!IScreen::introMusic.OpenFromFile("artwork/intro.ogg")) {
        std::cerr << "Error loading intro music!" << std::endl;
    }
    IScreen::introMusic.SetLoop(true);
    mute = true;
}

int MenuScreen::Run(sf::RenderWindow &App)
{
    App.SetFramerateLimit(60);
    IScreen::introMusic.SetVolume(30);

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

    if (!playBigImg.LoadFromFile("artwork/playb.png") || !playSmallImg.LoadFromFile("artwork/plays.png"))
        std::cerr << "Error loading text images!" << std::endl;
    playBigSprite.SetImage(playBigImg);
    playSmallSprite.SetImage(playSmallImg);

    float spacing = 20.0f, offsetY = 0, offsetX = 0;
    float totalSize = playBigSprite.GetSize().y + playSmallSprite.GetSize().y + spacing;

    playBigSprite.SetCenter(playBigSprite.GetSize().x/2.0f, playBigSprite.GetSize().y/2.0f);
    playBigSprite.SetPosition((App.ConvertCoords(App.GetWidth(), 0)).x/2.0f, (App.ConvertCoords(0, App.GetHeight()).y - totalSize + playBigSprite.GetSize().y)/2.0f);
    playSmallSprite.SetCenter(playSmallSprite.GetSize().x/2.0f, playSmallSprite.GetSize().y/2.0f);
    playSmallSprite.SetPosition((App.ConvertCoords(App.GetWidth(), 0)).x/2.0f, (App.ConvertCoords(0, App.GetHeight()).y - totalSize + playSmallSprite.GetSize().y)/2.0f + playBigSprite.GetSize().y + spacing);

    sf::FloatRect mouseRect, playRect;
    sf::Vector2f orig(0, 0), end;

    sf::Event e;

    while (Running) {
        while (App.GetEvent(e)) {
            switch (e.Type) {
            case sf::Event::Closed:
                return -1;
                break;
            case sf::Event::KeyPressed:
                if (e.Key.Code == sf::Key::M){
                    if (mute)
                        introMusic.SetVolume(0);
                    else
                        introMusic.SetVolume(80);
                    mute = !mute;
                }
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
        // normal - reduced size
        offsetY = playSmallImg.GetHeight() * (1 - playSmallSprite.GetScale().x);
        offsetX = playSmallImg.GetWidth() * (1 - playSmallSprite.GetScale().y);
        playRect.Left = playSmallSprite.TransformToGlobal(orig).x - offsetX/2;
        playRect.Top = playSmallSprite.TransformToGlobal(orig).y - offsetY/2;
        playRect.Right = playSmallSprite.TransformToGlobal(end).x + offsetX/2;
        playRect.Bottom = playSmallSprite.TransformToGlobal(end).y + offsetY/2;

        if (mouseRect.Intersects(playRect)) {
            playSmallSprite.SetScale(0.95f, 0.95f);
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
        offsetY = playSmallImg.GetHeight() * (1 - playBigSprite.GetScale().x);
        offsetX = playSmallImg.GetWidth() * (1 - playBigSprite.GetScale().y);
        playRect.Left = playBigSprite.TransformToGlobal(orig).x - offsetX/2;
        playRect.Top = playBigSprite.TransformToGlobal(orig).y - offsetY/2;
        playRect.Right = playBigSprite.TransformToGlobal(end).x + offsetX/2;
        playRect.Bottom = playBigSprite.TransformToGlobal(end).y + offsetY/2;

        if (mouseRect.Intersects(playRect)) {
            playBigSprite.SetScale(0.95f, 0.95f);
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
