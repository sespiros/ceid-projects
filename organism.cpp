#include "organism.h"
#include <SFML/Graphics.hpp>

int ZPlankton::count=0;
int PPlankton::count=0;
int Shrimp::count=0;
int Jelly::count=0;
int Eel::count=0;
int Balloon::count=0;
int Clown::count=0;
int Gtp::count=0;
int Magikarp::count=0;
int Narwhal::count=0;



Organism::Organism(int x, int y, int a, int s, float gr, int fr, int v, Organism::fishtype t):
    size(s),age(a),growthRate(gr),foodRequired(fr),velocity(v),type(t){
    (*this).x=x;
    (*this).y=y;

}

Organism::fishtype Organism::getType() {
    return type;
}

ZPlankton::ZPlankton(int x, int y):Organism(x, y, 1, 0, 1.0f, 5, 1, ZPL){
    count++;

    image.LoadFromFile("C:/Users/Spiros/Qt Projects/ocean_life/artwork/ZPlankton.png");
    sprite.SetImage(image);
    sprite.SetScaleX(0.25f);
    sprite.SetScaleY(0.25f);

    sprite.SetX((float)x);
    sprite.SetY((float)y);
}

PPlankton::PPlankton(int x, int y):Organism(x, y, 1, 0, 0, 0, 1, PPL){
    count++;
    image.LoadFromFile("C:/Users/Spiros/Qt Projects/ocean_life/artwork/PPlankton.png");
    sprite.SetImage(image);
    sprite.SetScaleX(0.25f);
    sprite.SetScaleY(0.25f);

    sprite.SetX((float)x);
    sprite.SetY((float)y);
}

Shrimp::Shrimp(int x, int y):Organism(x, y, 1, 0, 1.0f, 4, 2, SHRIMP){
    count++;
    image.LoadFromFile("C:/Users/Spiros/Qt Projects/ocean_life/artwork/Shrimp.png");
    sprite.SetImage(image);
    sprite.SetScaleX(0.25f);
    sprite.SetScaleY(0.25f);

    sprite.SetX((float)x);
    sprite.SetY((float)y);
}

Jelly::Jelly(int x, int y):Organism(x, y, 1, 0, 1.0f, 4, 1, JELLY){
    count++;
    image.LoadFromFile("C:/Users/Spiros/Qt Projects/ocean_life/artwork/Jelly.png");
    sprite.SetImage(image);
    sprite.SetScaleX(0.25f);
    sprite.SetScaleY(0.25f);

    sprite.SetX((float)x);
    sprite.SetY((float)y);
}

Eel::Eel(int x, int y):Organism(x, y, 1, 0, 2.0f, 4, 5, EEL){
    count++;
    image.LoadFromFile("C:/Users/Spiros/Qt Projects/ocean_life/artwork/Eel.png");
    sprite.SetImage(image);
    sprite.SetScaleX(0.25f);
    sprite.SetScaleY(0.25f);

    sprite.SetX((float)x);
    sprite.SetY((float)y);
}

Balloon::Balloon(int x, int y):Organism(x, y, 1, 0, 3.0f, 3, 3, BALLOON){
    count++;
    image.LoadFromFile("C:/Users/Spiros/Qt Projects/ocean_life/artwork/Baloon.png");
    sprite.SetImage(image);
    sprite.SetScaleX(0.25f);
    sprite.SetScaleY(0.25f);

    sprite.SetX((float)x);
    sprite.SetY((float)y);
}

Clown::Clown(int x, int y):Organism(x, y, 1, 0, 2.0f, 3, 4, CLOWN){
    count++;
    image.LoadFromFile("C:/Users/Spiros/Qt Projects/ocean_life/artwork/Clown.png");
    sprite.SetImage(image);
    sprite.SetScaleX(0.25f);
    sprite.SetScaleY(0.25f);

    sprite.SetX((float)x);
    sprite.SetY((float)y);
}

Gtp::Gtp(int x, int y):Organism(x, y, 1, 0, 2.0f, 3, 3, GTP){
    count++;
    image.LoadFromFile("C:/Users/Spiros/Qt Projects/ocean_life/artwork/Gtp.png");
    sprite.SetImage(image);
    sprite.SetScaleX(0.25f);
    sprite.SetScaleY(0.25f);

    sprite.SetX((float)x);
    sprite.SetY((float)y);
}

Magikarp::Magikarp(int x, int y):Organism(x, y, 1, 0, 2.0f, 3, 4, MAGIKARP){
    count++;
    image.LoadFromFile("C:/Users/Spiros/Qt Projects/ocean_life/artwork/LEGENDARY.png");
    sprite.SetImage(image);
    sprite.SetScaleX(0.25f);
    sprite.SetScaleY(0.25f);

    sprite.SetX((float)x);
    sprite.SetY((float)y);
}

Narwhal::Narwhal(int x, int y):Organism(x, y, 1, 0, 4.0f, 8, 6, NARWHAL){
    count++;
    image.LoadFromFile("C:/Users/Spiros/Qt Projects/ocean_life/artwork/Narwhal.png");
    sprite.SetImage(image);
    sprite.SetScaleX(0.25f);
    sprite.SetScaleY(0.25f);

    sprite.SetX((float)x);
    sprite.SetY((float)y);
}

int Organism::getX(){
    return x;
}
int Organism::getY(){
    return y;
}
void Organism::setX(int x){
    (*this).x=x;
    sprite.SetX(x*1.0f);
}
void Organism::setY(int y){
    (*this).y=y;
    sprite.SetY(y*1.0f);
}

