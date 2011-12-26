#include "organism.h"
#include "helper.h"


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

const std::map<Organism::fishtype, int> Organism::weightMap = Organism::createWeightMap();

Organism::Organism(int x, int y, int a, int s, float gr, int fr, int v, Organism::fishtype t):
    size(s),age(a),growthRate(gr),foodRequired(fr),velocity(v),type(t)
{
    (*this).x=x;
    (*this).y=y;

}

std::map<Organism::fishtype, int> Organism::createWeightMap() {
	std::map<Organism::fishtype, int> tmp;

	tmp.insert(std::pair<Organism::fishtype, int>(PPL, 25));
	tmp.insert(std::pair<Organism::fishtype, int>(ZPL, 20));
	tmp.insert(std::pair<Organism::fishtype, int>(SHRIMP, 9));
	tmp.insert(std::pair<Organism::fishtype, int>(JELLY, 8));
	tmp.insert(std::pair<Organism::fishtype, int>(EEL, 8));
	tmp.insert(std::pair<Organism::fishtype, int>(BALLOON, 8));
	tmp.insert(std::pair<Organism::fishtype, int>(CLOWN, 7));
	tmp.insert(std::pair<Organism::fishtype, int>(GTP, 9));
	tmp.insert(std::pair<Organism::fishtype, int>(MAGIKARP, 4));
	tmp.insert(std::pair<Organism::fishtype, int>(NARWHAL, 2));

	return tmp;
}

Organism::fishtype Organism::getType() {
    return type;
}

Plankton::Plankton(int x, int y, int a, int s, float gr, int fr, int v, Organism::fishtype t):
    Organism(x, y, a, s, gr, fr, v, t)
{
    //familyCount++;
}

ZPlankton::ZPlankton(int x, int y):Plankton(x, y, 1, 0, 1.0f, 5, 1, ZPL){
    count++;

	image.LoadFromFile("artwork/ZPlankton.png");
    sprite.SetImage(image);
    sprite.SetScaleX(0.25f);
    sprite.SetScaleY(0.25f);

    sprite.SetX(Helper::worldToPixel[x][y].x);
    sprite.SetY(Helper::worldToPixel[x][y].y);
}

PPlankton::PPlankton(int x, int y):Plankton(x, y, 1, 0, 0, 0, 1, PPL){
    count++;
	image.LoadFromFile("artwork/PPlankton.png");
    sprite.SetImage(image);
    sprite.SetScaleX(0.25f);
    sprite.SetScaleY(0.25f);

    sprite.SetX(Helper::worldToPixel[x][y].x);
    sprite.SetY(Helper::worldToPixel[x][y].y);
}

nonPlankton::nonPlankton(int x, int y, int a, int s, float gr, int fr, int v, Organism::fishtype t):
    Organism(x, y, a, s, gr, fr, v, t)
{
    //familyCount++;
}

Shrimp::Shrimp(int x, int y):nonPlankton(x, y, 1, 0, 1.0f, 4, 2, SHRIMP){
    count++;
	image.LoadFromFile("artwork/Shrimp.png");
    sprite.SetImage(image);
    sprite.SetScaleX(0.25f);
    sprite.SetScaleY(0.25f);

    sprite.SetX(Helper::worldToPixel[x][y].x);
    sprite.SetY(Helper::worldToPixel[x][y].y);
}

Jelly::Jelly(int x, int y):nonPlankton(x, y, 1, 0, 1.0f, 4, 1, JELLY){
    count++;
	image.LoadFromFile("artwork/Jelly.png");
    sprite.SetImage(image);
    sprite.SetScaleX(0.25f);
    sprite.SetScaleY(0.25f);

    sprite.SetX(Helper::worldToPixel[x][y].x);
    sprite.SetY(Helper::worldToPixel[x][y].y);
}

Eel::Eel(int x, int y):nonPlankton(x, y, 1, 0, 2.0f, 4, 5, EEL){
    count++;
	image.LoadFromFile("artwork/Eel.png");
    sprite.SetImage(image);
    sprite.SetScaleX(0.25f);
    sprite.SetScaleY(0.25f);

    sprite.SetX(Helper::worldToPixel[x][y].x);
    sprite.SetY(Helper::worldToPixel[x][y].y);
}

Balloon::Balloon(int x, int y):nonPlankton(x, y, 1, 0, 3.0f, 3, 3, BALLOON){
    count++;
	image.LoadFromFile("artwork/Baloon.png");
    sprite.SetImage(image);
    sprite.SetScaleX(0.25f);
    sprite.SetScaleY(0.25f);

    sprite.SetX(Helper::worldToPixel[x][y].x);
    sprite.SetY(Helper::worldToPixel[x][y].y);
}

Clown::Clown(int x, int y):nonPlankton(x, y, 1, 0, 2.0f, 3, 4, CLOWN){
    count++;
	image.LoadFromFile("artwork/Clown.png");
    sprite.SetImage(image);
    sprite.SetScaleX(0.25f);
    sprite.SetScaleY(0.25f);

    sprite.SetX(Helper::worldToPixel[x][y].x);
    sprite.SetY(Helper::worldToPixel[x][y].y);
}

Gtp::Gtp(int x, int y):nonPlankton(x, y, 1, 0, 2.0f, 3, 3, GTP){
    count++;
	image.LoadFromFile("artwork/Gtp.png");
    sprite.SetImage(image);
    sprite.SetScaleX(0.25f);
    sprite.SetScaleY(0.25f);

    sprite.SetX(Helper::worldToPixel[x][y].x);
    sprite.SetY(Helper::worldToPixel[x][y].y);
}

Magikarp::Magikarp(int x, int y):nonPlankton(x, y, 1, 0, 2.0f, 3, 4, MAGIKARP){
    count++;
	image.LoadFromFile("artwork/Magikarp.png");
    sprite.SetImage(image);
    sprite.SetScaleX(0.25f);
    sprite.SetScaleY(0.25f);

    sprite.SetX(Helper::worldToPixel[x][y].x);
    sprite.SetY(Helper::worldToPixel[x][y].y);
}

Narwhal::Narwhal(int x, int y):nonPlankton(x, y, 1, 0, 4.0f, 8, 6, NARWHAL){
    count++;
	image.LoadFromFile("artwork/Narwhal.png");
    sprite.SetImage(image);
    sprite.SetScaleX(0.25f);
    sprite.SetScaleY(0.25f);

    sprite.SetX(Helper::worldToPixel[x][y].x);
    sprite.SetY(Helper::worldToPixel[x][y].y);
}

int Organism::getX(){
    return x;
}
int Organism::getY(){
    return y;
}
void Organism::setX(int x){
    (*this).x=x;
    sprite.SetX(Helper::worldToPixel[x][y].x);
}
void Organism::setY(int y){
    (*this).y=y;
    sprite.SetY(Helper::worldToPixel[x][y].y);
}

