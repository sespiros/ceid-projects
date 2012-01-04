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

int ZPlankton::deaths=0;
int PPlankton::deaths=0;
int Shrimp::deaths=0;
int Jelly::deaths=0;
int Eel::deaths=0;
int Balloon::deaths=0;
int Clown::deaths=0;
int Gtp::deaths=0;
int Magikarp::deaths=0;
int Narwhal::deaths=0;

const std::map<Organism::fishtype, int> Organism::weightMap = Organism::createWeightMap();

Organism::Organism(int x, int y, int a, int s, float gr, int fr, int v, Organism::fishtype t):
    size(s),age(a),growthRate(gr),foodRequired(fr),speed(v),type(t)
{
    (*this).x=x;
    (*this).y=y;

    foodConsumed = 0;
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
    eatField = 0x0;
    //familyCount++;
}

ZPlankton::ZPlankton(int x, int y):Plankton(x, y, 1, 1, 1.0f, 5, 1, ZPL){
    count++;
    age = 0;

    sprite.SetImage(ClassRegistry::assocMapImages[0]);
    sprite.SetScale(0.25f, 0.25f);

    sprite.SetX(Helper::worldToPixel[x][y].x);
    sprite.SetY(Helper::worldToPixel[x][y].y);
}

void ZPlankton::kill() {
    count--;
}

PPlankton::PPlankton(int x, int y):Plankton(x, y, 1, 1, 0, 0, 1, PPL){
    count++;
    age = 0;

    sprite.SetImage(ClassRegistry::assocMapImages[1]);
    sprite.SetScale(0.25f, 0.25f);

    sprite.SetX(Helper::worldToPixel[x][y].x);
    sprite.SetY(Helper::worldToPixel[x][y].y);
}

void PPlankton::kill() {
    count--;
    deaths++;
}

nonPlankton::nonPlankton(int x, int y, int a, int s, float gr, int fr, int v, Organism::fishtype t):
    Organism(x, y, a, s, gr, fr, v, t)
{
    //familyCount++;
}

Shrimp::Shrimp(int x, int y):nonPlankton(x, y, 1, 1, 1.0f, 4, 2, SHRIMP){
    count++;
    age = 0;

    eatField = 1 << Organism::ZPL;

    sprite.SetImage(ClassRegistry::assocMapImages[2]);
    sprite.SetScale(0.25f, 0.25f);

    sprite.SetX(Helper::worldToPixel[x][y].x);
    sprite.SetY(Helper::worldToPixel[x][y].y);
}

void Shrimp::kill() {
    count--;
}

Jelly::Jelly(int x, int y):nonPlankton(x, y, 1, 1, 1.0f, 4, 1, JELLY){
    count++;
    age = 0;

    eatField = 1 << Organism::PPL;

    sprite.SetImage(ClassRegistry::assocMapImages[3]);
    sprite.SetScale(0.25f, 0.25f);

    sprite.SetX(Helper::worldToPixel[x][y].x);
    sprite.SetY(Helper::worldToPixel[x][y].y);
}

void Jelly::kill() {
    count--;
}

Eel::Eel(int x, int y):nonPlankton(x, y, 1, 1, 2.0f, 4, 5, EEL){
    count++;
    age = 0;

    eatField = (1 << Organism::SHRIMP) | (1 << Organism::GTP);

    sprite.SetImage(ClassRegistry::assocMapImages[4]);
    sprite.SetScale(0.25f, 0.25f);

    sprite.SetX(Helper::worldToPixel[x][y].x);
    sprite.SetY(Helper::worldToPixel[x][y].y);
}

void Eel::kill() {
    count--;
}

Balloon::Balloon(int x, int y):nonPlankton(x, y, 1, 1, 3.0f, 3, 3, BALLOON){
    count++;
    age = 0;

    eatField = (1 << Organism::SHRIMP) | (1 << Organism::GTP);

    sprite.SetImage(ClassRegistry::assocMapImages[5]);
    sprite.SetScale(0.25f, 0.25f);

    sprite.SetX(Helper::worldToPixel[x][y].x);
    sprite.SetY(Helper::worldToPixel[x][y].y);
}

void Balloon::kill() {
    count--;
}

Clown::Clown(int x, int y):nonPlankton(x, y, 1, 1, 2.0f, 3, 4, CLOWN){
    count++;
    age = 0;

    eatField = (1 << Organism::ZPL) | (1 << Organism::GTP);

    sprite.SetImage(ClassRegistry::assocMapImages[6]);
    sprite.SetScale(0.25f, 0.25f);

    sprite.SetX(Helper::worldToPixel[x][y].x);
    sprite.SetY(Helper::worldToPixel[x][y].y);
}

void Clown::kill() {
    count--;
}

Gtp::Gtp(int x, int y):nonPlankton(x, y, 1, 1, 2.0f, 3, 3, GTP){
    count++;
    age = 0;

    eatField = (1 << Organism::PPL);

    sprite.SetImage(ClassRegistry::assocMapImages[7]);
    sprite.SetScale(0.25f, 0.25f);

    sprite.SetX(Helper::worldToPixel[x][y].x);
    sprite.SetY(Helper::worldToPixel[x][y].y);
}

void Gtp::kill() {
    count--;
}

Magikarp::Magikarp(int x, int y):nonPlankton(x, y, 1, 1, 2.0f, 3, 4, MAGIKARP){
    count++;
    age = 0;

    eatField = (1 << Organism::ZPL) | (1 << Organism::JELLY);

    sprite.SetImage(ClassRegistry::assocMapImages[8]);
    sprite.SetScale(0.25f, 0.25f);

    sprite.SetX(Helper::worldToPixel[x][y].x);
    sprite.SetY(Helper::worldToPixel[x][y].y);
}

void Magikarp::kill() {
    count--;
}

Narwhal::Narwhal(int x, int y):nonPlankton(x, y, 1, 1, 4.0f, 8, 6, NARWHAL){
    count++;
    age = 0;

    eatField = 0xfffffffc;

    sprite.SetImage(ClassRegistry::assocMapImages[9]);
    sprite.SetScale(0.25f, 0.25f);

    sprite.SetX(Helper::worldToPixel[x][y].x);
    sprite.SetY(Helper::worldToPixel[x][y].y);
}

void Narwhal::kill() {
    count--;
}

int Organism::getX(){
    return x;
}
int Organism::getY(){
    return y;
}

int Organism::getSpeed() {
    return speed;
}

int Organism::getSize(){
    return size;
}

void Organism::setX(int x){
    (*this).x=x;
    sprite.SetX(Helper::worldToPixel[x][y].x);
}
void Organism::setY(int y){
    (*this).y=y;
    sprite.SetY(Helper::worldToPixel[x][y].y);
}

void Organism::eat(Organism* o) {
    foodConsumed += o->size;
    foodConsumedWeek += o->size;

    if (foodConsumed >= foodRequired) {
        size += foodConsumed/foodRequired;
        foodConsumed = foodConsumed%foodRequired;
        std::cout << "Size inc! for " << o->size << std::endl;
    }
}

bool Organism::canEat(Organism* o) {
    if ((eatField & (1 << o->type)) != 0)
        return true;
    else
        return false;
}

