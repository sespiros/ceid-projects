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

Organism::Organism(int x, int y, int s, float gr, int fr, int v, Organism::fishtype t):
    ttl(s),growthRate(gr),foodRequired(fr),speed(v),type(t)
{
    (*this).x=x;
    (*this).y=y;

    foodConsumed = 0;
    foodConsumedWeek = 0;
    size = 1;
    age = 0;
    isDone = false;
}

std::map<Organism::fishtype, int> Organism::createWeightMap() {
    std::map<Organism::fishtype, int> tmp;

    tmp.insert(std::pair<Organism::fishtype, int>(PPL, 25));
    tmp.insert(std::pair<Organism::fishtype, int>(ZPL, 15));
    tmp.insert(std::pair<Organism::fishtype, int>(SHRIMP, 13));
    tmp.insert(std::pair<Organism::fishtype, int>(JELLY, 12));
    tmp.insert(std::pair<Organism::fishtype, int>(EEL, 5));
    tmp.insert(std::pair<Organism::fishtype, int>(BALLOON, 6));
    tmp.insert(std::pair<Organism::fishtype, int>(CLOWN, 7));
    tmp.insert(std::pair<Organism::fishtype, int>(GTP, 5));
    tmp.insert(std::pair<Organism::fishtype, int>(MAGIKARP, 8));
    tmp.insert(std::pair<Organism::fishtype, int>(NARWHAL, 4));

    return tmp;
}

Organism::fishtype Organism::getType() {
    return type;
}

Plankton::Plankton(int x, int y, int s, float gr, int fr, int v, Organism::fishtype t):
    Organism(x, y, s, gr, fr, v, t)
{
    //familyCount++;
}

ZPlankton::ZPlankton(int x, int y):Plankton(x, y, 300, 1.0f, 1, 1, ZPL){
    count++;

    eatField = 1 << Organism::PPL;

    sprite.SetImage(ClassRegistry::assocMapImages[0]);
    sprite.SetScale(0.25f, 0.25f);

    sprite.SetX(Helper::worldToPixel[x][y].x);
    sprite.SetY(Helper::worldToPixel[x][y].y);
}

void ZPlankton::kill() {
    count--;
    deaths++;
}

PPlankton::PPlankton(int x, int y):Plankton(x, y, 1000, 0, 0, 1, PPL){
    count++;

    eatField = 0;

    sprite.SetImage(ClassRegistry::assocMapImages[1]);
    sprite.SetScale(0.25f, 0.25f);

    sprite.SetX(Helper::worldToPixel[x][y].x);
    sprite.SetY(Helper::worldToPixel[x][y].y);
}

void PPlankton::kill() {
    count--;
    deaths++;
}

nonPlankton::nonPlankton(int x, int y, int s, float gr, int fr, int v, Organism::fishtype t):
    Organism(x, y, s, gr, fr, v, t)
{
    //familyCount++;
}

Shrimp::Shrimp(int x, int y):nonPlankton(x, y, 80, 1.0f, 1, 2, SHRIMP){
    count++;

    eatField = 1 << Organism::ZPL | 1 << Organism::PPL;

    sprite.SetImage(ClassRegistry::assocMapImages[2]);
    sprite.SetScale(0.25f, 0.25f);

    sprite.SetX(Helper::worldToPixel[x][y].x);
    sprite.SetY(Helper::worldToPixel[x][y].y);
}

void Shrimp::kill() {
    count--;
    deaths++;
}

Jelly::Jelly(int x, int y):nonPlankton(x, y, 80, 1.0f, 1, 1, JELLY){
    count++;

    eatField = (1 << Organism::PPL) | (1 << Organism::ZPL);

    sprite.SetImage(ClassRegistry::assocMapImages[3]);
    sprite.SetScale(0.25f, 0.25f);

    sprite.SetX(Helper::worldToPixel[x][y].x);
    sprite.SetY(Helper::worldToPixel[x][y].y);
}

void Jelly::kill() {
    count--;
    deaths++;
}

Eel::Eel(int x, int y):nonPlankton(x, y, 60, 1.0f, 1, 5, EEL){
    count++;
    eatField = (1 << Organism::BALLOON) | (1 << Organism::CLOWN) | (1 << Organism::GTP);

    sprite.SetImage(ClassRegistry::assocMapImages[4]);
    sprite.SetScale(0.25f, 0.25f);

    sprite.SetX(Helper::worldToPixel[x][y].x);
    sprite.SetY(Helper::worldToPixel[x][y].y);
}

void Eel::kill() {
    count--;
    deaths++;
}

Balloon::Balloon(int x, int y):nonPlankton(x, y, 70, 1.0f, 1, 3, BALLOON){
    count++;

    eatField = (1 << Organism::SHRIMP) | (1 << Organism::PPL) | (1 << Organism::ZPL) | (1 << Organism::JELLY);

    sprite.SetImage(ClassRegistry::assocMapImages[5]);
    sprite.SetScale(0.25f, 0.25f);

    sprite.SetX(Helper::worldToPixel[x][y].x);
    sprite.SetY(Helper::worldToPixel[x][y].y);
}

void Balloon::kill() {
    count--;
    deaths++;
}

Clown::Clown(int x, int y):nonPlankton(x, y, 70, 1.0f, 1, 4, CLOWN){
    count++;

    eatField = (1 << Organism::SHRIMP) | (1 << Organism::PPL) | (1 << Organism::ZPL) | (1 << Organism::JELLY);

    sprite.SetImage(ClassRegistry::assocMapImages[6]);
    sprite.SetScale(0.25f, 0.25f);

    sprite.SetX(Helper::worldToPixel[x][y].x);
    sprite.SetY(Helper::worldToPixel[x][y].y);
}

void Clown::kill() {
    count--;
    deaths++;
}

Gtp::Gtp(int x, int y):nonPlankton(x, y, 70, 1.0f, 1, 3, GTP){
    count++;

    eatField = (1 << Organism::SHRIMP) | (1 << Organism::PPL) | (1 << Organism::ZPL) | (1 << Organism::JELLY);

    sprite.SetImage(ClassRegistry::assocMapImages[7]);
    sprite.SetScale(0.25f, 0.25f);

    sprite.SetX(Helper::worldToPixel[x][y].x);
    sprite.SetY(Helper::worldToPixel[x][y].y);
}

void Gtp::kill() {
    count--;
    deaths++;
}

Magikarp::Magikarp(int x, int y):nonPlankton(x, y, 70, 1.0f, 1, 4, MAGIKARP){
    count++;

    eatField = (1 << Organism::SHRIMP) | (1 << Organism::PPL) | (1 << Organism::ZPL) | (1 << Organism::JELLY);;

    sprite.SetImage(ClassRegistry::assocMapImages[8]);
    sprite.SetScale(0.25f, 0.25f);

    sprite.SetX(Helper::worldToPixel[x][y].x);
    sprite.SetY(Helper::worldToPixel[x][y].y);
}

void Magikarp::kill() {
    count--;
    deaths++;
}

Narwhal::Narwhal(int x, int y):nonPlankton(x, y, 50, 1.0f, 4, 6, NARWHAL){
    count++;

    eatField = ~0 & ~(1 << Organism::ZPL | 1 << Organism::PPL | 1 << Organism::NARWHAL);

    sprite.SetImage(ClassRegistry::assocMapImages[9]);
    sprite.SetScale(0.25f, 0.25f);

    sprite.SetX(Helper::worldToPixel[x][y].x);
    sprite.SetY(Helper::worldToPixel[x][y].y);
}

void Narwhal::kill() {
    count--;
    deaths++;
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

int Organism::getFoodConsumedWeek() {
    return foodConsumedWeek;
}

int Organism::getAge() {
    return age;
}

int Organism::getGrowthRate(){
    return growthRate;
}

int Organism::getFoodRequiredPerWeek(){
    return foodRequired*7;
}

void Organism::setX(int x){
    (*this).x=x;
    //sprite.SetX(Helper::worldToPixel[x][y].x);
}
void Organism::setY(int y){
    (*this).y=y;
    //sprite.SetY(Helper::worldToPixel[x][y].y);
}

void Organism::eat(Organism* o) {
    foodConsumed += o->size;
    foodConsumedWeek += o->size;

    if (foodConsumed >= foodRequired) {
        size += (foodConsumed/foodRequired)*growthRate;
        ttl += 50;
        foodConsumed = foodConsumed%foodRequired;
        std::stringstream ss;
        ss<<"Size inc! for "<< o;
        Ocean::regLog(ss.str());
    }
}

void Organism::weeklyReset() {
    foodConsumedWeek = 0;
}

bool Organism::levelUpCheck() {
    age++;
    isDone = false;
    ttl--;

    if(Ocean::easterEggs){
        if(age > 5 && size > 30 && type == MAGIKARP){
            Ocean::regLog("Gyarados evolve");
            eatField = 1023;
            sprite.SetImage(ClassRegistry::assocMapImages[10]);
            type = Organism::GYARADOS;
            ttl = 500;
        }
    }

    if(ttl <= 0){
        int hash = (*this).x + (*this).y * Ocean::MAX_X;
        isDone = true;
        std::stringstream ss;
        ss<<"Fish "<< hash <<"died from starvation";
        Ocean::regLog(ss.str());
        return true;
    }
    return false;
}

bool Organism::canEat(Organism* o) {
    if ((eatField & (1 << o->type)) != 0)
        return true;
    else
        return false;
}

int Organism::getWeightof(Organism::fishtype a){
    //std::map<Organism::fishtype, int> tmp = Organism::weightMap
    return 5;
}

