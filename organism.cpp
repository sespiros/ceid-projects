#include "organism.h"

int ZPlankton::count=0;
int PPlankton::count=0;
int Shrimp::count=0;
int Jelly::count=0;
int Eel::count=0;
int Balloon::count=0;
int Clown::count=0;
int Gtp::count=0;
int Magikarp::count=0;
int Natty::count=0;

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
}

PPlankton::PPlankton(int x, int y):Organism(x, y, 1, 0, 0, 0, 1, PPL){
    count++;
}

Shrimp::Shrimp(int x, int y):Organism(x, y, 1, 0, 1.0f, 4, 2, SHRIMP){
    count++;
}

Jelly::Jelly(int x, int y):Organism(x, y, 1, 0, 1.0f, 4, 1, JELLY){
    count++;
}

Eel::Eel(int x, int y):Organism(x, y, 1, 0, 2.0f, 4, 5, EEL){
    count++;
}

Balloon::Balloon(int x, int y):Organism(x, y, 1, 0, 3.0f, 3, 3, BALLOON){
    count++;
}

Clown::Clown(int x, int y):Organism(x, y, 1, 0, 2.0f, 3, 4, CLOWN){
    count++;
}

Gtp::Gtp(int x, int y):Organism(x, y, 1, 0, 2.0f, 3, 3, GTP){
    count++;
}

Magikarp::Magikarp(int x, int y):Organism(x, y, 1, 0, 2.0f, 3, 4, MAGIKARP){
    count++;
}

Natty::Natty(int x, int y):Organism(x, y, 1, 0, 4.0f, 8, 6, NATTY){
    count++;
}

int Organism::getX(){
    return x;
}
int Organism::getY(){
    return y;
}
int Organism::setX(int x){
    (*this).x=x;
}
int Organism::setY(int y){
    (*this).y=y;
}

