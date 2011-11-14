#include "organism.h"

unsigned int Organism::global_count = 0;

Organism::Organism(int x, int y, int a, int s, float gr, int fr, int v, Organism::fishtype t):
    size(s),age(a),growthRate(gr),foodRequired(fr),velocity(v),type(t)
{
    global_count++;
}

Organism::fishtype Organism::getType() {
    return type;
}

ZPlankton::ZPlankton(int x, int y):Organism(x, y, 1, 0, 1.0f, 5, 1, ZPL){

}

PPlankton::PPlankton(int x, int y):Organism(x, y, 1, 0, 0, 0, 1, PPL){

}

Shrimp::Shrimp(int x, int y):Organism(x, y, 1, 0, 1.0f, 4, 2, SHRIMP){

}

Jelly::Jelly(int x, int y):Organism(x, y, 1, 0, 1.0f, 4, 1, JELLY){

}

Eel::Eel(int x, int y):Organism(x, y, 1, 0, 2.0f, 4, 5, EEL){

}

Balloon::Balloon(int x, int y):Organism(x, y, 1, 0, 3.0f, 3, 3, BALLOON){

}

Clown::Clown(int x, int y):Organism(x, y, 1, 0, 2.0f, 3, 4, CLOWN){

}

Gtp::Gtp(int x, int y):Organism(x, y, 1, 0, 2.0f, 3, 3, GTP){

}

Magikarp::Magikarp(int x, int y):Organism(x, y, 1, 0, 2.0f, 3, 4, MAGIKARP){

}

Natty::Natty(int x, int y):Organism(x, y, 1, 0, 4.0f, 8, 6, NATTY){

}
