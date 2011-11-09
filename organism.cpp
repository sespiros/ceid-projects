#include "organism.h"

ZPlankton::ZPlankton(int x, int y):growthRate(1),foodRequired(5),velocity(1),type(ZPL){
    size=1;
    age=0;
}

PPlankton::PPlankton(int x, int y):growthRate(0),foodRequired(0),velocity(1),type(PPL){
    size=1;
    age=0;
}

Shrimp::Shrimp(int x, int y):growthRate(1),foodRequired(4),velocity(2),type(SHRIMP){
    size=1;
    age=0;
}

Jelly::Jelly(int x, int y):growthRate(1),foodRequired(4),velocity(1),type(JELLY){
    size=1;
    age=0;
}

Eel::Eel(int x, int y):growthRate(2),foodRequired(4),velocity(5),type(EEL){
    size=1;
    age=0;
}

Balloon::Balloon(int x, int y):growthRate(3),foodRequired(3),velocity(3),type(BALLOON){
    size=1;
    age=0;
}

Clown::Clown(int x, int y):growthRate(2),foodRequired(3),velocity(4),type(CLOWN){
    size=1;
    age=0;
}

Gtp::Gtp(int x, int y):growthRate(2),foodRequired(3),velocity(3),type(GTP){
    size=1;
    age=0;
}

Magikarp::Magikarp(int x, int y):growthRate(2),foodRequired(3),velocity(4),type(MAGIKARP){
    size=1;
    age=0;
}

Natty::Natty(int x, int y):growthRate(4),foodRequired(8),velocity(6),type(NATTY){
    size=1;
    age=0;
}






