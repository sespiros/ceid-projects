#ifndef ORGANISM_H
#define ORGANISM_H
#include <SFML/Graphics.hpp>

class Organism
{

public:
    enum fishtype{ZPL,PPL,SHRIMP,JELLY,EEL,BALLOON,CLOWN,GTP,MAGIKARP,NARWHAL};

    fishtype getType();
    virtual int getCount()=0;
    //virtual sf::Sprite getSprite()=0;
    float getX();
    float getY();
    void setX(float );
    void setY(float );

    sf::Image image;
    sf::Sprite sprite;

    Organism(float, float, int, int, float, int, int, fishtype);

private:
    float x, y;
    int size, age;
    const float growthRate;
    const int foodRequired;
    int foodConsumed;
    const int velocity;
    const fishtype type;

};

class Plankton:public Organism
{
public:
     Plankton(float, float, int, int, float, int, int, fishtype);
     static int familyCount;

     int getFcount(){
         return familyCount;
     }
};


class ZPlankton:public Plankton
{

public:
    ZPlankton(float,float);
    static int count;

    int getCount(){
        return count;
    };
};

class PPlankton:public Plankton
{

public:
    PPlankton(float,float);
    static int count;
    int getCount(){
        return count;
    };
};

class nonPlankton:public Organism
{
public:
     nonPlankton(float,float, int, int, float, int, int, fishtype);
     static int familyCount;

     int getFcount(){
         return familyCount;
     }
};

class Shrimp:public nonPlankton
{

public:
    Shrimp(float,float);
    static int count;
    int getCount(){
        return count;
    };
};

class Jelly:public nonPlankton
{

public:
    Jelly(float,float);
    static int count;
    int getCount(){
        return count;
    };
};

class Eel:public nonPlankton
{

public:
    Eel(float,float);
    static int count;
    int getCount(){
        return count;
    };
};

class Balloon:public nonPlankton
{

public:
    Balloon(float,float);
    static int count;
    int getCount(){
        return count;
    };
};

class Clown:public nonPlankton
{

public:
    Clown(float,float);
    static int count;
    int getCount(){
        return count;
    };
};

class Gtp:public nonPlankton
{

public:
    Gtp(float,float);
    static int count;
    int getCount(){
        return count;
    };
};

class Magikarp:public nonPlankton
{

public:
    Magikarp(float,float);
    static int count;
    int getCount(){
        return count;
    };
};

class Narwhal:public nonPlankton
{

public:
    Narwhal(float,float);
    static int count;
    int getCount(){
        return count;
    };
};


#endif // ORGANISM_H
