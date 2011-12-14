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
    int getX();
    int getY();
    void setX(int );
    void setY(int );

    sf::Image image;
    sf::Sprite sprite;

    Organism(int, int, int, int, float, int, int, fishtype);

private:
    int x, y;
    int size, age;
    const float growthRate;
    const int foodRequired;
    int foodConsumed;
    const int velocity;
    const fishtype type;

};

class ZPlankton:public Organism
{

public:
    ZPlankton(int,int);
    static int count;

    int getCount(){
        return count;
    };
};

class PPlankton:public Organism
{

public:
    PPlankton(int,int);
    static int count;
    int getCount(){
        return count;
    };
};

class Shrimp:public Organism
{

public:
    Shrimp(int,int);
    static int count;
    int getCount(){
        return count;
    };
};

class Jelly:public Organism
{

public:
    Jelly(int,int);
    static int count;
    int getCount(){
        return count;
    };
};

class Eel:public Organism
{

public:
    Eel(int,int);
    static int count;
    int getCount(){
        return count;
    };
};

class Balloon:public Organism
{

public:
    Balloon(int,int);
    static int count;
    int getCount(){
        return count;
    };
};

class Clown:public Organism
{

public:
    Clown(int,int);
    static int count;
    int getCount(){
        return count;
    };
};

class Gtp:public Organism
{

public:
    Gtp(int,int);
    static int count;
    int getCount(){
        return count;
    };
};

class Magikarp:public Organism
{

public:
    Magikarp(int,int);
    static int count;
    int getCount(){
        return count;
    };
};

class Narwhal:public Organism
{

public:
    Narwhal(int,int);
    static int count;
    int getCount(){
        return count;
    };
};


#endif // ORGANISM_H
