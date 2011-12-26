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

	static const std::map<fishtype, int> weightMap;
    sf::Image image;
    sf::Sprite sprite;

    Organism(int, int, int, int, float, int, int, fishtype);

private:
	static std::map<fishtype, int> createWeightMap();

    int x, y;
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
     Plankton(int, int, int, int, float, int, int, fishtype);
     static int familyCount;

     int getFcount(){
         return familyCount;
     }
};


class ZPlankton:public Plankton
{

public:
    ZPlankton(int,int);
    static int count;

    int getCount(){
        return count;
    };
};

class PPlankton:public Plankton
{

public:
    PPlankton(int,int);
    static int count;
    int getCount(){
        return count;
    };
};

class nonPlankton:public Organism
{
public:
     nonPlankton(int,int, int, int, float, int, int, fishtype);
     static int familyCount;

     int getFcount(){
         return familyCount;
     }
};

class Shrimp:public nonPlankton
{

public:
    Shrimp(int,int);
    static int count;
    int getCount(){
        return count;
    };
};

class Jelly:public nonPlankton
{

public:
    Jelly(int,int);
    static int count;
    int getCount(){
        return count;
    };
};

class Eel:public nonPlankton
{

public:
    Eel(int,int);
    static int count;
    int getCount(){
        return count;
    };
};

class Balloon:public nonPlankton
{

public:
    Balloon(int,int);
    static int count;
    int getCount(){
        return count;
    };
};

class Clown:public nonPlankton
{

public:
    Clown(int,int);
    static int count;
    int getCount(){
        return count;
    };
};

class Gtp:public nonPlankton
{

public:
    Gtp(int,int);
    static int count;
    int getCount(){
        return count;
    };
};

class Magikarp:public nonPlankton
{

public:
    Magikarp(int,int);
    static int count;
    int getCount(){
        return count;
    };
};

class Narwhal:public nonPlankton
{

public:
    Narwhal(int,int);
    static int count;
    int getCount(){
        return count;
    };
};


#endif // ORGANISM_H
