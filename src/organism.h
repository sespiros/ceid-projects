#ifndef ORGANISM_H
#define ORGANISM_H
#include <SFML/Graphics.hpp>


class Organism
{

public:
    enum fishtype{ZPL,PPL,SHRIMP,JELLY,EEL,BALLOON,CLOWN,GTP,MAGIKARP,NARWHAL};

    fishtype getType();
    virtual int getCount() = 0;
    virtual int getDeaths() = 0;
    //virtual sf::Sprite getSprite()=0;
    int getX();
    int getY();
    int getSpeed();
    int getSize();
	int getFoodConsumedWeek();
	int getAge();
    int getGrowthRate();
    int getFoodRequiredPerWeek();
    int getWeightof(Organism::fishtype);
    void setX(int );
    void setY(int );
    bool canEat(Organism*);
    virtual void kill() = 0;
    void eat(Organism*);
    bool levelUpCheck();
	void weeklyReset();

	static const std::map<fishtype, int> weightMap;
	sf::Sprite sprite;
    bool isDone;

    Organism(int, int, int, float, int, int, fishtype);

protected:
    int eatField;

private:
    static std::map<fishtype, int> createWeightMap();

    int x, y;
    int size;
    int ttl; //timetolive like a pro
	int age;
    const float growthRate;
    const int foodRequired;
	int foodConsumed, foodConsumedWeek;
    const int speed;
    const fishtype type;
};

class Plankton:public Organism
{
public:
    Plankton(int, int, int, float, int, int, fishtype);
    static int familyCount;

    int getFcount(){
        return familyCount;
    }
};


class ZPlankton:public Plankton
{

public:
    ZPlankton(int,int);
    virtual void kill();
    static int count;
    static int deaths;

    int getCount(){
        return count;
    };

    int getDeaths(){
        return deaths;
    };
    static void reset() {
        deaths = 0;
        count = 0;
    };
};

class PPlankton:public Plankton
{

public:
    PPlankton(int,int);
    virtual void kill();
    static int count;
    static int deaths;

    int getCount(){
        return count;
    };

    int getDeaths(){
        return deaths;
    };
    static void reset() {
        deaths = 0;
        count = 0;
    };
};

class nonPlankton:public Organism
{
public:
    nonPlankton(int, int, int, float, int, int, fishtype);
    static int familyCount;

    int getFcount(){
        return familyCount;
    }
};

class Shrimp:public nonPlankton
{

public:
    Shrimp(int,int);
    virtual void kill();
    static int count;
    static int deaths;

    int getCount(){
        return count;
    };

    int getDeaths(){
        return deaths;
    };
    static void reset() {
        deaths = 0;
        count = 0;
    };
};

class Jelly:public nonPlankton
{

public:
    Jelly(int,int);
    virtual void kill();
    static int count;
    static int deaths;

    int getCount(){
        return count;
    };

    int getDeaths(){
        return deaths;
    };
    static void reset() {
        deaths = 0;
        count = 0;
    };
};

class Eel:public nonPlankton
{

public:
    Eel(int,int);
    virtual void kill();
    static int count;
    static int deaths;

    int getCount(){
        return count;
    };

    int getDeaths(){
        return deaths;
    };
    static void reset() {
        deaths = 0;
        count = 0;
    };
};

class Balloon:public nonPlankton
{

public:
    Balloon(int,int);
    virtual void kill();
    static int count;
    static int deaths;

    int getCount(){
        return count;
    };

    int getDeaths(){
        return deaths;
    };
    static void reset() {
        deaths = 0;
        count = 0;
    };
};

class Clown:public nonPlankton
{

public:
    Clown(int,int);
    virtual void kill();
    static int count;
    static int deaths;

    int getCount(){
        return count;
    };

    int getDeaths(){
        return deaths;
    };
    static void reset() {
        deaths = 0;
        count = 0;
    };
};

class Gtp:public nonPlankton
{

public:
    Gtp(int,int);
    virtual void kill();
    static int count;
    static int deaths;

    int getCount(){
        return count;
    };

    int getDeaths(){
        return deaths;
    };
    static void reset() {
        deaths = 0;
        count = 0;
    };
};

class Magikarp:public nonPlankton
{

public:
    Magikarp(int,int);
    virtual void kill();
    static int count;
    static int deaths;

    int getCount(){
        return count;
    };

    int getDeaths(){
        return deaths;
    };
    static void reset() {
        deaths = 0;
        count = 0;
    };
};

class Narwhal:public nonPlankton
{

public:
    Narwhal(int,int);
    virtual void kill();
    static int count;
    static int deaths;

    int getCount(){
        return count;
    };

    int getDeaths(){
        return deaths;
    };
    static void reset() {
        deaths = 0;
        count = 0;
    };
};


#endif // ORGANISM_H
