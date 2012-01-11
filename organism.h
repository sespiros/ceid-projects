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
    void levelUp();
	void weeklyReset();

	static const std::map<fishtype, int> weightMap;
	sf::Sprite sprite;

    Organism(int, int, int, int, float, int, int, fishtype);

protected:
    int eatField;

private:
    static std::map<fishtype, int> createWeightMap();

    int x, y;
    int size;
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
    virtual void kill();
    static int count;
    static int deaths;

    int getCount(){
        return count;
    };

    int getDeaths(){
        return deaths;
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
    virtual void kill();
    static int count;
    static int deaths;

    int getCount(){
        return count;
    };

    int getDeaths(){
        return deaths;
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
};


#endif // ORGANISM_H
