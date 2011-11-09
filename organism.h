#ifndef ORGANISM_H
#define ORGANISM_H

class Organism
{

public:
    static unsigned int global_count;
    enum fishtype{ZPL,PPL,SHRIMP,JELLY,EEL,BALLOON,CLOWN,GTP,MAGIKARP,NATTY};

protected:
    int x, y;
    int age, size;
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

};

class PPlankton:public Organism
{

public:
    PPlankton(int,int);
    static int count;

};

class Shrimp:public Organism
{

public:
    Shrimp(int,int);
    static int count;

};

class Jelly:public Organism
{

public:
    Jelly(int,int);
    static int count;

};

class Eel:public Organism
{

public:
    Eel(int,int);
    static int count;

};

class Balloon:public Organism
{

public:
    Balloon(int,int);
    static int count;

};

class Clown:public Organism
{

public:
    Clown(int,int);
    static int count;

};

class Gtp:public Organism
{

public:
    Gtp(int,int);
    static int count;

};

class Magikarp:public Organism
{

public:
    Magikarp(int,int);
    static int count;

};

class Natty:public Organism
{

public:
    Natty(int,int);
    static int count;

};


#endif // ORGANISM_H
