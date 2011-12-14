#ifndef HELPERS_H
#define HELPERS_H

#include <iostream>
#include <iomanip>
#include <SFML/Graphics.hpp>
#include <fstream>
#include "screens.h"
#include "Ocean.h"
#include <map>


using namespace std;
using namespace sf;


Vector2f** ArrayCon(const int x, const int y, const float w,const float h){

    /*
                     x=800 itiles pointer: j

          y=600     |--|--|--|--|--|--|--|--|--|--|
                    |--|--|--|--|--|--|--|--|--|--|
         jtiles     |--|--|--|--|--|--|--|--|--|--|
                    |--|--|--|--|--|--|--|--|--|--|
        pointer: i  |--|--|--|--|--|--|--|--|--|--|
                    |--|--|--|--|--|--|--|--|--|--|
                    |--|--|--|--|--|--|--|--|--|--|
         --     =    Vector2f
         itiles =    800/(size of tile+5)
         jtiles =    600/(size of tile+5)
         +5 to add a gap between fish
      */

    int itiles=x/(w+5);
    int jtiles=y/(h+5);
    int i,j;
    float xs=0.f,ys=0.f;
    sf::Vector2f **conv;

    //buffer allocation
    conv=new sf::Vector2f*[jtiles];
    for(int i=0;i<jtiles;++i)
        conv[i]=new sf::Vector2f[itiles];

    //Fill buffer with window coordinates
    for(i=0;i<itiles;i++){
        ys=0.f;
        for(j=0;j<jtiles;j++){
            conv[j][i].x=xs;
            conv[j][i].y=ys;
            ys+=h+5;
        }
        xs+=w+5;
    }

    //Print buffer
    for(i=0;i<itiles;i++){
        cout<<"|";
        for(j=0;j<jtiles;j++){
            cout<<setw(3);
            cout<<conv[j][i].x<<" "<<conv[j][i].y<<"|";
        }
        cout<<endl;
    }

    return conv;
}

/*      CORE ORGANISM CREATION FUNCTIONS


*/

//returns custom constructor
template<class T> Organism* organism_factory(float x, float y)
{
        return new T(x, y);
}

//pointer to function which returns pointer to Organism
typedef Organism* (*organism_creator)(float, float);

//FUCKING AWESOME CLASS::
class ClassRegistry {
private:
        //map with enum as a key and pointer to function which returns pointer to Organism
        static std::map<int, organism_creator> assocMap;

        //easy access to above map for searching
        typedef std::map<int, organism_creator>::iterator iter;
public:
        //returns constructor based on enum type
        static organism_creator getConstructor(int s);

        //....add comment
        template<class className> static void associate(Organism::fishtype);
};

//defines the map
std::map<int, organism_creator> ClassRegistry::assocMap;

organism_creator ClassRegistry::getConstructor(int s) {
        iter i = assocMap.find(s);
        if (i != assocMap.end()) {
                return i->second;
        }
        return NULL;
}

template<class className> void ClassRegistry::associate(Organism::fishtype code){
        assocMap.insert(std::pair<int, organism_creator>(code, &organism_factory<className>));
}


#endif // HELPERS_H
