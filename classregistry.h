#ifndef CLASS_REGISTRY_H
#define CLASS_REGISTRY_H

#include "organism.h"

#include <map>

//pointer to function which returns pointer to Organism
typedef Organism* (*organism_creator)(int, int);

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
        static std::map<int, std::string> assocMapNames;
        static std::map<int, sf::Sprite> assocMapImages;

        //....add comment
        template<class className> static void associate(Organism::fishtype);
        static void registerClasses();
};

//returns custom constructor
template<class T> Organism* organism_factory(int x, int y);

#endif
