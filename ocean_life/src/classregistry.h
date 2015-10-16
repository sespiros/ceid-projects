#ifndef CLASS_REGISTRY_H
#define CLASS_REGISTRY_H

#include "organism.h"

#include <map>

//pointer to function which returns pointer to Organism
typedef Organism* (*organism_creator)(int, int);

//Class for registering class types, image resources and names
class ClassRegistry {
private:
    //map with enum as a key and pointer to function which returns pointer to Organism
    static std::map<int, organism_creator> assocMap;

    //easy access to above map for searching
    typedef std::map<int, organism_creator>::iterator iter;
public:
    //map with enum as a key, returns name of the corresponding type
    static std::map<int, std::string> assocMapNames;

    //map with enum as a key, returns image of the corresponding type
    static std::map<int, sf::Image> assocMapImages;

    //returns constructor based on enum type
    static organism_creator getConstructor(int s);


    //insert classes into assocMap
    template<class className> static void associate(Organism::fishtype);

    //registers class types, image resources and names
    static void registerClasses();
};

//returns custom constructor
template<class T> Organism* organism_factory(int x, int y);

#endif
