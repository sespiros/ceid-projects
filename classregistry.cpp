#include <map>

#include "classregistry.h"

using std::map;
using std::pair;

//defines the map
map<int, organism_creator> ClassRegistry::assocMap;
map<int, std::string> ClassRegistry::assocMapNames;
map<int, sf::Image> ClassRegistry::assocMapImages;

//returns custom constructor
template<class T> Organism* organism_factory(int x, int y)
{
    return new T(x, y);
}

organism_creator ClassRegistry::getConstructor(int s) {
    iter i = assocMap.find(s);
    if (i != assocMap.end()) {
        return i->second;
    }
    return NULL;
}

/*
 * Extensible by adding another map insertion here
 */
template<class className> void ClassRegistry::associate(Organism::fishtype code){
    assocMap.insert(pair<int, organism_creator>(code, &organism_factory<className>));
}

void ClassRegistry::registerClasses() {
    ClassRegistry::associate<ZPlankton>(Organism::ZPL);
    ClassRegistry::associate<PPlankton>(Organism::PPL);
    ClassRegistry::associate<Shrimp>(Organism::SHRIMP);
    ClassRegistry::associate<Jelly>(Organism::JELLY);
    ClassRegistry::associate<Eel>(Organism::EEL);
    ClassRegistry::associate<Balloon>(Organism::BALLOON);
    ClassRegistry::associate<Clown>(Organism::CLOWN);
    ClassRegistry::associate<Gtp>(Organism::GTP);
    ClassRegistry::associate<Magikarp>(Organism::MAGIKARP);
    ClassRegistry::associate<Narwhal>(Organism::NARWHAL);

    assocMapNames.insert(pair<int, std::string>(0, "Zoo Planktons"));
    assocMapNames.insert(pair<int, std::string>(1, "Plant Planktons"));
    assocMapNames.insert(pair<int, std::string>(2, "Shrimps"));
    assocMapNames.insert(pair<int, std::string>(3, "Jellies"));
    assocMapNames.insert(pair<int, std::string>(4, "Eels"));
    assocMapNames.insert(pair<int, std::string>(5, "Balloon fishes"));
    assocMapNames.insert(pair<int, std::string>(6, "Clown fishes"));
    assocMapNames.insert(pair<int, std::string>(7, "Giatonp**** fishes"));
    assocMapNames.insert(pair<int, std::string>(8, "Magikarps"));
    assocMapNames.insert(pair<int, std::string>(9, "Narwhals"));

    sf::Image image;
    image.LoadFromFile("artwork/ZPlankton.png");
    assocMapImages.insert(pair<int, sf::Image>(0,image));
    image.LoadFromFile("artwork/PPlankton.png");
    assocMapImages.insert(pair<int, sf::Image>(1,image));
    image.LoadFromFile("artwork/Shrimp.png");
    assocMapImages.insert(pair<int, sf::Image>(2,image));
    image.LoadFromFile("artwork/Jelly.png");
    assocMapImages.insert(pair<int, sf::Image>(3,image));
    image.LoadFromFile("artwork/Eel.png");
    assocMapImages.insert(pair<int, sf::Image>(4,image));
    image.LoadFromFile("artwork/Balloon.png");
    assocMapImages.insert(pair<int, sf::Image>(5,image));
    image.LoadFromFile("artwork/Clown.png");
    assocMapImages.insert(pair<int, sf::Image>(6,image));
    image.LoadFromFile("artwork/Gtp.png");
    assocMapImages.insert(pair<int, sf::Image>(7,image));
    image.LoadFromFile("artwork/Magikarp.png");
    assocMapImages.insert(pair<int, sf::Image>(8,image));
    image.LoadFromFile("artwork/Narwhal.png");
    assocMapImages.insert(pair<int, sf::Image>(9,image));



}

