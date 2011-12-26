#include <map>

#include "ClassRegistry.h"

using std::map;
using std::pair;

//defines the map
map<int, organism_creator> ClassRegistry::assocMap;

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
}

