#include "Ocean.h"
#include "classregistry.h"
#include "helper.h"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <string.h>
#include <sstream>

using  std::map;
using  std::pair;
/*
 * static declarations
 */
int Ocean::count = 0;
int Ocean::deaths = 0;
int Ocean::turns = 0;
std::map<int, Organism*> Ocean::fishMap;
std::vector<Pollution*> Ocean::pollution;
float Ocean::averageCategorySize[10];
float Ocean::averageConsumptionWeek[10];
float Ocean::averageDeathRate[10];
float Ocean::averageAge[10];
sf::String Ocean::categories[10];

void Ocean::init() {
    ClassRegistry::registerClasses();

    //this variable is for setting y coordinate for the strings of stats
    int y = 30;

    //Stats Arrays initialization
    for(int i = 0; i < 10 ; i++){
        averageCategorySize[i] = 0;
        averageConsumptionWeek[i] = 0;
        averageDeathRate[i] = 0;
        averageAge[i] = 0;

        //Stats String Categories Arrays initialization
        categories[i].SetText(ClassRegistry::assocMapNames[i]);
        categories[i].SetPosition(830,y);
        categories[i].SetColor(sf::Color::Black);
        categories[i].SetSize(11);
        y += 53;

        //Images initialization


    }

    Ocean::populate();
}

void Ocean::add(Organism *toAdd) {
    int hash = toAdd->getX() + toAdd->getY() * MAX_X;
    fishMap.insert(pair<int, Organism*>(hash, toAdd));
    count++;
}

void Ocean::kill(int key) {
    fishMap[key]->kill();
    delete fishMap[key];
    fishMap.erase(key);
    count--;
    deaths++;
}

void Ocean::createAndAddFish(int t, int x, int y) {
    organism_creator f = ClassRegistry::getConstructor(t);
    Ocean::add(f(x, y));
}

void Ocean::createAndAddRandFish(int x, int y) {
    organism_creator f = ClassRegistry::getConstructor(genRandType());
    Ocean::add(f(x, y));
}

void Ocean::populate() {
    for (int i = 0; i < Ocean::MAX_COUNT; i++) {
        int x = rand()%(Ocean::MAX_X - 1);
        int y = rand()%(Ocean::MAX_Y - 1);

        while (fishMap.count(x + y*MAX_X) != 0) {
            x = rand()%(Ocean::MAX_X - 1);
            y = rand()%(Ocean::MAX_Y - 1);
        }
        Ocean::createAndAddRandFish(x, y);
    }
}

Organism::fishtype Ocean::genRandType() {
    std::map<Organism::fishtype, int>::const_iterator it;
    int sum = 0;
    for (it = Organism::weightMap.begin(); it != Organism::weightMap.end(); it++) {
        sum += it->second;
    }

    int rnd = rand()%sum;
    for (it = Organism::weightMap.begin(); it != Organism::weightMap.end(); it++) {
        rnd -= it->second;
        if (rnd < 0)
            return it->first;
    }
    return (Organism::fishtype)0;
}

mapIter Ocean::move(int key, int x, int y) {
    int hash = x + y * MAX_X;
    mapIter it = Ocean::fishMap.find(key);

    it->second->setX(x);
    it->second->setY(y);

    fishMap.insert(pair<int, Organism*>(hash, fishMap[key]));
    //not Ocean::kill because in Ocean::we move an object and we don't want count--
    fishMap.erase(it++);
    return it;
}

mapIter Ocean::collide(int key){
    int curX = Ocean::fishMap[key]->getX();
    int curY = Ocean::fishMap[key]->getY();
    mapIter next = Ocean::fishMap.find(key);

    int dx, dy, x, y, hash;
    bool hasMoved = false;
    bool hasEat = false;
    bool hasBred = false;


    for (int i = 7; i > 0; i--) {
        int j = rand()%(i + 1);
        dx = Helper::dir[j][0];
        dy = Helper::dir[j][1];
        Helper::swapDir(j, i);

        if (!isValid(curX + dx, Ocean::MAX_X) || !isValid(curY + dy, Ocean::MAX_Y))
            continue;

        x = curX + dx;
        y = curY + dy;

        hash = x + y * MAX_X;
        if(Ocean::fishMap.find(hash) != Ocean::fishMap.end()){
            if(Ocean::fishMap[key]->canEat(Ocean::fishMap[hash])){
                hasEat = true;
            }else if(Ocean::fishMap[hash]->getType() == Ocean::fishMap[key]->getType()){
                hasBred = true;
            }
        }else{
            hasMoved = true;
        }

        if(hasEat || hasBred || hasMoved)
            break;
    }

    if(hasEat){
        Ocean::fishMap[key]->eat(Ocean::fishMap[hash]);
        kill(hash);
        next = move(key, x, y);
        //add stats print
    }else if(hasBred){
        //to be added
        //add stats print
        next++;
    }else if(hasMoved){
        next = move(key, x, y);
        //add stats print
    }
    else {
        next++;
    }
    return next;
}

void Ocean::tickPollution() {
    for (std::vector<Pollution*>::const_iterator i = Ocean::pollution.begin(); i != Ocean::pollution.end(); ++i) {
        Pollution* p = *i;
        int cy = p->getY();
        int cx = p->getX();
        int r = p->getRadius();

        for (int j = -r; j <= r; j++) {
            for (int k = abs(j) - r; k <= r - abs(j); k++) {
                if (!Ocean::isValid(cy + j, MAX_Y) || !Ocean::isValid(cx + k, MAX_X))
                    continue;
                if (Ocean::fishMap.count((cx + k) + (cy + j)*MAX_X) != 0)
                    Ocean::kill((cx + k) + (cy + j)*MAX_X);
            }
        }

        p->tick();
    }

    Ocean::pollution.erase(std::remove_if(Ocean::pollution.begin(), Ocean::pollution.end(), Pollution::isDone), Ocean::pollution.end());
}

void Ocean::update() {
    srand(time(0));
    mapIter it;

    tickPollution();

    for (it = Ocean::fishMap.begin(); it != Ocean::fishMap.end(); ) {
        it = collide(it->first);
    }

    Ocean::stats();


}

void Ocean::info() {
    std::cout << Ocean::count << " fish." << std::endl;
    mapIter it;
    //	for (it = fishMap.begin(); it != fishMap.end(); it++){
    //		std::cout << "Fish " << i++ << " is of type " << it->second->getType() << std::endl;
    //		std::cout << "Position: " << it->second->getX() << " " << it->second->getY() << std::endl;
    //	}
}

bool Ocean::isValid(int a, int max)
{
    return ((a >= 0) && (a < max));
}

void Ocean::pollute(int r, int x, int y)
{
    Pollution* p = new Pollution(r, x, y);
    Ocean::pollution.insert(Ocean::pollution.begin(), p);
    //debug
    //std::cout << "Inserted pollution source at (" << x << ", " << y << "), radius " << r << std::endl;
}

void Ocean::stats(){

    //Average size of category
    //Average consumption last week
    //AverageDeathRate

    mapIter it;
    for(it = fishMap.begin(); it != fishMap.end();it++){
        averageCategorySize [it->second->getType()] += it->second->getSize()/it->second->getCount();
        averageDeathRate [it->second->getType()] += it->second->getDeaths()*(100.0f/Ocean::deaths);
        averageAge [it->second->getType()] += it->second->age/it->second->getCount();
        it->second->age++;
    }

    if(turns%7 == 0){
        for(it = fishMap.begin(); it != fishMap.end();it++){
            averageConsumptionWeek [it->second->getType()] += it->second->foodConsumedWeek/it->second->getCount();
            it->second->foodConsumedWeek = 0;
        }
    }



}

void Ocean::drawStats(sf::RenderWindow *o, bool choice, bool choice2){
    sf::String StatsTitle;
    sf::String clickToExpand;
    sf::String identifier;
    sf::Image Log;
    sf::Sprite spriteLog;
    StatsTitle.SetText("Stats:");
    StatsTitle.SetSize(20);
    StatsTitle.SetPosition(830,10);         //830-left 895-center 10-top
    StatsTitle.SetColor(sf::Color::Black);
    identifier.SetSize(15);
    identifier.SetPosition(885,15);         //830-left 895-center 10-top
    identifier.SetColor(sf::Color::Black);
    clickToExpand.SetSize(15);
    clickToExpand.SetPosition(845,570);
    clickToExpand.SetColor(sf::Color::Black);
    sf::String data[40];


    if (!Log.LoadFromFile("artwork/log.png")){
        std::cerr<<"Error loading background image"<<std::endl;
    }
    spriteLog.SetImage(Log);



    if(!choice2){
        identifier.SetText("Categories");
        for(int i = 0; i < 10; i++){
            o->Draw(categories[i]);
        }

        //averageCategorySize strings
        int y = 40;
        for(int i = 0; i < 10 ;i ++){
            std::stringstream ss;
            ss << "Average category size = " << averageCategorySize[i];
            data[i].SetText(ss.str());
            data[i].SetColor(sf::Color::Black);
            data[i].SetPosition(860,y);
            data[i].SetSize(11);
            y += 53;
            o->Draw(data[i]);
        }

        //averageComsuptionWeek strings
        y = 50;
        for(int i = 10; i < 20 ;i ++){
            std::stringstream ss;
            ss << "Average consumption/week = " << averageConsumptionWeek[i];
            data[i].SetText(ss.str());
            data[i].SetColor(sf::Color::Black);
            data[i].SetPosition(860,y);
            data[i].SetSize(11);
            y += 53;
            o->Draw(data[i]);
        }

        //averageDeathRate strings
        y = 60;
        for(int i = 20; i < 30 ;i ++){
            std::stringstream ss;
            ss << "Average death rate = " << averageDeathRate[i];
            data[i].SetText(ss.str());
            data[i].SetColor(sf::Color::Black);
            data[i].SetPosition(860,y);
            data[i].SetSize(11);
            y += 53;
            o->Draw(data[i]);
        }

        //averageAge strings
        y = 70;
        for(int i = 30; i < 40 ;i ++){
            std::stringstream ss;
            ss << "Average age = " << averageAge[i];
            data[i].SetText(ss.str());
            data[i].SetColor(sf::Color::Black);
            data[i].SetPosition(860,y);
            data[i].SetSize(11);
            y += 53;
            o->Draw(data[i]);
        }


    }else{

    }

    if(choice){
        o->Draw(spriteLog);
        //TO ADD LOG INFORMATION


        clickToExpand.SetText("Press click to close log");
    }else{
        clickToExpand.SetText("Press click to show log");
    }


    o->Draw(identifier);
    o->Draw(clickToExpand);
    o->Draw(StatsTitle);
}
