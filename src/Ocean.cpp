#include "Ocean.h"
#include "classregistry.h"
#include "helper.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <algorithm>

using  std::map;
using  std::pair;
/*
 * static declarations
 */
int Ocean::count = 0;
int Ocean::deaths = 0;
int Ocean::turns = 0;
int Ocean::births = 0;
bool Ocean::choice = false;
int Ocean::choiceHash = 0;
std::stringstream Ocean::log;
std::map<int, Organism*> Ocean::fishMap;
std::vector<Pollution*> Ocean::pollution;
float Ocean::averageCategorySize[10];
float Ocean::averageConsumptionWeek[10];
float Ocean::averageDeathRate[10];
float Ocean::averageAge[10];
sf::String Ocean::categories[10];
sf::Sprite Ocean::Images[10];
sf::Font Ocean::GlobalFont;
bool Ocean::worldIsBig;
bool Ocean::easterEggs = true;
int Ocean::MAX_COUNT;

void Ocean::setup() {
    ClassRegistry::registerClasses();

    GlobalFont.LoadFromFile("artwork/font.ttf");

    //this variable is for setting y coordinate for the strings of stats
    int y = 15;//30 HAHAHHHAHA I PIO AKYRH METAVLITI POU MPOREI NA YPHRXE

    //Stats Arrays initialization
    for(int i = 0; i < 10 ; i++){
        averageConsumptionWeek[i] = 0;
        averageDeathRate[i] = 0;

        //Stats String Categories Arrays initialization
        categories[i].SetText(ClassRegistry::assocMapNames[i]);
        categories[i].SetPosition(830,y);
        categories[i].SetColor(sf::Color::White);
        categories[i].SetFont(GlobalFont);
        categories[i].SetSize(11);

        Images[i].SetImage(ClassRegistry::assocMapImages[i]);
        Images[i].SetPosition(830,y+19);
        Images[i].SetScale(0.20f, 0.20f);

        y += 53;
    }
}

void Ocean::init(bool choice) {
    // first, clear previous ocean contents
    Ocean::fishMap.clear();
    if (Helper::worldToPixel != NULL)
        Helper::cleanup();

    Ocean::worldIsBig = choice;

    if(worldIsBig){
        MAX_COUNT = 200;
        MAX_X = 49;
        MAX_Y = 36;
    }else{
        MAX_COUNT = 100;
        MAX_X = 22;
        MAX_Y = 16;
    }

    Helper::worldToPixel = Helper::getWorldScreenMapping();

    Ocean::populate();
}

void Ocean::add(Organism *toAdd) {
    int hash = toAdd->getX() + toAdd->getY() * MAX_X;
    fishMap.insert(pair<int, Organism*>(hash, toAdd));
    count++;
}

void Ocean::kill(int key) {

    if(key == Ocean::choiceHash && Ocean::choice){
        Ocean::choiceHash = -1;
        Ocean::choice = false;
    }

    fishMap[key]->kill();
    delete fishMap[key];
    fishMap.erase(key);
    count--;
    deaths++;

    regLog("A fish has died..");
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

    if(key == Ocean::choiceHash && Ocean::choice){
        Ocean::choiceHash = hash;
    }

    //not Ocean::kill because in Ocean::we move an object and we don't want count--
    fishMap.erase(it++);
    return it;
}

mapIter Ocean::collide(int key){
    mapIter next = Ocean::fishMap.find(key);
    int curX = Ocean::fishMap[key]->getX();
    int curY = Ocean::fishMap[key]->getY();

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
        std::stringstream ss;
        ss << "Fish " << key << " ate fish " << hash;
        regLog(ss.str());
    }else if(hasBred){
        next = Ocean::breed(key, hash);
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
    std::for_each(Ocean::pollution.begin(), Ocean::pollution.end(), std::mem_fun(&Pollution::tick));
    Ocean::pollution.erase(std::remove_if(Ocean::pollution.begin(), Ocean::pollution.end(), Pollution::isDone), Ocean::pollution.end());
}

void Ocean::update() {
    srand(time(0));
    mapIter it;

    tickPollution();
    turns++;
    for (it = Ocean::fishMap.begin(); it != Ocean::fishMap.end();) {
        if (it->second->isDone)
            ++it;
        else{
            it->second->isDone = true;
            it = collide(it->first);
        }
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

void Ocean::pollute(int r, int x, int y, int t)
{
    Pollution* p = new Pollution(r, x, y, t);
    Ocean::pollution.insert(Ocean::pollution.begin(), p);
    //debug
    //std::cout << "Inserted pollution source at (" << x << ", " << y << "), radius " << r << std::endl;
}

void Ocean::stats(){

    //Average size of category
    //Average consumption last week
    //AverageDeathRate

    memset(averageCategorySize, 0, sizeof(averageCategorySize));
    memset(averageAge, 0, sizeof(averageAge));
    if(turns%7 == 0)
        memset(averageConsumptionWeek, 0, sizeof(averageConsumptionWeek));

    int toBeKilled[200], c = 0;

    mapIter it;
    for(it = fishMap.begin(); it != fishMap.end();it++){
        averageCategorySize [it->second->getType()] += (it->second->getCount() != 0) ? it->second->getSize()/static_cast<float>(it->second->getCount()) : 0;
        averageDeathRate [it->second->getType()] = (Ocean::deaths != 0) ? 100.0f*it->second->getDeaths()/static_cast<float>(Ocean::deaths) : 0;
        averageAge [it->second->getType()] += (it->second->getCount() != 0) ? it->second->getAge()/static_cast<float>(it->second->getCount()) : 0;
        if(it->second->levelUpCheck()){
            toBeKilled[c++] = it->first;
        }

        if(turns%7 == 0){
            averageConsumptionWeek [it->second->getType()] += (it->second->getCount() != 0) ? it->second->getFoodConsumedWeek()/static_cast<float>(it->second->getCount()) : 0;
            it->second->weeklyReset();
        }
    }

    for(int i = 0; i < c; i++){
        Ocean::kill(toBeKilled[i]);
    }

}

void Ocean::drawStats(sf::RenderWindow *o, bool choice, bool choice2){
    int y;
    sf::String StatsTitle;
    sf::String clickToExpand;
    sf::String identifier;
    sf::String logs;
    sf::String general;
    sf::Image Log;
    sf::Sprite spriteLog;
    StatsTitle.SetText("Stats:");
    StatsTitle.SetSize(30);
    StatsTitle.SetPosition(830,0);
    StatsTitle.SetColor(sf::Color::White);
    StatsTitle.SetFont(GlobalFont);
    clickToExpand.SetSize(15);
    clickToExpand.SetPosition(837,570);
    clickToExpand.SetColor(sf::Color::White);
    clickToExpand.SetFont(GlobalFont);
    identifier.SetSize(20);
    identifier.SetPosition(915,8);
    identifier.SetColor(sf::Color::White);
    identifier.SetFont(GlobalFont);
    logs.SetText(Ocean::log.str());
    logs.SetSize(11);
    logs.SetPosition(835,330);
    logs.SetColor(sf::Color::Black);

    if (!Log.LoadFromFile("artwork/log.png")){
        std::cerr<<"Error loading background image"<<std::endl;
    }
    spriteLog.SetImage(Log);

    //sf::String data[40];
    sf::String data[10];

    std::stringstream ss;
    ss <<"Runtime "     <<  std::setw(6)    <<Ocean::turns         <<"  | ";
    ss <<"Week "        <<  std::setw(6)    <<Ocean::turns / 7 + 1 <<"  | ";
    ss <<"Total alive " <<  std::setw(6)    <<Ocean::count         <<"  | ";
    ss <<"Deaths "      <<  std::setw(6)    <<Ocean::deaths        <<"  | ";
    ss <<"Births "      <<  std::setw(5)    <<Ocean::births        <<" | "<<std::endl;

    general.SetText(ss.str());
    general.SetSize(14);
    general.SetPosition(8,576);
    general.SetColor(sf::Color::White);
    general.SetFont(GlobalFont);

    if(!choice2){

        identifier.SetText("Categories");

        if(Ocean::choiceHash == -1){
            regLog("Chosen fish died");
            choiceHash = 0;
        }

        for(int i = 0; i < 10; i++){
            //categories[i].SetSize(11);
            //o->Draw(categories[i]);
            Images[i].SetScale(0.35,0.35);
            o->Draw(Images[i]);
        }

        int y = 45;
        for(int i = 0; i < 10; i++){
            std::stringstream ss;
            ss << std::setprecision(1) << std::setw(2) << std::setiosflags(std::ios::fixed)<<averageCategorySize[i]    << "  ";
            ss << averageConsumptionWeek[i] << "  ";
            ss << averageDeathRate[i]        << "  ";
            ss << averageAge[i]              << "  ";

            data[i].SetText(ss.str());
            data[i].SetColor(sf::Color::White);
            data[i].SetPosition(875,y);
            data[i].SetSize(15);
            data[i].SetFont(GlobalFont);
            y += 53;
            o->Draw(data[i]);

        }

    }else{
        std::stringstream sb;
        sb<<ClassRegistry::assocMapNames[Ocean::fishMap[Ocean::choiceHash]->getType()];

        identifier.SetText(sb.str());
        sf::Sprite id = Ocean::fishMap[Ocean::choiceHash]->sprite;

        id.SetPosition(830,30);
        id.SetScale(1.5f,1.5f);

        std::stringstream sa;
        sa << " Position: "<< Ocean::fishMap[Ocean::choiceHash]->getX()<<", "<<Ocean::fishMap[Ocean::choiceHash]->getY()<<std::endl;
        y = 220;
        data[0].SetText(sa.str());
        data[0].SetColor(sf::Color::White);
        data[0].SetPosition(830,y);
        data[0].SetSize(12);
        data[0].SetFont(GlobalFont);
        std::stringstream s0;
        s0 << " age = "<< Ocean::fishMap[Ocean::choiceHash]->getAge();
        y += 15;
        data[1].SetText(s0.str());
        data[1].SetColor(sf::Color::White);
        data[1].SetPosition(830,y);
        data[1].SetSize(12);
        data[1].SetFont(GlobalFont);
        std::stringstream s1;
        s1 << " size = "<< Ocean::fishMap[Ocean::choiceHash]->getSize();
        y += 15;
        data[2].SetText(s1.str());
        data[2].SetColor(sf::Color::White);
        data[2].SetPosition(830,y);
        data[2].SetSize(12);
        data[2].SetFont(GlobalFont);
        std::stringstream s2;
        s2 << " growthRate = "<< Ocean::fishMap[Ocean::choiceHash]->getGrowthRate();
        y += 15;
        data[3].SetText(s2.str());
        data[3].SetColor(sf::Color::White);
        data[3].SetPosition(830,y);
        data[3].SetSize(12);
        data[3].SetFont(GlobalFont);
        std::stringstream s3;
        s3 << " food Required/Week = "<< Ocean::fishMap[Ocean::choiceHash]->getFoodRequiredPerWeek();
        y += 15;
        data[4].SetText(s3.str());
        data[4].SetColor(sf::Color::White);
        data[4].SetPosition(830,y);
        data[4].SetSize(12);
        data[4].SetFont(GlobalFont);
        std::stringstream s4;
        s4 << " food Consumed/Week = "<< Ocean::fishMap[Ocean::choiceHash]->getFoodConsumedWeek();
        y += 15;
        data[5].SetText(s4.str());
        data[5].SetColor(sf::Color::White);
        data[5].SetPosition(830,y);
        data[5].SetSize(12);
        data[5].SetFont(GlobalFont);

        o->Draw(id);
        for(int i = 0; i < 6; i++)
            o->Draw(data[i]);
    }

    if(choice){
        o->Draw(spriteLog);


        o->Draw(logs);
        clickToExpand.SetText("Press click to close log");
        clickToExpand.SetColor(sf::Color::Black);
    }else{
        clickToExpand.SetText("Press click to show log");
    }



    o->Draw(general);
    o->Draw(identifier);
    o->Draw(clickToExpand);
    o->Draw(StatsTitle);
}

void Ocean::regLog(std::string subj){
    static int counter;

    if(counter%21 == 0){
        Ocean::log.str("");
    }

    Ocean::log << subj <<std::endl;
    std::cout << subj <<std::endl; //temporary
    counter++;

}

mapIter Ocean::breed(int key1, int key2){
    bool done = false;
    int parents[2];
    parents[0] = key1;
    parents[1] = key2;
    int k = 0;
    Organism::fishtype type = Ocean::fishMap[key1]->getType();
    std::map<Organism::fishtype, int>::const_iterator iter;
    iter = Organism::weightMap.find(type);
    // int weight = iter->second;

    //as China because we can
    int breedLimit;
    if(Ocean::fishMap[key1]->getCount()>15){
        breedLimit = Ocean::fishMap[key1]->getCount();
    }else{
        breedLimit = 6;
    }

    mapIter it;
    it = Ocean::fishMap.find(key1);

    while(k < 2 && !done){
        int random = rand()%breedLimit;
        for (int i = 7; i > 0 && !done; i--) {
            int j = rand()%(i + 1);

            int dx = Helper::dir[j][0];
            int dy = Helper::dir[j][1];

            Helper::swapDir(j, i);

            if (!isValid(Ocean::fishMap[parents[k]]->getX() + dx, Ocean::MAX_X) || !isValid(Ocean::fishMap[parents[k]]->getY() + dy, Ocean::MAX_Y))
                continue;

            int x = Ocean::fishMap[parents[k]]->getX() + dx;
            int y = Ocean::fishMap[parents[k]]->getY() + dy;

            int hash = x + y * Ocean::MAX_X;

            if(Ocean::fishMap.find(hash) == Ocean::fishMap.end() && random == 1){
                Ocean::createAndAddFish(type, x, y);
                std::stringstream ss;
                ss << "A "<<ClassRegistry::assocMapNames[type]<<" was born";
                regLog(ss.str());
                Ocean::births++;
                done = true;
            }
        }
        k++;
    }

    return it;
}
