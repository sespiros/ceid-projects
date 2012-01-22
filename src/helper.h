#ifndef HELPER_H
#define HELPER_H

#include <SFML/Graphics.hpp>
#include "classregistry.h"
#include "Ocean.h"
#include <iostream>

class Helper {
public:
    //initializes worldToPixel
    static sf::Vector2f** getWorldScreenMapping();

    //deallocates worldToPixel
    static void cleanup();

    //swaps dir for implementing Fisher-Yates shuffle
    static void swapDir(int, int);

    //conversion from screen coordinates to local coordinates
    static sf::Vector2i getLocalCoords(float, float);

    //conversion from local coordinates to screen coordinates
    static sf::Vector2f** worldToPixel;

    //Array holding available moves
    static int dir[8][2];
};

#endif

/**
 * UGV0cm9zIE1wYW50b2xhcw0KU3Bpcm9zIFNlaW1lbmlzDQoyMDEy
 */
