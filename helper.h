#ifndef HELPER_H
#define HELPER_H

#include <SFML/Graphics.hpp>
#include "classregistry.h"
#include "Ocean.h"
#include <iostream>

class Helper {
public:
        static sf::Vector2f** getWorldScreenMapping();
		static void cleanup();
		static void swapDir(int, int);

        static sf::Vector2f** worldToPixel;
		static int dir[8][2];
};

#endif
