#ifndef HELPER_H
#define HELPER_H

#include <SFML/Graphics.hpp>
#include "classregistry.h"
#include "Ocean.h"
#include <iostream>

class Helper {
public:
		static sf::Vector2f** getWorldScreenMapping(const int x, const int y, const int w, const int h);

		static sf::Vector2f** worldToPixel;
};

#endif
