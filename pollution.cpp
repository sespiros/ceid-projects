#include "pollution.h"
#include "helper.h"

sf::RenderWindow* Pollution::window = 0;

bool isDone(Pollution* p)
{
	bool ret = false;
	if (p->roundsRun >= p->lifespan) {
		delete p;
		p = 0;
		ret = true;
	}
	return ret;
}

Pollution::Pollution(int r, int x, int y, int ls)
{
	radius = r;
	Pollution::x = x;
	Pollution::y = y;
	lifespan = ls;
	roundsRun = 0;

	if (!img.LoadFromFile("artwork/Pollution.png"))
		std::cerr << "Could not load image (pollution.png)" << std::endl;
	sprite.SetImage(img);
	sprite.SetScale(0.25f, 0.25f);
	sprite.SetPosition(Helper::worldToPixel[x][y].x, Helper::worldToPixel[x][y].y);
}

void Pollution::bind(sf::RenderWindow* w)
{
	window = w;
}

void Pollution::tick()
{
	roundsRun++;
}

void Pollution::draw()
{
	for (int yy = -radius; yy <= radius; yy++) {
		for (int xx = abs(yy) - radius; xx <= radius - abs(yy); xx++) {
			if (!Ocean::isValid(y + yy, Ocean::MAX_Y) || !Ocean::isValid(x + xx, Ocean::MAX_X))
				continue;
			sprite.SetPosition(Helper::worldToPixel[x + xx][y + yy].x, Helper::worldToPixel[x + xx][y + yy].y);
			window->Draw(sprite);
		}
	}
}

int Pollution::getRadius() const
{
	return radius;
}

int Pollution::getX() const
{
	return x;
}

int Pollution::getY() const
{
	return y;
}
