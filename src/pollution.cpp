#include "pollution.h"
#include "helper.h"

sf::RenderWindow* Pollution::window = 0;
bool Pollution::isPaused_;

Pollution::Pollution(int r, int x, int y, int ls) : maxRadius(r)
{
	radius = 0;
	Pollution::x = x;
	Pollution::y = y;
	lifespan = ls;
	innerRadius = 0;
	roundsRun = 0;

    if (!img.LoadFromFile("artwork/pollution.png"))
		std::cerr << "Could not load image (pollution.png)" << std::endl;
	sprite.SetImage(img);
	sprite.SetScale(0.25f, 0.25f);
	sprite.SetPosition(Helper::worldToPixel[x][y].x, Helper::worldToPixel[x][y].y);
}

void Pollution::bind(sf::RenderWindow* w, bool isPaused)
{
	window = w;
	isPaused_ = isPaused;
}

void Pollution::tick()
{
	for (int yy = -radius; yy <= radius; yy++) {
		for (int xx = abs(yy) - radius; xx <= radius - abs(yy); xx++) {
			if (!Ocean::isValid(y + yy, Ocean::MAX_Y) || !Ocean::isValid(x + xx, Ocean::MAX_X))
				continue;
			if (xx * xx + yy * yy > radius * radius)
				continue;
			if (xx * xx + yy * yy < innerRadius * innerRadius)
				continue;
			if (Ocean::fishMap.count((x + xx) + (y + yy)*Ocean::MAX_X) != 0)
				Ocean::kill((x + xx) + (y + yy)*Ocean::MAX_X);
		}
	}

	roundsRun++;
	// radius = -abs(2.0 * maxRadius / lifespan * roundsRun - maxRadius) + maxRadius;
	radius = (roundsRun >= lifespan / 2) ? maxRadius : static_cast<int>(2.0f * maxRadius / lifespan * (roundsRun + 1));
	innerRadius += (roundsRun < lifespan / 2) ? 0 : 1;
}

void Pollution::draw()
{
	for (int yy = -radius; yy <= radius; yy++) {
		for (int xx = -radius; xx <= radius; xx++) {
			if (!Ocean::isValid(y + yy, Ocean::MAX_Y) || !Ocean::isValid(x + xx, Ocean::MAX_X))
				continue;
			if (xx * xx + yy * yy > radius * radius)
				continue;
			if (xx * xx + yy * yy < innerRadius * innerRadius)
				continue;
			sprite.SetPosition(Helper::worldToPixel[x + xx][y + yy].x, Helper::worldToPixel[x + xx][y + yy].y);
			if (!isPaused_) {
				sprite.FlipX((rand() % 2 == 0) ? true : false);
				sprite.FlipY((rand() % 2 == 0) ? true : false);
			}
			window->Draw(sprite);
		}
	}
}

bool Pollution::isDone(Pollution* p)
{
	bool ret = false;
	if (p->roundsRun >= p->lifespan) {
		delete p;
		p = 0;
		ret = true;
	}
	return ret;
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
