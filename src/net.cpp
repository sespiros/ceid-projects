#include "net.h"
#include "helper.h"

sf::RenderWindow* Net::window = 0;
bool Net::isPaused_;

Net::Net(int r, int x, int y, int ls) : maxRadius(r)
{
	radius = 0;
    count = 0;
	Net::x = x;
	Net::y = y;
    lifespan = ls;
	roundsRun = 0;

	if (!img.LoadFromFile("artwork/Net.png"))
		std::cerr << "Could not load image (Net.png)" << std::endl;
	sprite.SetImage(img);
	sprite.SetScale(0.25f, 0.25f);
	sprite.SetPosition(Helper::worldToPixel[x][y].x, Helper::worldToPixel[x][y].y);
}

void Net::bind(sf::RenderWindow* w)
{
    window = w;
}

void Net::tick()
{
	for (int yy = -radius; yy <= radius; yy++) {
		for (int xx = abs(yy) - radius; xx <= radius - abs(yy); xx++) {
			if (!Ocean::isValid(y + yy, Ocean::MAX_Y) || !Ocean::isValid(x + xx, Ocean::MAX_X))
				continue;
			if (xx * xx + yy * yy > radius * radius)
				continue;
            if (Ocean::fishMap.count((x + xx) + (y + yy)*Ocean::MAX_X) != 0){
                Ocean::kill((x + xx) + (y + yy)*Ocean::MAX_X);
                count++;
            }
		}
	}

	roundsRun++;
	// radius = -abs(2.0 * maxRadius / lifespan * roundsRun - maxRadius) + maxRadius;
	radius = (roundsRun >= lifespan / 2) ? maxRadius : static_cast<int>(2.0f * maxRadius / lifespan * (roundsRun + 1));
}

void Net::draw()
{
	for (int yy = -radius; yy <= radius; yy++) {
		for (int xx = -radius; xx <= radius; xx++) {
			if (!Ocean::isValid(y + yy, Ocean::MAX_Y) || !Ocean::isValid(x + xx, Ocean::MAX_X))
				continue;
			if (xx * xx + yy * yy > radius * radius)
				continue;
            sprite.SetPosition(Helper::worldToPixel[x + xx][y + yy].x, Helper::worldToPixel[x + xx][y + yy].y);
			window->Draw(sprite);
		}
	}
}

bool Net::isDone(Net* p)
{
	bool ret = false;
	if (p->roundsRun >= p->lifespan) {
		delete p;
		p = 0;
		ret = true;
	}
	return ret;
}

int Net::getRadius() const
{
	return radius;
}

int Net::getX() const
{
	return x;
}

int Net::getY() const
{
	return y;
}

std::string Net::getCount() const
{
    std::stringstream ss;
    ss << count << " fish were caught.";
    return ss.str();
}
