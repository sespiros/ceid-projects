#include "helper.h"
#include "screenmanager.h"

int main(int argc, char** argv)
{
	// Ocean Initialization
	Ocean::init();

	// Screen Initialization
	ScreenManager::init();

	// Main Loop
	ScreenManager::run();

    return EXIT_SUCCESS;
}
