#include "helper.h"
#include <iomanip>

using namespace std;
using namespace sf;

int Helper::dir[8][2] = {
    {-1, -1},
    {-1, 0},
    {-1, 1},
    {0, -1},
    {0, 1},
    {1, -1},
    {1, 0},
    {1, 1},
};

Vector2f** Helper::getWorldScreenMapping() {

    /*
         x=1024 jtiles pointer: i

         y=600      |--|--|--|--|--|--|--|--|--|--|
                    |--|--|--|--|--|--|--|--|--|--|
         jtiles     |--|--|--|--|--|--|--|--|--|--|
                    |--|--|--|--|--|--|--|--|--|--|
         pointer: j |--|--|--|--|--|--|--|--|--|--|
                    |--|--|--|--|--|--|--|--|--|--|
                    |--|--|--|--|--|--|--|--|--|--|

         --     =    Vector2f
         jtiles =    1024/(size of tile+5)
         itiles =    600/(size of tile+5)
         +5 to add a gap between fish
         */
    int w = 32;
    int jtiles = Ocean::MAX_Y;
    int itiles = Ocean::MAX_X;

    int i,j;
    int xs=0,ys=0;
    Vector2f **conv;

    //buffer allocation
    conv = new Vector2f*[itiles];
    for (int i=0;i<itiles;++i)
        conv[i] = new Vector2f[jtiles];

    //Fill buffer with window coordinates
    for(i=0;i<itiles;i++){
		ys=0.f;
        for(j=0;j<jtiles;j++){
            conv[i][j].x=xs;
            conv[i][j].y=ys;
			ys+=w+5;
        }
		xs+=w+5;
    }

//    Print buffer for debugging purposes(DO NOT DELETE!)
//    for(i=0;i<itiles;i++){
//        cout<<"|";
//        for(j=0;j<jtiles;j++){
//            cout<<setw(3);
//            cout<<conv[i][j].x<<" "<<conv[i][j].y<<"|";
//        }
//        cout<<endl;
//    }

    return conv;
}

void Helper::cleanup() {
	for (int i = 0; i < Ocean::MAX_X; i++) {
		delete[] Helper::worldToPixel[i];
	}

	delete[] Helper::worldToPixel;
}

void Helper::swapDir(int i, int j) {
    int tmp = Helper::dir[i][0];
    Helper::dir[i][0] = Helper::dir[j][0];
    Helper::dir[j][0] = tmp;

    tmp = Helper::dir[i][1];
    Helper::dir[i][1] = Helper::dir[j][1];
    Helper::dir[j][1] = tmp;
}
