#include "organism.h"
#include <iostream>

using namespace std;

int main() {
    ZPlankton g(2, 2);
    ZPlankton c(5, 5);
    Eel fuck(0, 0);

    cout<< g.getX()<<endl;

    g.setX(7);

    cout<< g.getX()<<endl;
    return 0;
}
