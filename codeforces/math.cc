#include<iostream>
int main(){
    int numOrderedTriples = 0;
    for (int x = 1; x <= 7; x++)
    for (int y = 1; y <= 7; y++)
    for (int z = 1; z <= 7; z++)
    if (x+y+z <= 7) {numOrderedTriples++; std::cout<<x; std::cout<<y;std::cout<<z; std::cout<<"\n";}
//<<std::endl;}
    std::cout << numOrderedTriples << std::endl;
}
