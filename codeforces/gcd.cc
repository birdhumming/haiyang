#include <iostream>
using namespace std;

int gcd(int a, int b){

    return b ? gcd(b, a%b) : a; 
    //return a ? gcd(b%a, a) : b;

}
// Bezout identity ax+by=d

int exgcd(int a, int b, int &x, int &y) {
    if(!b){
        x=1;y=0;return a;
    }
    int d = exgcd(b, a%b, y, x);
    y-=(a/b)*x;
    return d;
}

int main(){

    int a,b;
    while (cin>>a>>b)
    {
        //cout<<gcd(a,b)<<endl;
        int x,y;
        int d = exgcd(a,b,x,y);
        printf("%d * %d + %d * %d = %d\n", a, x, b, y, d);
    }
    return 0;
}