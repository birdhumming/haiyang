#include <iostream>
using namespace std;




//1. factorization 1. number of factors for a given number;
//FTA https://brilliant.org/wiki/fundamental-theorem-of-arithmetic/
// 2. Euler's totient/phi function 
// phi(n) = n * (1-1/p1)*(1-1/p2)*...*(1-1/pm)
// 

int gcd(int a, int b){

    return b ? gcd(b, a%b) : a; 
    //return a ? gcd(b%a, a) : b;

}

int main(){
    int n;
    while(cin>>n){
	int nn=n; //keep a copy of n;
        cout<<n<<" = ";
        int sumd =1, phi=n;
        bool isfirst=true; //check if this is the first factor number;
        for (int i=2; i<=n; i++){
            int a=0;
            while (n%i==0){
                n /= i;
                a++;
            }
            if(a){
                sumd*=(a+1);
                phi=phi - phi/i;
                if(!isfirst) cout<< '*'; //only output * if not the first factor
                isfirst=false;
                cout<<i<<"^"<<a;

            }
        }
        cout<<endl;
        cout<<"Number of factors: "<<sumd<<endl;

	n=nn;
	for(int i=1;i<=n;i++){
	if(n%i==0) cout<<i<<' ';
	}
	cout<<endl;
        cout<<"Euler phi(n) is: "<<phi<<endl;
	n=nn;
        for(int i =1;i<=n;i++){
            if(gcd(i,n)==1) cout<<i<<' ';}
        cout<<endl;

    }

return 0;

}
