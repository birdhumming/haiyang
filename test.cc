#include<iostream>
#include<math.h>
using namespace std;
typedef long long ll;

int qmi(int m, int k, int p)
{
    int res = 1%p, t = m;
    while (k)
    {
        if (k&1) res = res * t % p;
        t = t * t % p;
        k >>= 1;
    }
    return res;
}

//get x^y in log(y) time by converting power to multiplication
ll qexp(ll x, ll y){
	ll ans=1;
	while(y){
		if(y&1) ans=ans*x;
		x*=x;
		y>>=1;
	}
	return ans;
}


//get x*y in log(y) time by converting multiplication to addition
ll qmulti(ll x, ll y){
        ll ans=0;
        while(y){
                if(y&1) ans=ans+x;
                x+=x;
                y>>=1;
        }
        return ans;
}

int main()
{
    long long x,y,p;
    cin>>x>>y>>p;
    long long res=1;
	
    res=qexp(x,y);
    //cout<<res<<endl;
    res=qmulti(x,y);
    //cout<<res<<endl;
    res=qmi(x,y,p);
    cout<<res<<endl;

    res=1;
	//get x^y in log N time by converting power to multiplication
    while(y)
    {
        if(y&1) res=res*x;
        x=x*x;
        y>>=1;
    }
    //cout<<res<<endl;
    return 0;
}
