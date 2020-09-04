#include<iostream>
using namespace std;
typedef long long LL;

LL KSM(LL a,LL b,LL m){
    if (b==0)//边界处理
        return 1;
    if(b&1){
        return a*KSM(a,b-1,m)%m;   
    }else{
        LL mid=KSM(a,b/2,m);
        return mid*mid%m;
    }
}

int main(){
    LL a,b,m;
    while(cin>>a>>b>>m){
        a%=m;
        cout<<KSM(a,b,m)<<endl;
    }


    return 0;
}
