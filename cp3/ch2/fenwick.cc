#include <bits/stdc++.h>
#define LSOne(S) (S & (-S))
using namespace std;
typedef long long int L;

int rsq(L b, L x[]){
    int sum=0;
    for(; b; b-=LSOne(b)) sum+=x[b];
    return sum;
}

void adjust(int k, L v, L x[], int n){
    for(; k<=n;k+=LSOne(k)) x[k]+=v;
}

int main(){
    int n,q; cin>>n>>q;
    L ft[n+1]; memset(ft,0,sizeof(ft));
    while(q--){
        char x; int i,j; cin>>x;
        if(x=='+'){
            cin>>i>>j;
            adjust(i+1,j,ft,n);
        } else if(x=='?') {
            cin>>i;
            cout<<rsq(i,ft)<<endl;
        }
    }
}
