#include <iostream>
#include <ctime>
#include <cstring>
using namespace std;

#define N 10000010

int primes[N],cnt;

// brutal force listing out primes; method 1
// complexity n*sqrt(n);

int p1(int n){
    for (int i=2; i<=n;i++){
        bool flag = false; //flag means not a prime number;
        for(int j=2; j*j<=i;j++){
            if(i%j==0){flag=true;break;}
        }
        if(!flag)primes[cnt++]=i;
    }
    for(int i=0;i<cnt;i++)cout<<primes[i]<<" ";
    cout<<endl;
}

//sieve simple method 2 for prime numbers
//complexity n*ln(n) b/c 1+1/2+1/3+...1/n = ln(n)+c0

bool sv[N];  
//sieve array to mark composite numbers for 2 multiples; 3 multiples; 5 multiples etc

int p2(int n){

    for (int i=2;i<=n;i++){
        if(sv[i]) continue;
        primes[cnt++]=i;
        for(int j=i+i;j<=n;j+=i)
            sv[j]=true;
    }
    //for(int i=0;i<cnt;i++)cout<<primes[i]<<" ";
    //cout<<endl;
}

//improved sieve; linear sieve method O(n);

void p3(int n){
    for(int i=2; i<=n;i++){
        
        if(!sv[i]) primes[cnt++] = i;

        for(int j=0; (j<cnt) && (i*primes[j]<=n);j++){
    
            sv[primes[j]*i]=true;  //only done once for one non-prime
            
            //debug line for understanding
            cout<<primes[j]*i<<" for i "<<i<<" and j "<<j<<endl;

            if(i%primes[j]==0) break;  //pruning 
        }
    
    }
    
    //output primes

    for(int i=0;i<cnt;i++)cout<<primes[i]<<" ";
    cout<<endl;

}

//get euler phi number in linear time, with linear sieve

int euler[N];
void p4(int n){
    euler[1]=1;

    for(int i=2; i<=n;i++){
        
        if(!sv[i]) {
            primes[cnt++] = i; // we get a prime number
            euler[i]=i-1; //for a prime number p, the phi(p)=p-1;
        }

        //for composite numbers
        for(int j=0; j<cnt&&i*primes[j]<=n;j++){
    
            sv[primes[j]*i]=true;  //identifying a non-prime; only done once for a non-prime
    
            if(i%primes[j]==0) {
                euler[i*primes[j]]=euler[i]*primes[j];
                break;
            }
            euler[i*primes[j]] = euler[i]*(primes[j]-1); 
            //3*2, 5*2 5*3, 7*2 7*3 7*5, 9*2 11*2 11*3 11*7 11*5
            
            //cout<<"case special doing euler for i "<<i<< " and j "<<j<<endl;
        }
    
    }
    
    //output euler numbers

    for(int i=1;i<=n;i++)cout<<"Euler[" <<i<<"] = " <<euler[i]<<endl;
    cout<<endl;

}





int main(){
    int n;
    while(cin>>n){
        
        //p1(n);

        int start= clock();
        //p2(n);
        //cout<<"naive sieve: "<<clock()-start<<endl;
        
        //start= clock();

        //p3(n);
       //cout<<"linear sieve: "<<clock()-start<<endl;
    
    //get euler phi numbers;
    p3(n);

    //p4(n);
    }

return 0;

}
