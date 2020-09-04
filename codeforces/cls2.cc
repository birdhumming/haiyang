
#include <algorithm>
#include <cstdio>
#include <iostream>
 
using namespace std;
 
const int maxn=1010;
 
int father[maxn];
int isRoot[maxn]={0};
int hobby[maxn]={0}; 
 
int n;
 
void init(){
    for(int i=1;i<=n;i++){
        father[i]=i;
        isRoot[i]=0;
    }
}
 
int findFather(int x){
    int a=x;
    while(x!=father[x]){
        x=father[x];
    }
 
    //路径压缩
    while(a!=father[a]){
        int z=father[a];
        father[a]=x;
        a=z;
    }
    return x;
}
 
void Union(int a,int b){
    int aF=findFather(a);
    int bF=findFather(b);
    if(aF!=bF){
        father[aF]=bF;
    }
}
 
bool cmp(int a,int b){
    return a>b;
}
 
int main(){
    cin >> n;
    init();
    int num,h;
 
 
    for(int i=1;i<=n;i++){
        scanf("%d:",&num);
        for(int j=0;j<num;j++){
            scanf("%d",&h);
            if(hobby[h]==0){  //h还没人喜欢过
                hobby[h]=i;
            }
            Union(i,hobby[h]);  //把新人i和同样喜欢h的人合并一组
        }
    }
 
    //记录每组人数
    for(int i=1;i<=n;i++){
        isRoot[findFather(i)]++;
    }
 
    //人数不为0即有一组
    int ans=0;
    for(int i=1;i<=n;i++){
        if(isRoot[i]!=0)
            ans++;
    }
 
    //人数由大到小输出
    sort(isRoot+1,isRoot+1+n,cmp);
    cout << ans << endl;
    for(int i=1;i<=ans;i++){
        printf("%d",isRoot[i]);
        if(i!=ans)
            printf(" ");
    }

}
