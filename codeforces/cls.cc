#include<iostream>
#include<algorithm>
#include<vector>
using namespace std;
int n;
const int N =1002;
vector<int> hobby[N];
int p[N],num[N];
vector<int> res;
/*
从样例解释中可以看到 爱好具有传递性
即：3 和 5 有共同的爱好 3，3 和 7 有共同的爱好 5，那么357也是一个集群
*/
void init()
{
    for(int i=1;i<=n;i++) p[i]=i;
}

int find(int x)
{
    if(p[x]!=x) p[x]=find(p[x]);
    return p[x];
}
bool cmp(int &a,int &b)
{
    return a>=b;
}
int main()
{
    cin>>n;

    for(int i=1;i<=n;i++)//第i人的爱好
    {
        int k;
        scanf("%d",&k);
        getchar();
        for(int j=0;j<k;j++)
        {
            int x;
            scanf("%d",&x);
            hobby[x].push_back(i);
        }
    }

    init();

    for(int i=1;i<N;i++)
    {
        for(int j=1;j<hobby[i].size();j++)
        {
            int a=hobby[i][0],b=hobby[i][j];
            int pa=find(a),pb=find(b);
            if(pa!=pb)
            {
                p[pb]=pa;
            }
        }
    }


    //cout<<n<<endl;
    for(int i=1;i<=n;i++)
    {
        cout<<i<<"  ";
        cout<<find(i)<<endl;
        num[find(i)]++;
    }
    for(int i=1;i<=n;i++)
    {
        cout<<num[i]<<endl;
    }
    sort(num+1,num+n,cmp);

    //统计个数
    int cnt=0;
    for(int i=0;i<=n;i++) 
    {
        if(num[i]) cnt++;
        else break;
    }
    cout<<cnt<<endl;
    for(int i=0;i<cnt;i++) cout<<num[i]<<" ";
    return 0;
}
