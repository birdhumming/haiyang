#include <iostream>
#include <unordered_map>
#include <vector>
using namespace std;
typedef pair<int, int> ii;

int N, a, b;
unordered_map<int, vector<int> > x; //y[i] is x coordinates of points with y coord i
unordered_map<int, vector<int> > y; //vv
unordered_map<ii, char> m; //mirror location, orientation
vector<ii> ml; //mirror locations in order

int main()
{
    cin>>N>>a>>b;
    x[a]=b; y[b]=a;
    for (int i=0; i<N; i++) {
        int w, z; char c;
        cin>>w>>z>>c;
        x[w].push_back(z);
        y[z].push_back(w);
        ii temp; temp.first=w, temp.second=z;
        m[temp]=c;
        ml.push_back(temp);
    }
    long sum=0;
    int upp=min((2*K)+1, max_dist);
    for (int i=1; i<=upp; i++) {
        if (u.count(i)==1) {
            sum+=u[i];
        }
    }
    if (upp==max_dist) {
        cout<<sum<<'\n';
        return 0;
    }
    long ans=sum;
    for (int i=2; i<=max_dist; i++) {
        if (u.count(i-1)==1) {
            sum-=u[i-1];
        }
        if ((2*K)+i <= max_dist) {
            if (u.count((2*K)+i)==1) {
                sum+=u[(2*K)+i];
            }
        }
        //cout<<sum<<'\n';
        if (sum>ans) ans=sum;
    }
    cout<<ans<<'\n';
    
}
