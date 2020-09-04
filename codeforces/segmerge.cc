//segments merge

//given n segments [l,r], merge those intersect, output final number of segments

#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

#define minn -2e9
typedef pair<int, int> PII;

void merge(vector<PII> &segs)
{
    vector<PII> res;

    sort(segs.begin(), segs.end());

    int st = minn, ed = minn;
    for (auto seg : segs)
        if (ed < seg.first) // no intersection
        {
            if (st != minn) res.push_back({st, ed});
            st = seg.first, ed = seg.second;
        }
        else ed = max(ed, seg.second); //has intersection

    if (st != minn) res.push_back({st, ed});

    segs = res;
}

int main()
{
    int n;
    scanf("%d", &n);

    vector<PII> segs;
    for (int i = 0; i < n; i ++ )
    {
        int l, r;
        scanf("%d%d", &l, &r);
        segs.push_back({l, r});
    }

    merge(segs);

    cout << segs.size() << endl;

    return 0;
}
