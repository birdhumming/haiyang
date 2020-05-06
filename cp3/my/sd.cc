#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

int N;
string stalls;

int main()
{
   ifstream infile ("socdist1.in");
   ofstream outfile ("socdist1.out");

   infile >> N >> stalls;

   vector <int> cow_index;
   vector <int> dist;

   for (int i = 0; i < N; i++)
   {
       if (stalls[i] == '1')
       {
           cow_index.push_back(i);
       }
   }
    
    cout << "point 1" <<endl;

   if (cow_index.size() == 0)
   {
       //outfile << N-1 << endl;
       cout << N-1 << endl;
   }
   else
   {
       cout << "point 2: " << cow_index.size() <<endl;
       //int distance = i[0] - 0;
       //dist.push_back(distance);
       for (int i = 0; i < cow_index.size() - 1; i++)
       {
           int distance = cow_index[i+1] - cow_index[i];
           cout << "distance is" << distance <<endl;
           dist.push_back(distance);
       }
       sort(dist.begin(), dist.end());
       int max_length_fm = dist[dist.size()-1];
       int smax_length_fm = dist[dist.size()-2];
       int beg_dist = cow_index[0] - 0;
       int end_dist = N-1 - cow_index[cow_index.size()-1];

       cout << "distance from beginning: " << beg_dist << endl;
       cout << "distance from ending: " << end_dist << endl;
       cout << "max length in middle: " << max_length_fm << endl;
       cout << "second max length in middle: " << smax_length_fm << endl;

       vector <int> ans;
       //both at beginning
       ans.push_back(beg_dist/2);
       //one at beginning, one at end
       ans.push_back(min(beg_dist, end_dist));
       //both at end
       ans.push_back(end_dist/2);
       //one at end, one in middle somewhere
       ans.push_back(min(end_dist, max_length_fm/2));
       //both in middle somewhere in same length
       ans.push_back(max_length_fm/3);
       //both in middle somewhere in different lengths
       ans.push_back(min(max_length_fm/2, smax_length_fm/2));
       //one in middle, one in end
       ans.push_back(min(end_dist, max_length_fm/2));

       for (int i = 0; i < ans.size(); i++)
       {
           cout << ans[i] << endl;
       }
       //sort(ans.begin(), ans.end());
       //cout << ans[ans.size()-1] << endl;


   }
}
