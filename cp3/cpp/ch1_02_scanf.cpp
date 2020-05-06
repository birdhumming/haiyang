#include <cstdio>
using namespace std;

int N;         // using global variables in contests can be a good strategy
char x[110];  // make it a habit to set array size a bit larger than needed

int main() {
  scanf("%d\n", &N);
  while (N--) {                  // we simply loop from N, N-1, N-2, ..., 0
    scanf("0.%[0-9]...\n", &x);   // `&' is optional when x is a char array
                         // note: if you are surprised with the trick above,
                      // please check scanf details in www.cppreference.com
    printf("the digits are 0.%s\n", x);
} } // return 0;

/* 3
0.1227...
0.517611738...
0.7341231223444344389923899277...

store input in a file; need to run

./a.out <input_data
*/
