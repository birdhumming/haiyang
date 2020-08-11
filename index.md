


CS50 class, CP3 singapore class; check every year

sqlite3 to create db:
.mode csv
.import("filename.csv")

then sqlie3 test.db will allow running SQL queries
.schema
.import FILE TABLE
CRUD - create read update delete
BLOB INTEGER smallit integer bigint REAL real double precision TEXT char(n) varchar(n) text  NUMERIC boolean date datetime numeric(scale, precision), time, timestamp
A = [(1, 2, 3), (4, 5, 6)]                       # list A = 2 tuples of 3
[*zip(*A)]                                       # [(1, 4), (2, 5), (3, 6)]

print(math.factorial(225))
python has no overflow of integers!
https://cloud.google.com/speech-to-text/docs/languages
https://realpython.com/python-speech-recognition/
https://pypi.org/project/SpeechRecognition/
pdf https://dev.to/mustafaanaskh99/convert-any-pdf-file-into-an-audio-book-with-python-1gk4
python text to speech pdf
https://pythonprogramminglanguage.com/text-to-speech/
https://stackoverflow.com/questions/50985619/how-to-read-pdf-files-which-are-in-asian-languages-chinese-japanese-thai-etc


to install python speech recognition on mac:
brew install portaudio
pip install --global-option='build_ext' --global-option='-I/usr/local/include' --global-option='-L/usr/local/lib' pyaudio
pip install SpeechRecognition


harvard cs50 and stanford classes 讲透了；even MIT not clear; yale princeton not sure - not actionable
find . -name "*py" -exec cp {} /tmp/python/ \;
\; – Indicates it that the commands to be executed are now complete, and to carry out the command again on the next match.
https://www.geeksforgeeks.org/speech-recognition-in-python-using-google-speech-api/
https://www.programiz.com/python-programming/methods/built-in/eval
python eval()



usaco training portal - print outs - castle problem stuck; identify problem types - greedy, divide and conquer, dynamic programming, flood fill, graph bfs/dfs,  is it practical, what makes it hard? others' code is hard to understand; 
google it, youtube it!
 basic sense of logic is much more important.
https://www.youtube.com/channel/UCdBXs_F9YlBodIH7eK0BYfQ/videos

check corner cases; check indices/bounds; do it on paper; check efficiency/complexity(space and time analysis); isolate your problem
small matrix - bash probably works?
www.ideone.com
docs.python.org/3/tutorial
cplusplus.com
docs.oracle.com/javase/8/docs/api/
informatics textbooks
https://open.umn.edu/opentextbooks/subjects/computer-science-information-systems



segment tree takes O(n*logn) space or O(4n)?, range sum query/update takes O(logn) time

fenwick tree takes O(n) space, rsq/update takes O(lg n) time; 1) less space; 2) easier to code compared to segment tree
range sum query = rsq = prefix sum
https://www.hackerearth.com/practice/data-structures/advanced-data-structures/fenwick-binary-indexed-trees/tutorial/


The function std::swap() is a built-in function in the C++ Standard Template Library (STL) which swaps the value of two variables. Syntax: swap(a, b) Parameters: The function accepts two mandatory parameters a and b which are to be swapped. The parameters can be of any data type.

return memo[id][w] = max(value(id + 1, w), V[id] + value(id + 1, w - W[id]));

to learn recursion, you have to learn other things
geometry - points, lines, circles, polygon, triangles; shoelace,dot product; cross product
https://github.iu.edu/haiyang/haiyang-scratch/tree/master/cpbook-code/ch7

#include <bits/stdc++.h>
using namespace std;

#define INF 1e9
#define EPS 1e-9
#define PI acos(-1.0) // important constant; alternative #define PI (2.0 * acos(0.0))

double DEG_to_RAD(double d) { return d*PI / 180.0; }

double RAD_to_DEG(double r) { return r*180.0 / PI; }

// struct point_i { int x, y; };    // basic raw form, minimalist mode
struct point_i { int x, y;     // whenever possible, work with point_i
 point_i() { x = y = 0; }                      // default constructor
 point_i(int _x, int _y) : x(_x), y(_y) {} };         // user-defined

struct point { double x, y;   // only used if more precision is needed
 point() { x = y = 0.0; }                      // default constructor
 point(double _x, double _y) : x(_x), y(_y) {}        // user-defined
 bool operator < (point other) const { // override less than operator
   if (fabs(x-other.x) > EPS)                   // useful for sorting
     return x < other.x;          // first criteria , by x-coordinate
   return y < other.y; }          // second criteria, by y-coordinate
 // use EPS (1e-9) when testing equality of two floating points
 bool operator == (point other) const {
  return (fabs(x-other.x) < EPS && (fabs(y-other.y) < EPS)); } };

double dist(point p1, point p2) {                // Euclidean distance
                     // hypot(dx, dy) returns sqrt(dx * dx + dy * dy)
 return hypot(p1.x-p2.x, p1.y-p2.y); }               // return double

// rotate p by theta degrees CCW w.r.t origin (0, 0)
point rotate(point p, double theta) {
 double rad = DEG_to_RAD(theta);    // multiply theta with PI / 180.0
 return point(p.x * cos(rad) - p.y*sin(rad),
              p.x * sin(rad) + p.y*cos(rad)); }

struct line { double a, b, c; };          // a way to represent a line

// the answer is stored in the third parameter (pass by reference)
void pointsToLine(point p1, point p2, line &l) {
 if (fabs(p1.x-p2.x) < EPS)                  // vertical line is fine
   l = {1.0, 0.0, -p1.x};                           // default values
 else {
   double a = -(double)(p1.y-p2.y) / (p1.x-p2.x);
   l = {a,
        1.0,              // IMPORTANT: we fix the value of b to 1.0
        -(double)(a*p1.x) - p1.y}; }
 }
// not needed since we will use the more robust form: ax + by + c = 0
struct line2 { double m, c; };      // another way to represent a line

int pointsToLine2(point p1, point p2, line2 &l) {
if (abs(p1.x-p2.x) < EPS) {            // special case: vertical line
  l.m = INF;                    // l contains m = INF and c = x_value
  l.c = p1.x;                  // to denote vertical line x = x_value
  return 0;   // we need this return variable to differentiate result
}
else {
  l.m = (double)(p1.y-p2.y) / (p1.x-p2.x);
  l.c = p1.y - l.m*p1.x;
  return 1;     // l contains m and c of the line equation y = mx + c
} }

bool areParallel(line l1, line l2) {       // check coefficients a & b
 return (fabs(l1.a-l2.a) < EPS) && (fabs(l1.b-l2.b) < EPS); }

bool areSame(line l1, line l2) {           // also check coefficient c
 return areParallel(l1 ,l2) && (fabs(l1.c-l2.c) < EPS); }

// returns true (+ intersection point) if two lines are intersect
bool areIntersect(line l1, line l2, point &p) {
 if (areParallel(l1, l2)) return false;            // no intersection
 // solve system of 2 linear algebraic equations with 2 unknowns
 p.x = (l2.b*l1.c - l1.b*l2.c) / (l2.a*l1.b - l1.a*l2.b);
 // special case: test for vertical line to avoid division by zero
 if (fabs(l1.b) > EPS) p.y = -(l1.a*p.x + l1.c);
 else                  p.y = -(l2.a*p.x + l2.c);
 return true; }

struct vec { double x, y;  // name: `vec' is different from STL vector
 vec(double _x, double _y) : x(_x), y(_y) {} };

vec toVec(point a, point b) {       // convert 2 points to vector a->b
 return vec(b.x-a.x, b.y-a.y); }

vec scale(vec v, double s) {        // nonnegative s = [<1 .. 1 .. >1]
 return vec(v.x*s, v.y*s); }                   // shorter.same.longer

point translate(point p, vec v) {        // translate p according to v
 return point(p.x+v.x, p.y+v.y); }

// convert point and gradient/slope to line
void pointSlopeToLine(point p, double m, line &l) {
 l.a = -m;                                               // always -m
 l.b = 1;                                                 // always 1
 l.c = -((l.a*p.x) + (l.b*p.y)); }                    // compute this

void closestPoint(line l, point p, point &ans) {
 line perpendicular;         // perpendicular to l and pass through p
 if (fabs(l.b) < EPS) {              // special case 1: vertical line
   ans.x = -(l.c);   ans.y = p.y;      return; }

 if (fabs(l.a) < EPS) {            // special case 2: horizontal line
   ans.x = p.x;      ans.y = -(l.c);   return; }

 pointSlopeToLine(p, 1/l.a, perpendicular);            // normal line
 // intersect line l with this perpendicular line
 // the intersection point is the closest point
 areIntersect(l, perpendicular, ans); }

// returns the reflection of point on a line
void reflectionPoint(line l, point p, point &ans) {
 point b;
 closestPoint(l, p, b);                     // similar to distToLine
 vec v = toVec(p, b);                             // create a vector
 ans = translate(translate(p, v), v); }         // translate p twice

// returns the dot product of two vectors a and b
double dot(vec a, vec b) { return (a.x*b.x + a.y*b.y); }

// returns the squared value of the normalized vector
double norm_sq(vec v) { return v.x*v.x + v.y*v.y; }

// returns the distance from p to the line defined by
// two points a and b (a and b must be different)
// the closest point is stored in the 4th parameter (byref)
double distToLine(point p, point a, point b, point &c) {
 // formula: c = a + u*ab
 vec ap = toVec(a, p), ab = toVec(a, b);
 double u = dot(ap, ab) / norm_sq(ab);
 c = translate(a, scale(ab, u));                  // translate a to c
 return dist(p, c); }           // Euclidean distance between p and c

// returns the distance from p to the line segment ab defined by
// two points a and b (still OK if a == b)
// the closest point is stored in the 4th parameter (byref)
double distToLineSegment(point p, point a, point b, point &c) {
 vec ap = toVec(a, p), ab = toVec(a, b);
 double u = dot(ap, ab) / norm_sq(ab);
 if (u < 0.0) { c = point(a.x, a.y);                   // closer to a
   return dist(p, a); }         // Euclidean distance between p and a
 if (u > 1.0) { c = point(b.x, b.y);                   // closer to b
   return dist(p, b); }         // Euclidean distance between p and b
 return distToLine(p, a, b, c); }          // run distToLine as above

double angle(point a, point o, point b) {  // returns angle aob in rad
 vec oa = toVec(o, a), ob = toVec(o, b);
 return acos(dot(oa, ob) / sqrt(norm_sq(oa)*norm_sq(ob))); }

// returns the cross product of two vectors a and b
double cross(vec a, vec b) { return a.x*b.y - a.y*b.x; }

//// another variant
// returns 'twice' the area of this triangle A-B-c
// int area2(point p, point q, point r) {
//   return p.x * q.y - p.y * q.x +
//          q.x * r.y - q.y * r.x +
//          r.x * p.y - r.y * p.x;
// }

// note: to accept collinear points, we have to change the `> 0'
// returns true if point r is on the left side of line pq
bool ccw(point p, point q, point r) {
 return cross(toVec(p, q), toVec(p, r)) > -EPS; }

// returns true if point r is on the same line as the line pq
bool collinear(point p, point q, point r) {
 return fabs(cross(toVec(p, q), toVec(p, r))) < EPS; }

int main() {
 point P1, P2, P3(0, 1); // note that both P1 and P2 are (0.00, 0.00)
 printf("%d\n", P1 == P2);                                    // true
 printf("%d\n", P1 == P3);                                   // false

 vector<point> P;
 P.push_back(point(2, 2));
 P.push_back(point(4, 3));
 P.push_back(point(2, 4));
 P.push_back(point(6, 6));
 P.push_back(point(2, 6));
 P.push_back(point(6, 5));

 // sorting points demo
 sort(P.begin(), P.end());
 for (int i = 0; i < (int)P.size(); i++)
   printf("(%.2lf, %.2lf)\n", P[i].x, P[i].y);

 // rearrange the points as shown in the diagram below
 P.clear();
 P.push_back(point(2, 2));
 P.push_back(point(4, 3));
 P.push_back(point(2, 4));
 P.push_back(point(6, 6));
 P.push_back(point(2, 6));
 P.push_back(point(6, 5));
 P.push_back(point(8, 6));

 /*
 // the positions of these 7 points (0-based indexing)
 6   P4      P3  P6
 5           P5
 4   P2
 3       P1
 2   P0
 1
 0 1 2 3 4 5 6 7 8
 */

 double d = dist(P[0], P[5]);
 printf("Euclidean distance between P[0] and P[5] = %.2lf\n", d); // should be 5.000

 // line equations
 line l1, l2, l3, l4;
 pointsToLine(P[0], P[1], l1);
 printf("%.2lf * x + %.2lf * y + %.2lf = 0.00\n", l1.a, l1.b, l1.c); // should be -0.50 * x + 1.00 * y - 1.00 = 0.00

 pointsToLine(P[0], P[2], l2); // a vertical line, not a problem in "ax + by + c = 0" representation
 printf("%.2lf * x + %.2lf * y + %.2lf = 0.00\n", l2.a, l2.b, l2.c); // should be 1.00 * x + 0.00 * y - 2.00 = 0.00

 // parallel, same, and line intersection tests
 pointsToLine(P[2], P[3], l3);
 printf("l1 & l2 are parallel? %d\n", areParallel(l1, l2)); // no
 printf("l1 & l3 are parallel? %d\n", areParallel(l1, l3)); // yes, l1 (P[0]-P[1]) and l3 (P[2]-P[3]) are parallel

 pointsToLine(P[2], P[4], l4);
 printf("l1 & l2 are the same? %d\n", areSame(l1, l2)); // no
 printf("l2 & l4 are the same? %d\n", areSame(l2, l4)); // yes, l2 (P[0]-P[2]) and l4 (P[2]-P[4]) are the same line (note, they are two different line segments, but same line)
  point p12;
 bool res = areIntersect(l1, l2, p12); // yes, l1 (P[0]-P[1]) and l2 (P[0]-P[2]) are intersect at (2.0, 2.0)
 printf("l1 & l2 are intersect? %d, at (%.2lf, %.2lf)\n", res, p12.x, p12.y);

 // other distances
 point ans;
 d = distToLine(P[0], P[2], P[3], ans);
 printf("Closest point from P[0] to line         (P[2]-P[3]): (%.2lf, %.2lf), dist = %.2lf\n", ans.x, ans.y, d);
 closestPoint(l3, P[0], ans);
 printf("Closest point from P[0] to line V2      (P[2]-P[3]): (%.2lf, %.2lf), dist = %.2lf\n", ans.x, ans.y, dist(P[0], ans));

 d = distToLineSegment(P[0], P[2], P[3], ans);
 printf("Closest point from P[0] to line SEGMENT (P[2]-P[3]): (%.2lf, %.2lf), dist = %.2lf\n", ans.x, ans.y, d); // closer to A (or P[2]) = (2.00, 4.00)
 d = distToLineSegment(P[1], P[2], P[3], ans);
 printf("Closest point from P[1] to line SEGMENT (P[2]-P[3]): (%.2lf, %.2lf), dist = %.2lf\n", ans.x, ans.y, d); // closer to midway between AB = (3.20, 4.60)
 d = distToLineSegment(P[6], P[2], P[3], ans);
 printf("Closest point from P[6] to line SEGMENT (P[2]-P[3]): (%.2lf, %.2lf), dist = %.2lf\n", ans.x, ans.y, d); // closer to B (or P[3]) = (6.00, 6.00)

 reflectionPoint(l4, P[1], ans);
 printf("Reflection point from P[1] to line      (P[2]-P[4]): (%.2lf, %.2lf)\n", ans.x, ans.y); // should be (0.00, 3.00)

 printf("Angle P[0]-P[4]-P[3] = %.2lf\n", RAD_to_DEG(angle(P[0], P[4], P[3]))); // 90 degrees
 printf("Angle P[0]-P[2]-P[1] = %.2lf\n", RAD_to_DEG(angle(P[0], P[2], P[1]))); // 63.43 degrees
 printf("Angle P[4]-P[3]-P[6] = %.2lf\n", RAD_to_DEG(angle(P[4], P[3], P[6]))); // 180 degrees

 printf("P[0], P[2], P[3] form A left turn? %d\n", ccw(P[0], P[2], P[3])); // no
 printf("P[0], P[3], P[2] form A left turn? %d\n", ccw(P[0], P[3], P[2])); // yes

 printf("P[0], P[2], P[3] are collinear? %d\n", collinear(P[0], P[2], P[3])); // no
 printf("P[0], P[2], P[4] are collinear? %d\n", collinear(P[0], P[2], P[4])); // yes

 point p(3, 7), q(11, 13), r(35, 30); // collinear if r(35, 31)
 printf("r is on the %s of line p-r\n", ccw(p, q, r) ? "left" : "right"); // right

 /*
 // the positions of these 6 points
    E<--  4
          3       B D<--
          2   A C
          1
 -4-3-2-1 0 1 2 3 4 5 6
         -1
         -2
  F<--   -3
 */

 // translation
 point A(2.0, 2.0);
 point B(4.0, 3.0);
 vec v = toVec(A, B); // imagine there is an arrow from A to B (see the diagram above)
 point C(3.0, 2.0);
 point D = translate(C, v); // D will be located in coordinate (3.0 + 2.0, 2.0 + 1.0) = (5.0, 3.0)
 printf("D = (%.2lf, %.2lf)\n", D.x, D.y);
 point E = translate(C, scale(v, 0.5)); // E will be located in coordinate (3.0 + 1/2 * 2.0, 2.0 + 1/2 * 1.0) = (4.0, 2.5)
 printf("E = (%.2lf, %.2lf)\n", E.x, E.y);

 // rotation
 printf("B = (%.2lf, %.2lf)\n", B.x, B.y); // B = (4.0, 3.0)
 point F = rotate(B, 90); // rotate B by 90 degrees COUNTER clockwise, F = (-3.0, 4.0)
 printf("F = (%.2lf, %.2lf)\n", F.x, F.y);
 point G = rotate(B, 180); // rotate B by 180 degrees COUNTER clockwise, G = (-4.0, -3.0)
 printf("G = (%.2lf, %.2lf)\n", G.x, G.y);

 return 0;

// circles
#include <bits/stdc++.h>
using namespace std;

#define INF 1e9
#define EPS 1e-9
#define PI acos(-1.0)

double DEG_to_RAD(double d) { return d * PI / 180.0; }

double RAD_to_DEG(double r) { return r * 180.0 / PI; }

struct point_i { int x, y;     // whenever possible, work with point_i
 point_i() { x = y = 0; }                      // default constructor
 point_i(int _x, int _y) : x(_x), y(_y) {} };          // constructor

struct point { double x, y;   // only used if more precision is needed
 point() { x = y = 0.0; }                      // default constructor
 point(double _x, double _y) : x(_x), y(_y) {} };      // constructor

int insideCircle(point_i p, point_i c, int r) { // all integer version
 int dx = p.x - c.x, dy = p.y - c.y;
 int Euc = dx * dx + dy * dy, rSq = r * r;             // all integer
 return Euc < rSq ? 0 : Euc == rSq ? 1 : 2; } //inside/border/outside

bool circle2PtsRad(point p1, point p2, double r, point &c) {
 double d2 = (p1.x - p2.x) * (p1.x - p2.x) +
             (p1.y - p2.y) * (p1.y - p2.y);
 double det = r * r / d2 - 0.25;
 if (det < 0.0) return false;
 double h = sqrt(det);
 c.x = (p1.x + p2.x) * 0.5 + (p1.y - p2.y) * h;
 c.y = (p1.y + p2.y) * 0.5 + (p2.x - p1.x) * h;
 return true; }         // to get the other center, reverse p1 and p2

int main() {
 // circle equation, inside, border, outside
 point_i pt(2, 2);
 int r = 7;
 point_i inside(8, 2);
 printf("%d\n", insideCircle(inside, pt, r));             // 0-inside
 point_i border(9, 2);
 printf("%d\n", insideCircle(border, pt, r));          // 1-at border
 point_i outside(10, 2);
 printf("%d\n", insideCircle(outside, pt, r));           // 2-outside

 double d = 2 * r;
 printf("Diameter = %.2lf\n", d);
 double c = PI * d;
 printf("Circumference (Perimeter) = %.2lf\n", c);
 double A = PI * r * r;
 printf("Area of circle = %.2lf\n", A);

 printf("Length of arc   (central angle = 60 degrees) = %.2lf\n", 60.0 / 360.0 * c);
 printf("Length of chord (central angle = 60 degrees) = %.2lf\n", sqrt((2 * r * r) * (1 - cos(DEG_to_RAD(60.0)))));
 printf("Area of sector  (central angle = 60 degrees) = %.2lf\n", 60.0 / 360.0 * A);

 point p1;
 point p2(0.0, -1.0);
 point ans;
 circle2PtsRad(p1, p2, 2.0, ans);
 printf("One of the center is (%.2lf, %.2lf)\n", ans.x, ans.y);
 circle2PtsRad(p2, p1, 2.0, ans);     // we simply reverse p1 with p2
 printf("The other center  is (%.2lf, %.2lf)\n", ans.x, ans.y);

 return 0;
}

//polygon
#include <bits/stdc++.h>
using namespace std;

#define EPS 1e-9
#define PI acos(-1.0)

double DEG_to_RAD(double d) { return d*PI / 180.0; }

double RAD_to_DEG(double r) { return r*180.0 / PI; }

struct point { double x, y;   // only used if more precision is needed
 point() { x = y = 0.0; }                      // default constructor
 point(double _x, double _y) : x(_x), y(_y) {}        // user-defined
 bool operator == (point other) const {
  return (fabs(x-other.x) < EPS && (fabs(y-other.y) < EPS)); }
 bool operator <(const point &p) const {
  return x < p.x || (abs(x-p.x) < EPS && y < p.y); } };

struct vec { double x, y;  // name: `vec' is different from STL vector
 vec(double _x, double _y) : x(_x), y(_y) {} };

vec toVec(point a, point b) {       // convert 2 points to vector a->b
 return vec(b.x-a.x, b.y-a.y); }

double dist(point p1, point p2) {                // Euclidean distance
 return hypot(p1.x-p2.x, p1.y-p2.y); }               // return double

// returns the perimeter, which is the sum of Euclidian distances
// of consecutive line segments (polygon edges)
double perimeter(const vector<point> &P) {
 double result = 0.0;
 for (int i = 0; i < (int)P.size()-1; i++)  // remember that P[0] = P[n-1]
   result += dist(P[i], P[i+1]);
 return result; }

// returns the area
double area(const vector<point> &P) {
 double result = 0.0;
 for (int i = 0; i < (int)P.size()-1; i++)             // Shoelace formula
   result += (P[i].x*P[i+1].y - P[i+1].x*P[i].y); // if all points are int
 return fabs(result)/2.0; }     // result can be int(eger) until last step

double dot(vec a, vec b) { return (a.x*b.x + a.y*b.y); }

double norm_sq(vec v) { return v.x*v.x + v.y*v.y; }

double angle(point a, point o, point b) {  // returns angle aob in rad
 vec oa = toVec(o, a), ob = toVec(o, b);
 return acos(dot(oa, ob) / sqrt(norm_sq(oa) * norm_sq(ob))); }

double cross(vec a, vec b) { return a.x*b.y - a.y*b.x; }

double area_alternative(const vector<point> &P) {
 double result = 0.0; point O(0.0, 0.0);
 for (int i = 0; i < (int)P.size()-1; i++)
   result += cross(toVec(O, P[i]), toVec(O, P[i+1]));
 return fabs(result) / 2.0; }

// note: to accept collinear points, we have to change the `> 0'
// returns true if point r is on the left side of line pq
bool ccw(point p, point q, point r) {
 return cross(toVec(p, q), toVec(p, r)) > 0; }

// returns true if point r is on the same line as the line pq
bool collinear(point p, point q, point r) {
 return fabs(cross(toVec(p, q), toVec(p, r))) < EPS; }

// returns true if we always make the same turn while examining
// all the edges of the polygon one by one
bool isConvex(const vector<point> &P) {
 int sz = (int)P.size();
 if (sz <= 3) return false;   // a point/sz=2 or a line/sz=3 is not convex
 bool firstTurn = ccw(P[0], P[1], P[2]);            // remember one result
 for (int i = 1; i < sz-1; i++)            // then compare with the others
   if (ccw(P[i], P[i+1], P[(i+2) == sz ? 1 : i+2]) != firstTurn)
     return false;            // different sign -> this polygon is concave
 return true; }                                  // this polygon is convex

// returns true if point p is in either convex/concave polygon P
bool inPolygon(point pt, const vector<point> &P) {
 if ((int)P.size() < 3) return false;               // avoid point or line
 double sum = 0;    // assume the first vertex is equal to the last vertex
 for (int i = 0; i < (int)P.size()-1; i++) {
   if (ccw(pt, P[i], P[i+1]))
        sum += angle(P[i], pt, P[i+1]);                   // left turn/ccw
   else sum -= angle(P[i], pt, P[i+1]); }                 // right turn/cw
 return fabs(sum) > PI; }   // 360d -> in, 0d -> out, we have large margin

// line segment p-q intersect with line A-B.
point lineIntersectSeg(point p, point q, point A, point B) {
 double a = B.y - A.y;
 double b = A.x - B.x;
 double c = B.x * A.y - A.x * B.y;
 double u = fabs(a * p.x + b * p.y + c);
 double v = fabs(a * q.x + b * q.y + c);
 return point((p.x * v + q.x * u) / (u+v), (p.y * v + q.y * u) / (u+v)); }

// cuts polygon Q along the line formed by point a -> point b
// (note: the last point must be the same as the first point)
vector<point> cutPolygon(point a, point b, const vector<point> &Q) {
 vector<point> P;
 for (int i = 0; i < (int)Q.size(); i++) {
   double left1 = cross(toVec(a, b), toVec(a, Q[i])), left2 = 0;
   if (i != (int)Q.size()-1) left2 = cross(toVec(a, b), toVec(a, Q[i+1]));
   if (left1 > -EPS) P.push_back(Q[i]);       // Q[i] is on the left of ab
   if (left1 * left2 < -EPS)        // edge (Q[i], Q[i+1]) crosses line ab
     P.push_back(lineIntersectSeg(Q[i], Q[i+1], a, b));
 }
 if (!P.empty() && !(P.back() == P.front()))
   P.push_back(P.front());        // make P's first point = P's last point
 return P; }

vector<point> CH_Andrew(vector<point> &Pts) {
 int n = Pts.size(), k = 0;
 vector<point> H(2*n);
 sort(Pts.begin(), Pts.end());        // sort the points lexicographically
 for (int i = 0; i < n; i++) {                         // build lower hull
   while (k >= 2 && ccw(H[k-2], H[k-1], Pts[i]) <= 0) k--;
   H[k++] = Pts[i];
 }
 for (int i = n-2, t = k+1; i >= 0; i--) {             // build upper hull
   while (k >= t && ccw(H[k-2], H[k-1], Pts[i]) <= 0) k--;
   H[k++] = Pts[i];
 }
 H.resize(k);
 return H;
}

point pivot(0, 0);
vector<point> CH_Graham(vector<point> &Pts) {
 vector<point> P(Pts);      // copy all points so that Pts is not affected
 int i, j, n = (int)P.size();
 if (n <= 3) {          // corner cases: n=1=point, n=2=line, n=3=triangle
   if (!(P[0] == P[n-1])) P.push_back(P[0]); // safeguard from corner case
   return P; }                                       // the CH is P itself

 // first, find P0 = point with lowest Y and if tie: rightmost X
 int P0 = 0;
 for (i = 1; i < n; i++)                                           // O(n)
   if (P[i].y < P[P0].y || (P[i].y == P[P0].y && P[i].x > P[P0].x))
     P0 = i;
 swap(P[0], P[P0]);                                // swap P[P0] with P[0]

 // second, sort points by angle w.r.t. pivot P0, O(n log n) for this sort
 pivot = P[0];                    // use this global variable as reference
 sort(++P.begin(), P.end(), [](point a, point b) {  // we do not sort P[0]
   if (collinear(pivot, a, b))                             // special case
     return dist(pivot, a) < dist(pivot, b);  // check which one is closer
   double d1x = a.x-pivot.x, d1y = a.y-pivot.y;
   double d2x = b.x-pivot.x, d2y = b.y-pivot.y;
   return (atan2(d1y, d1x) - atan2(d2y, d2x)) < 0; }); // compare 2 angles

 // third, the ccw tests, although complex, it is just O(n)
 vector<point> S;
 S.push_back(P[n-1]); S.push_back(P[0]); S.push_back(P[1]);   // initial S
 i = 2;                                         // then, we check the rest
 while (i < n) {     // note: n must be >= 3 for this method to work, O(n)
   j = (int)S.size()-1;
   if (ccw(S[j-1], S[j], P[i])) S.push_back(P[i++]);  // left turn, accept
   else S.pop_back(); }   // or pop the top of S until we have a left turn
 return S; } // return the result, overall O(n log n) due to angle sorting

int main() {
 // 6 points, entered in counter clockwise order, 0-based indexing
 vector<point> P;
 P.emplace_back(1, 1);
 P.emplace_back(3, 3);
 P.emplace_back(9, 1);
 P.emplace_back(12, 4);
 P.emplace_back(9, 7);
 P.emplace_back(1, 7);
 P.push_back(P[0]); // loop back

 printf("Perimeter of polygon = %.2lf\n", perimeter(P)); // 31.64
 printf("Area of polygon = %.2lf\n", area(P)); // 49.00
 printf("Area of polygon = %.2lf\n", area_alternative(P)); // 49.00
 printf("Is convex = %d\n", isConvex(P)); // false (P1 is the culprit)

 //// the positions of P6 and P7 w.r.t the polygon
 //7 P5--------------P4
 //6 |                  \
 //5 |                    \
 //4 |   P7                P3
 //3 |   P1___            /
 //2 | / P6    \ ___    /
 //1 P0              P2
 //0 1 2 3 4 5 6 7 8 9 101112

 point P6(3, 2); // outside this (concave) polygon
 printf("Point P6 is inside this polygon = %d\n", inPolygon(P6, P)); // false
 point P7(3, 4); // inside this (concave) polygon
 printf("Point P7 is inside this polygon = %d\n", inPolygon(P7, P)); // true

 // cutting the original polygon based on line P[2] -> P[4] (get the left side)
 //7 P5--------------P4
 //6 |               |  \
 //5 |               |    \
 //4 |               |     P3
 //3 |   P1___       |    /
 //2 | /       \ ___ |  /
 //1 P0              P2
 //0 1 2 3 4 5 6 7 8 9 101112
 // new polygon (notice the index are different now):
 //7 P4--------------P3
 //6 |               |
 //5 |               |
 //4 |               |
 //3 |   P1___       |
 //2 | /       \ ___ |
 //1 P0              P2
 //0 1 2 3 4 5 6 7 8 9

 P = cutPolygon(P[2], P[4], P);
 printf("Perimeter of polygon = %.2lf\n", perimeter(P)); // smaller now 29.15
 printf("Area of polygon = %.2lf\n", area(P)); // 40.00

 // running convex hull of the resulting polygon (index changes again)
 //7 P3--------------P2
 //6 |               |
 //5 |               |
 //4 |   P7          |
 //3 |               |
 //2 |               |
 //1 P0--------------P1
 //0 1 2 3 4 5 6 7 8 9

 P = CH_Andrew(P); // now this is a rectangle
 printf("Perimeter of polygon = %.2lf\n", perimeter(P)); // precisely 28.00
 printf("Area of polygon = %.2lf\n", area(P)); // precisely 48.00
 printf("Is convex = %d\n", isConvex(P)); // true
 printf("Point P6 is inside this polygon = %d\n", inPolygon(P6, P)); // true
 printf("Point P7 is inside this polygon = %d\n", inPolygon(P7, P)); // true

 return 0;
}

//triangles
#include <bits/stdc++.h>
using namespace std;

#define EPS 1e-9
#define PI acos(-1.0)

double DEG_to_RAD(double d) { return d * PI / 180.0; }

double RAD_to_DEG(double r) { return r * 180.0 / PI; }

struct point_i { int x, y;     // whenever possible, work with point_i
 point_i() { x = y = 0; }                      // default constructor
 point_i(int _x, int _y) : x(_x), y(_y) {} };          // constructor

struct point { double x, y;   // only used if more precision is needed
 point() { x = y = 0.0; }                      // default constructor
 point(double _x, double _y) : x(_x), y(_y) {} };      // constructor

double dist(point p1, point p2) {
 return hypot(p1.x - p2.x, p1.y - p2.y); }

double perimeter(double ab, double bc, double ca) {
 return ab + bc + ca; }

double perimeter(point a, point b, point c) {
 return dist(a, b) + dist(b, c) + dist(c, a); }

double area(double ab, double bc, double ca) {
 // Heron's formula, split sqrt(a * b) into sqrt(a) * sqrt(b); in implementation
 double s = 0.5 * perimeter(ab, bc, ca);
 return sqrt(s) * sqrt(s - ab) * sqrt(s - bc) * sqrt(s - ca); }

double area(point a, point b, point c) {
 return area(dist(a, b), dist(b, c), dist(c, a)); }

//====================================================================
// from ch7_01_points_lines
struct line { double a, b, c; }; // a way to represent a line

// the answer is stored in the third parameter (pass by reference)
void pointsToLine(point p1, point p2, line &l) {
 if (fabs(p1.x - p2.x) < EPS) {              // vertical line is fine
   l.a = 1.0;   l.b = 0.0;   l.c = -p1.x;           // default values
 } else {
   l.a = -(double)(p1.y - p2.y) / (p1.x - p2.x);
   l.b = 1.0;              // IMPORTANT: we fix the value of b to 1.0
   l.c = -(double)(l.a * p1.x) - p1.y;
} }

bool areParallel(line l1, line l2) {        // check coefficient a + b
 return (fabs(l1.a-l2.a) < EPS) && (fabs(l1.b-l2.b) < EPS); }

// returns true (+ intersection point) if two lines are intersect
bool areIntersect(line l1, line l2, point &p) {
 if (areParallel(l1, l2)) return false;            // no intersection
 // solve system of 2 linear algebraic equations with 2 unknowns
 p.x = (l2.b * l1.c - l1.b * l2.c) / (l2.a * l1.b - l1.a * l2.b);
 // special case: test for vertical line to avoid division by zero
 if (fabs(l1.b) > EPS) p.y = -(l1.a * p.x + l1.c);
 else                  p.y = -(l2.a * p.x + l2.c);
 return true; }

struct vec { double x, y;  // name: `vec' is different from STL vector
 vec(double _x, double _y) : x(_x), y(_y) {} };

vec toVec(point a, point b) {       // convert 2 points to vector a->b
 return vec(b.x - a.x, b.y - a.y); }

vec scale(vec v, double s) {        // nonnegative s = [<1 .. 1 .. >1]
 return vec(v.x * s, v.y * s); }               // shorter.same.longer

point translate(point p, vec v) {        // translate p according to v
 return point(p.x + v.x , p.y + v.y); }
//====================================================================

double rInCircle(double ab, double bc, double ca) {
 return area(ab, bc, ca) / (0.5 * perimeter(ab, bc, ca)); }

double rInCircle(point a, point b, point c) {
 return rInCircle(dist(a, b), dist(b, c), dist(c, a)); }

// assumption: the required points/lines functions have been written
// returns 1 if there is an inCircle center, returns 0 otherwise
// if this function returns 1, ctr will be the inCircle center
// and r is the same as rInCircle
int inCircle(point p1, point p2, point p3, point &ctr, double &r) {
 r = rInCircle(p1, p2, p3);
 if (fabs(r) < EPS) return 0;                   // no inCircle center

 line l1, l2;                    // compute these two angle bisectors
 double ratio = dist(p1, p2) / dist(p1, p3);
 point p = translate(p2, scale(toVec(p2, p3), ratio / (1 + ratio)));
 pointsToLine(p1, p, l1);

 ratio = dist(p2, p1) / dist(p2, p3);
 p = translate(p1, scale(toVec(p1, p3), ratio / (1 + ratio)));
 pointsToLine(p2, p, l2);

 areIntersect(l1, l2, ctr);           // get their intersection point
 return 1; }

double rCircumCircle(double ab, double bc, double ca) {
 return ab * bc * ca / (4.0 * area(ab, bc, ca)); }

double rCircumCircle(point a, point b, point c) {
 return rCircumCircle(dist(a, b), dist(b, c), dist(c, a)); }

// assumption: the required points/lines functions have been written
// returns 1 if there is a circumCenter center, returns 0 otherwise
// if this function returns 1, ctr will be the circumCircle center
// and r is the same as rCircumCircle
int circumCircle(point p1, point p2, point p3, point &ctr, double &r){
 double a = p2.x - p1.x, b = p2.y - p1.y;
 double c = p3.x - p1.x, d = p3.y - p1.y;
 double e = a * (p1.x + p2.x) + b * (p1.y + p2.y);
 double f = c * (p1.x + p3.x) + d * (p1.y + p3.y);
 double g = 2.0 * (a * (p3.y - p2.y) - b * (p3.x - p2.x));
 if (fabs(g) < EPS) return 0;

 ctr.x = (d*e - b*f) / g;
 ctr.y = (a*f - c*e) / g;
 r = dist(p1, ctr);  // r = distance from center to 1 of the 3 points
 return 1; }

// returns true if point d is inside the circumCircle defined by a,b,c
int inCircumCircle(point a, point b, point c, point d) {
 return (a.x - d.x) * (b.y - d.y) * ((c.x - d.x) * (c.x - d.x) + (c.y - d.y) * (c.y - d.y)) +
        (a.y - d.y) * ((b.x - d.x) * (b.x - d.x) + (b.y - d.y) * (b.y - d.y)) * (c.x - d.x) +
        ((a.x - d.x) * (a.x - d.x) + (a.y - d.y) * (a.y - d.y)) * (b.x - d.x) * (c.y - d.y) -
        ((a.x - d.x) * (a.x - d.x) + (a.y - d.y) * (a.y - d.y)) * (b.y - d.y) * (c.x - d.x) -
        (a.y - d.y) * (b.x - d.x) * ((c.x - d.x) * (c.x - d.x) + (c.y - d.y) * (c.y - d.y)) -
        (a.x - d.x) * ((b.x - d.x) * (b.x - d.x) + (b.y - d.y) * (b.y - d.y)) * (c.y - d.y) > 0 ? 1 : 0;
}

bool canFormTriangle(double a, double b, double c) {
 return (a + b > c) && (a + c > b) && (b + c > a); }

int main() {
 double base = 4.0, h = 3.0;
 double A = 0.5 * base * h;
 printf("Area = %.2lf\n", A);

 point a;                                         // a right triangle
 point b(4.0, 0.0);
 point c(4.0, 3.0);

 double p = perimeter(a, b, c);
 double s = 0.5 * p;
 A = area(a, b, c);
 printf("Area = %.2lf\n", A);            // must be the same as above

 double r = rInCircle(a, b, c);
 printf("R1 (radius of incircle) = %.2lf\n", r);              // 1.00
 point ctr;
 int res = inCircle(a, b, c, ctr, r);
 printf("R1 (radius of incircle) = %.2lf\n", r);        // same, 1.00
 printf("Center = (%.2lf, %.2lf)\n", ctr.x, ctr.y);   // (3.00, 1.00)

 printf("R2 (radius of circumcircle) = %.2lf\n", rCircumCircle(a, b, c)); // 2.50
 res = circumCircle(a, b, c, ctr, r);
 printf("R2 (radius of circumcircle) = %.2lf\n", r);   // same, 2.50
 printf("Center = (%.2lf, %.2lf)\n", ctr.x, ctr.y);   // (2.00, 1.50)

 point d(2.0, 1.0);               // inside triangle and circumCircle
 printf("d inside circumCircle (a, b, c) ? %d\n", inCircumCircle(a, b, c, d));
 point e(2.0, 3.9);   // outside the triangle but inside circumCircle
 printf("e inside circumCircle (a, b, c) ? %d\n", inCircumCircle(a, b, c, e));
 point f(2.0, -1.1);                              // slightly outside
 printf("f inside circumCircle (a, b, c) ? %d\n", inCircumCircle(a, b, c, f));

 // Law of Cosines
 double ab = dist(a, b);
 double bc = dist(b, c);
 double ca = dist(c, a);
 double alpha = RAD_to_DEG(acos((ca * ca + ab * ab - bc * bc) / (2.0 * ca * ab)));
 printf("alpha = %.2lf\n", alpha);
 double beta  = RAD_to_DEG(acos((ab * ab + bc * bc - ca * ca) / (2.0 * ab * bc)));
 printf("beta  = %.2lf\n", beta);
 double gamma = RAD_to_DEG(acos((bc * bc + ca * ca - ab * ab) / (2.0 * bc * ca)));
 printf("gamma = %.2lf\n", gamma);

 // Law of Sines
 printf("%.2lf == %.2lf == %.2lf\n", bc / sin(DEG_to_RAD(alpha)), ca / sin(DEG_to_RAD(beta)), ab / sin(DEG_to_RAD(gamma)));

 // Phytagorean Theorem
 printf("%.2lf^2 == %.2lf^2 + %.2lf^2\n", ca, ab, bc);

 // Triangle Inequality
 printf("(%d, %d, %d) => can form triangle? %d\n", 3, 4, 5, canFormTriangle(3, 4, 5)); // yes
 printf("(%d, %d, %d) => can form triangle? %d\n", 3, 4, 7, canFormTriangle(3, 4, 7)); // no, actually straight line
 printf("(%d, %d, %d) => can form triangle? %d\n", 3, 4, 8, canFormTriangle(3, 4, 8)); // no

 return 0;
}

convex hull - need to figure out;








data structures; Union-Find Disjoint Set, stack, queue, priority queue - listing these doesn't mean you understand them, meanless 						
chap 3 paradigms, strategies					
					 							3.2  Complete Search						 							3.3  Divide and Conquer						 							3.4  Greedy						 							3.5  Dynamic Programming  			
chap 4 graph
strongly connected components explained -
https://people.eecs.berkeley.edu/~vazirani/s99cs170/notes/lec12.pdf

https://cs.stackexchange.com/questions/12559/why-is-the-node-with-the-greatest-dfs-post-order-number-not-necessarily-a-sink

https://www.cs.cmu.edu/~avrim/451f13/lectures/lect0919.pdf
priority queue
https://www.geeksforgeeks.org/priority-queue-set-1-introduction/

warshall's algorithm					
						
for (int k = 0; k < V; k++)
 for (int i = 0; i < V; i++)
for (int j = 0; j < V; j++)
AdjMat[i][j] = min(AdjMat[i][j], max(AdjMat[i][k], AdjMat[k][j]));
					
				
			
		
ch6 is about strings - trie is pronounced same as tree; suffix array is a new data structure;  4 categories of problems: ad hoc string problem; string matching; DP string processing; suffix trie/tree/array; her reading books is waste of time; explain and teach her instead of letting her read herself; do more practice problems; practice, practice, practice.
	
practice practice practice
array index i j k very easy to make mistakes; use ii, jj, kk or more meaningful names to reduce mistakes
many makes mess; links don't mean anything, can't make sense out of it; special terms don't mean anything, can't make sense and use it; what are useful?
https://sites.google.com/view/procoding 
https://sites.google.com/iu.edu/programming

					 				
			
		
	 
