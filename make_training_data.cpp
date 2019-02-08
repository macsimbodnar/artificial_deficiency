#include <iostream>
#include <cmath>
#include <cstdlib>

using namespace std;

int main(int argc, char *argv[])
{
    // Random training sets for XOR

    cout << "topology: 3 4 3 2" << endl;
    for (int i = 1000000; i >= 0; i--)
    {
        int n1 = (int) (2.0 * rand() / double(RAND_MAX));
        int n2 = (int) (2.0 * rand() / double(RAND_MAX));
        int n3 = (int) (2.0 * rand() / double(RAND_MAX));

        int t1 = n1 & n2; // should be 0 or 1
        int t2 = n2 ^ n3;

        cout << "in: " << n1 << ".0 " << n2 << ".0 " << n3 << ".0" << endl;
        cout << "out: " << t1 << ".0 " << t2 << ".0" << endl;
    } 
}