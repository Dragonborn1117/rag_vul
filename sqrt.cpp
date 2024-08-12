#include <iostream>

using namespace std;

const double N = 1e-8;

void binary_sqrt(double x) {
    double l = 0, r = x, mid;
    while (r -l > N) {
        mid = (l + r) / 2;
        if (mid * mid * mid >= x) {
            r = mid;
        } else {
            l = mid;
        }
    }
    printf("%lf\n", mid); 
}

int main() {
    double x = 101;
    binary_sqrt(x);
}