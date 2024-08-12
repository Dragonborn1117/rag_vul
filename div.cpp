#include <iostream>
#include <vector>

using namespace std;

vector<int> div(vector<int> &A, int b) {
    int carry = 0;
    vector<int> C;

    for(int i = 0; i < A.size(); i++) {
        carry += A[i] * b;
        C.push_back(carry % 10);
        carry /= 10;
    }
    if(carry >= 1) {
        C.push_back(carry);
    }
    return C;
}

int main() {
    string a;
    int b;
    vector<int> A;

    cin >> a >> b;
    for(int i = a.size() - 1; i >= 0; i--) {
        A.push_back(a[i] - '0');
    }

    auto C = div(A, b);
    for (int i = C.size() - 1; i >= 0; i--) {
        printf("%d", C[i]);
    }
    printf("\n");
    
}