#include <iostream>
#include <vector>

using namespace std;

vector<int> add(vector<int> &A, vector<int> &B) {
    int carry = 0;
    vector<int> C;
    for(int i = 0; i < A.size() || i < B.size(); i++) {
        if(i < A.size()) {
            carry += A[i];
        }
        if(i < B.size()) {
            carry += B[i];
        }
        C.push_back(carry % 10);
        carry /= 10;
    }
    if(carry == 1) {
        C.push_back(1);
    }
    return C;
}

int main() {
    string a, b; //1234
    vector<int> A, B;

    cin >> a >> b;
    for(int i = a.size() - 1; i >= 0; i--) {
        A.push_back(a[i] - '0');  //4321
    }
    for(int i = b.size() - 1; i >= 0; i--) {
        B.push_back(b[i] - '0');
    }

    auto C = add(A, B);
    for(int i = C.size() - 1; i >= 0; i--) {
        printf("%d", C[i]);
    }
    printf("\n");
    return 0;
}