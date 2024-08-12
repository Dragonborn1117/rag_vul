#include <iostream>

using namespace std;

const int N = 100000;
int q[N];

int binary_search(int arr[], int numsize, int target) {
    int l = 0, r = numsize - 1, mid;
    while(l < r) {
        mid = l + r >> 1;
        if(target <= arr[mid]) r = mid;
        else l = mid + 1;
    }
    if (arr[r] != target) {
        return -1;
    }
    return r;
}

int binary_search_2(int arr[], int numsize, int target) {
    int l = 0, r = numsize - 1, mid;
    while (l < r) {
        mid = l + r + 1 >> 1;
        if(arr[mid] <= target) {
            l = mid;
        } else {
            r = mid - 1;
        }
    }
    if(arr[r] != target) {
        return -1;
    }
    return r;
}

void quicksort(int arr[], int low, int high) {
    if(low >= high) {
        return;
    }
    int i = low - 1, j = high + 1, pivot = arr[low];
    while(i < j) {
        do {
            i++;
        } while(pivot > arr[i]);
        do {
            j--;
        } while(pivot < arr[j]);
        if (i < j) {
            swap(arr[i], arr[j]);
        }
    }
    quicksort(arr, low, j);
    quicksort(arr, j + 1, high);

}

void mergesort(int arr[], int low, int high) {
    if (low >= high) {
        return;
    }

    int mid = low + high >> 1;
    mergesort(arr, low, mid);
    mergesort(arr, mid + 1, high);

    int i = low, j = mid + 1, k = 0;
    while(i <= mid && j <= high) {
        if(arr[i] < arr[j]) {
            q[k++] = arr[i++];
        } else {
            q[k++] = arr[j++];
        }
    }
    while(i <= mid) {
        q[k++] = arr[i++];
    }
    while(j <= high) {
        q[k++] = arr[j++];
    }
    for(int i = low, j = 0; i <= high; i++, j++) {
        arr[i] = q[j];
    }
}

int main(int argc, char *argv[]) {  
    int num[] = {5, 3, 3, -1, 7};
    int size = sizeof(num)/sizeof(*num);
    //quicksort(num, 0, size - 1);
    mergesort(num, 0, size-1);
    for (int i=0; i < size; i++) {
        printf("%d ", num[i]);
    }
    int index = binary_search(num, size, 7);
    printf("index: %d", index);
    printf("\n");
    return 0;
}