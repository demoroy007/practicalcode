#include <iostream>
#include <omp.h>
#include <vector>
#include <climits>
using namespace std;
int main()
{
    int n;
    cout << "Enter number of elements: ";
    cin >> n;
    vector<int> arr(n);
    cout << "Enter " << n << " elements:\n";
    for (int i = 0; i < n; ++i)
        cin >> arr[i];
    // Sequential Operations
    int seq_min = arr[0], seq_max = arr[0], seq_sum = 0;
    double seq_start = omp_get_wtime();
    for (int i = 0; i < n; ++i)
    {
        if (arr[i] < seq_min)
            seq_min = arr[i];
        if (arr[i] > seq_max)
            seq_max = arr[i];
        seq_sum += arr[i];
    }
    double seq_end = omp_get_wtime();
    // Parallel Reduction Operations
    int par_min = INT_MAX, par_max = INT_MIN, par_sum = 0;
    double par_start = omp_get_wtime();

#pragma omp parallel for reduction(min : par_min) reduction(max : par_max) reduction(+ : par_sum)
    for (int i = 0; i < n; ++i)
    {
        par_min = min(par_min, arr[i]);
        par_max = max(par_max, arr[i]);
        par_sum += arr[i];
    }
    double par_end = omp_get_wtime();
    // Output Results
    cout << "\nSequential Results:\n";
    cout << "Min: " << seq_min << ", Max: " << seq_max << ", Sum: " << seq_sum
         << ", Average: " << (double)seq_sum / n << endl;
    cout << "Time taken: " << (seq_end - seq_start) << " seconds\n";
    cout << "\nParallel Results (using OpenMP reduction):\n";
    cout << "Min: " << par_min << ", Max: " << par_max << ", Sum: " << par_sum
         << ", Average: " << (double)par_sum / n << endl;
    cout << "Time taken: " << (par_end - par_start) << " seconds\n";
    return 0;
}
