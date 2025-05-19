#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <chrono>
#include <Eigen/Dense>

// Inline dot product for 3-element float arrays
inline float dot3(const float a[3], const float b[3]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}
//copyright joshuah rainstar 2025 joshuah.rainstar@gmail.com
void matmul_23op(const float A[3][3], const float B[3][3], float C[3][3]) {
    constexpr float ONE_THIRD = 1.0f / 3.0f;
    constexpr float ONE_HALF  = 0.5f;

    // --- B columnwise vectors (shared across all A rows)
    float Bc[3], Bdiff[3], Bskew[3];

    for (int j = 0; j < 3; ++j) {
        Bc[j]    = B[0][j] + B[1][j] + B[2][j];
        Bdiff[j] = B[1][j] - B[2][j];
        Bskew[j] = B[0][j] - ONE_HALF * (B[1][j] + B[2][j]);
    }

    for (int i = 0; i < 3; ++i) {
        // ---- A-row composite coefficients (scalar for each row)
        float Arow  = (A[i][0] + A[i][1] + A[i][2]) * ONE_THIRD;
        float A02neg = -A[i][2];
        float Amid  = (A[i][1] + A02neg) * ONE_HALF;
        float Acomb = (2*A[i][0] - A[i][1] + A02neg) * ONE_THIRD;

        // ---- Accumulate each C[i][j] via dot product
        float coeffs[3] = {Arow, Amid, Acomb};

        for (int j = 0; j < 3; ++j) {
            float vecs[3] = {Bc[j], Bdiff[j], Bskew[j]};
            C[i][j] = dot3(coeffs, vecs);
        }
    }
}

void matmul_18op_hoisted_opt2(const float A[3][3], const float B[3][3], float C[3][3]) {
    const float* B0 = B[0];
    const float* B1 = B[1];
    const float* B2 = B[2];

    float SB[3], DB[3];
    for (int k = 0; k < 3; ++k) {
        SB[k] = B1[k] + B2[k];
        DB[k] = B1[k] - B2[k];
    }

    for (int i = 0; i < 3; ++i) {
        float a0 = A[i][0];
        float a1 = A[i][1];
        float a2 = A[i][2];
        float sa = a1 + a2;
        float da = a1 - a2;

        for (int k = 0; k < 3; ++k) {
            C[i][k] = a0 * B0[k] + 0.5f * (sa * SB[k] + da * DB[k]);
        }
    }
}

// ------------------------------------------------------------------
// Naive 3x3 baseline
// ------------------------------------------------------------------
void naive_matmul(const float A[3][3], const float B[3][3], float C[3][3]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < 3; ++k)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
}

// ------------------------------------------------------------------
// Eigen-based 3x3 multiply
// ------------------------------------------------------------------
void eigen_matmul(const float A[3][3], const float B[3][3], float C[3][3]) {
    Eigen::Matrix3f Amat, Bmat, Cmat;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            Amat(i,j) = A[i][j];
            Bmat(i,j) = B[i][j];
        }
    Cmat = Amat * Bmat;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            C[i][j] = Cmat(i,j);
}

// ------------------------------------------------------------------
// Benchmark Utility
// ------------------------------------------------------------------
template<typename Func>
double benchmark(Func f, int trials = 1000000) {
    float A[3][3], B[3][3], C[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            A[i][j] = static_cast<float>(rand()) / RAND_MAX;
            B[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < trials; ++i) {
        f(A, B, C);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

// ------------------------------------------------------------------
// Main entry
// ------------------------------------------------------------------
int main() {
    srand(42);

    const int TRIALS = 1'000'000;
    
    double t_custom18 = benchmark(matmul_18op_hoisted_opt2, TRIALS);
    double t_custom23 = benchmark(matmul_23op, TRIALS);
    double t_naive  = benchmark(naive_matmul,  TRIALS);
    double t_eigen  = benchmark(eigen_matmul,  TRIALS);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "18-op kernel : " << t_custom18 << " sec\n";
    std::cout << "23-op kernel : " << t_custom23 << " sec\n";
    std::cout << "Naive triple : " << t_naive  << " sec\n";
    std::cout << "Eigen        : " << t_eigen  << " sec\n";
}
