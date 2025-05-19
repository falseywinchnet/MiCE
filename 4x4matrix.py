#copyright rainstar 2025
# Optimized 4x4 matrix multiplication using hoisted symmetry (generalized from 3x3 approach)
#copyright joshuah.rainstar@gmail.com 2025
@njit
def matmul_4x4_fused_opt_scaled(A, B):
    C = np.zeros((4, 4))

    # Hoist and scale SB12, DB12 once
    SB12 = np.empty(4)
    DB12 = np.empty(4)
    for j in range(4):
        SB12[j] = 0.5 * (B[1, j] + B[2, j])
        DB12[j] = 0.5 * (B[1, j] - B[2, j])

    for i in range(4):
        a0 = A[i, 0]
        a1 = A[i, 1]
        a2 = A[i, 2]
        a3 = A[i, 3]

        sa12 = a1 + a2
        da12 = a1 - a2

        for j in range(4):
            b0 = B[0, j]
            b3 = B[3, j]
            sb12 = SB12[j]
            db12 = DB12[j]

            C[i, j] = a0 * b0 + a3 * b3
            C[i, j] += sa12 * sb12 + da12 * db12

    return C

# Evaluate accuracy and timing
C_scaled = matmul_4x4_fused_opt_scaled(A, B)
accuracy_error_scaled = np.max(np.abs(C_naive - C_scaled))

start_scaled = time.time()
for _ in range(10000):
    matmul_4x4_fused_opt_scaled(A, B)
end_scaled = time.time()

scaled_time = end_scaled - start_scaled

accuracy_error_scaled, scaled_time
