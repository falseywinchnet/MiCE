#copyright rainstar 2025
# Optimized 4x4 matrix multiplication using hoisted symmetry (generalized from 3x3 approach)
@njit
def matmul_4x4_fused_opt(A, B):
    C = np.zeros((4, 4))

    # Hoist symmetric terms for (1,2) mirror pairs across columns
    SB12 = np.empty(4)
    DB12 = np.empty(4)
    for j in range(4):
        SB12[j] = B[1, j] + B[2, j]
        DB12[j] = B[1, j] - B[2, j]

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
            C[i, j] += 0.5 * (sa12 * sb12 + da12 * db12)

    return C

# Evaluate accuracy and timing
C_fused_opt = matmul_4x4_fused_opt(A, B)
accuracy_error_fused_opt = np.max(np.abs(C_naive - C_fused_opt))

start_fused_opt = time.time()
for _ in range(10000):
    matmul_4x4_fused_opt(A, B)
end_fused_opt = time.time()

fused_opt_time = end_fused_opt - start_fused_opt

accuracy_error_fused_opt, fused_opt_time
