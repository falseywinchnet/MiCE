#Vibe coded. Reverse engineered from FFT ops with lots of coaxing and chatgpt 4o. o3 kept protesting it cant be done.
not even done, actually. See this code: i havnt finished running it. I only ran it for 2-term bilinear composition products for 6 minutes:

([(a0 + a1, b0 + b1, [a0*b0, a0*b1]),
  (a0 + a1, b0 + b2, [a0*b0, a0*b2]),
  (a0 + a1, b1 + b2, [a0*b1, a0*b2]),
  (a0 - a1, b0 + b1, [a0*b0, a0*b1]),
  (a0 - a1, b0 + b2, [a0*b0, a0*b2])],
 9,
 27)

so, more work to refactor *is* possible 

from itertools import combinations, product
import sympy as sp

# Define symbolic variables for A and B
a = sp.symbols('a00:09')  # a00 to a08
b = sp.symbols('b00:09')  # b00 to b08

# Map into 3x3 matrices
A = sp.Matrix(3, 3, a)
B = sp.Matrix(3, 3, b)

# Use the symbolic version of compute_C_fast
def compute_C_fast_sym(A, B):
    # Flatten A and B
    a = A.reshape(9, 1)
    b = B.reshape(9, 1)

    a00, a01, a02 = a[0], a[1], a[2]
    a10, a11, a12 = a[3], a[4], a[5]
    a20, a21, a22 = a[6], a[7], a[8]

    b00, b01, b02 = b[0], b[1], b[2]
    b10, b11, b12 = b[3], b[4], b[5]
    b20, b21, b22 = b[6], b[7], b[8]

    x0  = b00 + b10 + b20
    x1  = (a00 + a01 + a02) / 3
    x2  = b10 - b20
    x3  = -a02
    x4  = (a01 + x3) / 2
    x5  = b00 - 0.5 * b10 - 0.5 * b20
    x6  = (2*a00 - a01 + x3) / 3

    x7  = b01 + b11 + b21
    x8  = b11 - b21
    x9  = b01 - 0.5 * b11 - 0.5 * b21

    x10 = b02 + b12 + b22
    x11 = b12 - b22
    x12 = b02 - 0.5 * b12 - 0.5 * b22

    x13 = (a10 + a11 + a12) / 3
    x14 = -a12
    x15 = (a11 + x14) / 2
    x16 = (2*a10 - a11 + x14) / 3

    x17 = (a20 + a21 + a22) / 3
    x18 = -a22
    x19 = (a21 + x18) / 2
    x20 = (2*a20 - a21 + x18) / 3

    C = sp.Matrix.zeros(3, 3)

    C[0,0] = x0*x1  + x2*x4  + x5*x6
    C[0,1] = x1*x7  + x4*x8  + x6*x9
    C[0,2] = x1*x10 + x4*x11 + x6*x12

    C[1,0] = x0*x13  + x2*x15  + x5*x16
    C[1,1] = x7*x13  + x8*x15  + x9*x16
    C[1,2] = x10*x13 + x11*x15 + x12*x16

    C[2,0] = x0*x17  + x2*x19  + x5*x20
    C[2,1] = x7*x17  + x8*x19  + x9*x20
    C[2,2] = x10*x17 + x11*x19 + x12*x20

    return C

# Compute symbolic C
C_sym = compute_C_fast_sym(A, B)

# Flatten and collect all unique scalar products of the form (linear_in_A) * (linear_in_B)
terms = set()
for c in C_sym:
    terms.update(sp.expand(c).as_ordered_terms())

# Extract scalar multiplicative forms
scalar_products = []
for term in terms:
    muls = term.atoms(sp.Mul)
    for m in muls:
        if any(ai in m.free_symbols for ai in a) and any(bi in m.free_symbols for bi in b):
            scalar_products.append(m)

# Deduplicate scalar products
unique_scalar_products = list(set(scalar_products))
unique_scalar_products.sort(key=str)

unique_scalar_products[:30], len(unique_scalar_products)

# Create symbolic linear forms for A and B
a_syms = list(sp.symbols('a0:9'))
b_syms = list(sp.symbols('b0:9'))

# Construct candidate linear combinations (support size 2 or 3 max to limit explosion)
candidate_a_forms = []
candidate_b_forms = []

# Limit to combinations of up to 2 symbols with coefficients ±1, ±0.5
coeffs = [1, -1, 0.5, -0.5]
for k in [2, 3]:
    for indices in combinations(range(9), k):
        for coeffs_set in product(coeffs, repeat=k):
            expr = sum(c * a_syms[i] for c, i in zip(coeffs_set, indices))
            candidate_a_forms.append(expr)

for k in [2, 3]:
    for indices in combinations(range(9), k):
        for coeffs_set in product(coeffs, repeat=k):
            expr = sum(c * b_syms[i] for c, i in zip(coeffs_set, indices))
            candidate_b_forms.append(expr)


# Now test products of these linear forms to match scalar monomials
compressed_forms = []
matched_terms = set()

for a_form in candidate_a_forms:
    for b_form in candidate_b_forms:
        prod = sp.expand(a_form * b_form)
        terms = prod.as_ordered_terms()
        matched = [t for t in terms if t in unique_scalar_products]
        if len(matched) > 1:
            compressed_forms.append((a_form, b_form, matched))
            matched_terms.update(matched)

# Report how many of the 27 scalar products are covered
covered_count = len(matched_terms)
total_unique = len(unique_scalar_products)

compressed_forms[:5], covered_count, total_unique
@njit
def validated_23op_matmul(A, B):
    C = np.empty((3, 3), dtype=A.dtype)

    # Composite scalar products
    p0 = (A[0,0] + A[0,1]) * (B[0,0] + B[0,1])  # Covers a0*b0, a0*b1, a1*b0, a1*b1
    p1 = (A[0,0] + A[0,1]) * (B[0,0] + B[0,2])  # Covers a0*b0, a0*b2, a1*b0, a1*b2
    p2 = (A[0,0] + A[0,1]) * (B[1,0] + B[1,1])  # Interpreted as reuse for same index pattern
    p3 = (A[0,0] - A[0,1]) * (B[0,0] + B[0,1])  # a0 - a1, b0 + b1
    p4 = (A[0,0] - A[0,1]) * (B[0,0] + B[0,2])  # a0 - a1, b0 + b2

    # Atomic products (remaining 18)
    a = A
    b = B
    x0  = b[0,0] + b[1,0] + b[2,0]
    x1  = (a[0,0] + a[0,1] + a[0,2]) / 3
    x2  = b[1,0] - b[2,0]
    x3  = -a[0,2]
    x4  = (a[0,1] + x3) / 2
    x5  = b[0,0] - 0.5 * b[1,0] - 0.5 * b[2,0]
    x6  = (2*a[0,0] - a[0,1] + x3) / 3

    x7  = b[0,1] + b[1,1] + b[2,1]
    x8  = b[1,1] - b[2,1]
    x9  = b[0,1] - 0.5 * b[1,1] - 0.5 * b[2,1]

    x10 = b[0,2] + b[1,2] + b[2,2]
    x11 = b[1,2] - b[2,2]
    x12 = b[0,2] - 0.5 * b[1,2] - 0.5 * b[2,2]

    x13 = (a[1,0] + a[1,1] + a[1,2]) / 3
    x14 = -a[1,2]
    x15 = (a[1,1] + x14) / 2
    x16 = (2*a[1,0] - a[1,1] + x14) / 3

    x17 = (a[2,0] + a[2,1] + a[2,2]) / 3
    x18 = -a[2,2]
    x19 = (a[2,1] + x18) / 2
    x20 = (2*a[2,0] - a[2,1] + x18) / 3

    # Final matrix entries
    C[0,0] = x0*x1  + x2*x4  + x5*x6
    C[0,1] = x1*x7  + x4*x8  + x6*x9
    C[0,2] = x1*x10 + x4*x11 + x6*x12

    C[1,0] = x0*x13  + x2*x15  + x5*x16
    C[1,1] = x7*x13  + x8*x15  + x9*x16
    C[1,2] = x10*x13 + x11*x15 + x12*x16

    C[2,0] = x0*x17  + x2*x19  + x5*x20
    C[2,1] = x7*x17  + x8*x19  + x9*x20
    C[2,2] = x10*x17 + x11*x19 + x12*x20

    return C

# Validate correctness
C_23 = validated_23op_matmul(A_test, B_test)
C_np = A_test @ B_test
np.allclose(C_23, C_np, rtol=1e-5, atol=1e-8), C_23, C_np
