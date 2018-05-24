import numpy as np


# ------------------------------
# Exercise 2.1: Gaussian Elimination
# ------------------------------
def gaussian_elimination(a, b, c, y):
    """Given coefficients in an NxN tridiagonal matrix,
       reduce to echelon form.
       a, b, c, y: Nx1 arrays such that
       a_i x_{i-1} + b_i x_i + c_i x_{i+1} = y_i
       Note that a_1 and c_N should be 0.
       """
    N = a.size

    for i in range(1, N):
        y[i] = b[i - 1] * y[i] - a[i] * y[i - 1]
        b[i] = b[i - 1] * b[i] - a[i] * c[i - 1]
        c[i] = c[i] * b[i - 1]
        a[i] = 0
    return b, c, y


# ------------------------------
#  Exercise 2.2: Backwards Substitution
#  ------------------------------
def backwards_substitution(b, c, y):
    """Given coefficients in an NxN echelon form matrix,
    use backwards substitution to solve for x.
    All inputs are Nx1 arrays such that
    b_i x_i + c_i x_{i+1} = y_i.
    Note that c_N should be 0.
    """
    N = b.size
    x = np.zeros(N + 1)  # For last row
    for i in range(N - 1, -1, -1):
        x[i] = (y[i] - c[i] * x[i + 1]) / b[i]
    return x[:-1]


# --------------------------------
#  Exercise 2.3: Gaussian Solve
# -------------------------------
def gaussian_solve(a, b, c, y):
    """Given coefficients in an NxN tridiagonal matrix,
        use gaussian elimination and backwards substitution
        to solve for x.
        All inputs are Nx1 arrays such that
        a_i x_{i-1} + b_i x_i + c_i x_{i+1} = y_i.
        Note that a_0, c_N should be 0.
        """
    b, c, y = gaussian_elimination(a, b, c, y)
    x = backwards_substitution(b, c, y)
    return x


# -----------------------
# Exercise 1.4: Example
# ----------------------

# Set-up
N = 10
a = np.ones(N) * -1.;
a[0] = 0
b = np.ones(N) * 2.
c = np.ones(N) * -1.;
c[-1] = 0
y = np.ones(N) * 0.1

# Solve
x = gaussian_solve(a, b, c, y)  # Note that a, b, c, y are modified to echelon form
print("Solution: x = {}".format(x))

# -----------------------
# Exercise 1.5: Check
# ----------------------
x = np.append(x, 0)  # for last row
for i in range(0, N):
    if i == 0:
        y_predicted = 2. * x[i] - 1. * x[i + 1]
    else:
        y_predicted = -1. * x[i - 1] + 2. * x[i] - 1. * x[i + 1]
    print("predicted: {}. Actual: 0.1. Difference: {}. Relative error: {}".format(
        y_predicted, 0.1 - y_predicted, np.abs(0.1 - y_predicted) / 0.1))
