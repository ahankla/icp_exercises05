# Introduction to Computational Physics
# Exercise 5
# Robin Greif, Lia Hankla


# 1.  Numerical linear algebra methods
#     Consider the following matrix equation
#       |  e   1/2 |  | x |  =  | 1/2 |
#       | 1/2  1/2 |  | y |  =  | 1/4 |
#     e = 10^-6  or  10^-12
import numpy as np
e = 10**(-12)
left_matrix = np.array([[e, 1/2], [1/2, 1/2]])
right_matrix = np.array([1/2, 1/4])

# check for existence of solution
if left_matrix.shape[0] == left_matrix.shape[1]:
    if np.linalg.det(left_matrix) == 0:
        print("Degeneracy. Solutions not unique.")
    else:
        pass
elif left_matrix.shape[0] > left_matrix.shape[1]:
    print("More equations than unknown variables. System overdetermined.")
    print("Attempt to find approximation.")
elif left_matrix.shape[0] < left_matrix.shape[1]:
    print("Less equations than unknown variables. System underdetermined.")
    print("")
print()

# a) Solve the above system numerically by hand (or write a small program that does this) using:
#    (i) either the Gauß-Jordan method,  or the Gaussian elimination and backsubstitution technique (your choice) 
#        but without pivoting. ϵ = 10−6 (single precision),  ϵ = 10−12 (double precision)


#   (ii) Check the result by back-substituting (x,y), check you get the correct right-hand-side, i.e. (1/2,1/4).
#  (iii) Repeat with row-wise pivoting. 
#   (iv) What do you notice, compared to the previous attempt? 
#    (v) How small can you make ϵ without running into precision problems? 

# b) (i) Solve the above equations using the Numerical Recipes routines for LU-decomposition and back-substitution 
#        
# from a library of your choice
#   (ii) Check if the same results are obtained.

# c) This matrix is symmetric. But it also works for non-symmetric matrices. 
#    Try one out, and check that you use the correct order of indices for the matrix: 
#    Some programming languages use (row,column) order while others use (column,row) order, 
#    and it can also depend on the implementation of the linear algebra software.


# 2. Tridiagonal matrices (homework)



