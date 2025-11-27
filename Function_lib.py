# Comprehensive Numerical Methods Function Library
# Name: Roshan Yadav
# Roll No: 2311144

import math
import copy
import numpy as np
import matplotlib.pyplot as plt


# ==================== ASSIGNMENT 1: Random Number Generators ====================

def pRNG(s=0.123456789, c=3.9, n=10000):
    '''This function is a Pseudo random number generator 
    which uses equation x(i+1) = c*x(i)*(1-x(i)), 
    where x(0)=s and c are given as input.
    Returns a list of n random numbers'''

    L = []
    L.append(s)  # x(0) = s
    
    for i in range(n-1):
        t = c * L[i] * (1 - L[i])  # x(i+1) = c*x(i)*(1-x(i))
        L.append(t)
    
    return L


def LCG(a=1103515245, c=12345, m=32768, s=10, n=10000):
    '''This function is a Linear Congruential Generator (LCG) 
    which generates pseudo random numbers using the formula:
    x(i+1) = (a*x(i) + c) mod m, 
    where x(0)=s, a, c, m are given as inputs.
    Returns a list of n random numbers'''

    L = []
    L.append(s)  # x(0) = s
    
    for i in range(n-1):
        t = (a * L[i] + c) % m  # x(i+1) = (a*x(i) + c) mod m
        t = t/m  # Normalize to [0, 1)
        L.append(t)
    
    return L


# ==================== FILE I/O FUNCTIONS ====================

def read_matrix(filename):
    '''Read matrix from a file'''
    with open(filename, 'r') as f:
        matrix = []
        for line in f:
            # Convert each line into a list of floats
            row = [float(num) for num in line.strip().split()]
            matrix.append(row)
    return matrix


def read_vector(filename):
    '''Read a vector from a file (one number per line)'''
    with open(filename, 'r') as f:
        vector = []
        for line in f:
            vector.append(float(line.strip()))
    return vector


# ==================== ASSIGNMENT 2: Matrix Class & Gauss-Jordan ====================

class matrix():
    def __init__(self, data=None):
        self.m = len(data)
        self.n = len(data[0])
        self.data = data
    
    def display(self):
        print(self.data)
        return None
    
    def row_swap(self, u=None, p=None):
        self.data[u], self.data[p] = self.data[p], self.data[u]
    
    def con_mul(self, c=0, p=0):
        '''This function does the 
        constant 'c' multiplication 
        to a row no 'p'.'''
        for i in range(len(self.data[p])):
            self.data[p][i] = float(c) * float(self.data[p][i])
    
    def row_ops(self, f=None, r=None, c=None):
        '''This function does the following row operation:
        Row[f] ->Row[f] + c*Row[r]'''
        for i in range(len(self.data[f])):
            self.data[f][i] = float(self.data[f][i]) + float(c) * float(self.data[r][i])
    
    def aug(self, b=None):
        # Create a deep copy to avoid modifying original data
        A = copy.deepcopy(self.data)
        # Handle both 1D and 2D b vectors
        for i in range(len(A)):
            if isinstance(b[0], list):
                A[i].append(b[i][0])  # b is 2D, take first element
            else:
                A[i].append(b[i])     # b is 1D
        return A
    
    def zero(self, i=0):
        if i == 0:
            return 0
        else:
            return float(1/i)
    
    def gauss_jordan(self, b):
        """
        Gauss-Jordan elimination with partial pivoting to solve Ax = b
        
        Parameters:
        b: right-hand side vector (list)
        
        Returns:
        solution vector x (list)
        """
        
        # Create augmented matrix [A|b]
        A_data = []
        for i in range(self.m):
            # Copy row from A and append b[i]
            row = self.data[i][:] + [b[i]]
            A_data.append(row)

        A = matrix(A_data)
        m = self.m
        n = self.n
        
        # Forward elimination with partial pivoting
        for i in range(min(m, n)):
            # Find pivot (largest absolute value in column i from row i onwards)
            max_row = i
            for k in range(i + 1, m):
                if abs(A.data[k][i]) > abs(A.data[max_row][i]):
                    max_row = k
            
            # Swap rows if needed
            if max_row != i:
                A.row_swap(i, max_row)
            
            # Check if pivot is zero (singular matrix)
            if abs(A.data[i][i]) < 1e-10:
                print(f"Matrix is singular at column {i}")
                return None
            
            # Make diagonal element 1 (normalize pivot row)
            pivot = A.data[i][i]
            A.con_mul(c=1.0/pivot, p=i)
            
            # Make all elements in column i (below pivot) zero
            for j in range(i + 1, m):
                if abs(A.data[j][i]) > 1e-10:
                    multiplier = -A.data[j][i]
                    A.row_ops(j, i, multiplier)
        
        # Back substitution (Jordan part)
        # Make all elements above the diagonal zero
        for i in range(min(m, n) - 1, -1, -1):
            for j in range(i):
                if abs(A.data[j][i]) > 1e-10:
                    multiplier = -A.data[j][i]
                    A.row_ops(j, i, multiplier)
        
        # Extract solution vector from the augmented matrix
        # The solution is in column n (the first column after the original matrix A)
        solution = []
        for i in range(m):
            solution.append(A.data[i][n])
        
        return solution
        
    
    # ==================== ASSIGNMENT 3: LU Decomposition ====================
    
    def LU_decomposition(self):
        """
        Returns LU decomposition of matrix using Doolittle method.
        Modifies the matrix in-place to store both L and U.
        """
        A = copy.deepcopy(self.data)
        n = len(A)
        
        # Doolittle LU Decomposition Algorithm
        for i in range(n):
            # Calculate U matrix elements (upper triangular)
            for k in range(i, n):
                sum_val = 0
                for j in range(i):
                    sum_val += A[i][j] * A[j][k]
                A[i][k] = A[i][k] - sum_val
            
            # Calculate L matrix elements (lower triangular)
            for k in range(i+1, n):
                sum_val = 0
                for j in range(i):
                    sum_val += A[k][j] * A[j][i]
                A[k][i] = (A[k][i] - sum_val) / A[i][i]
        
        return A
    
    def solve_LU(self, b):
        """
        Solves system of linear equations using LU decomposition
        with forward-backward substitution.
        """
        n = len(self.data)
        
        # Step 1: Get LU decomposition
        LU_combined = self.LU_decomposition()
        
        # Step 2: Extract L and U matrices from combined LU
        L = []
        U = []
        for i in range(n):
            L_row = []
            U_row = []
            for j in range(n):
                if i == j:
                    L_row.append(1.0)  # Diagonal of L is 1 (Doolittle)
                    U_row.append(LU_combined[i][j])
                elif i > j:
                    L_row.append(LU_combined[i][j])  # Lower triangular
                    U_row.append(0.0)
                else:
                    L_row.append(0.0)
                    U_row.append(LU_combined[i][j])  # Upper triangular
            L.append(L_row)
            U.append(U_row)
        
        # Step 3: Forward Substitution - Solve Ly = b
        y = []
        for i in range(n):
            sum_val = 0
            for j in range(i):
                sum_val += L[i][j] * y[j]
            y.append(b[i] - sum_val)
        
        # Step 4: Backward Substitution - Solve Ux = y
        x = [0.0] * n
        for i in range(n-1, -1, -1):
            sum_val = 0
            for j in range(i+1, n):
                sum_val += U[i][j] * x[j]
            x[i] = (y[i] - sum_val) / U[i][i]
        
        return x
    
    # ==================== ASSIGNMENT 4 & 5: Cholesky, Jacobi, Gauss-Seidel ====================

    # ==================== Cholesky Decomposition ====================
    
    def check_symmetric(self):
        """Check if matrix is symmetric"""
        n = len(self.data)
        for i in range(n):
            for j in range(n):
                if abs(self.data[i][j] - self.data[j][i]) > 1e-10:
                    return False
        return True
    
    def cholesky(self, b):
        """
        Solves system of linear equations using Cholesky decomposition
        with forward-backward substitution.
        Returns: (solution_vector, L_matrix) or (None, None) if not symmetric
        """
        n = len(self.data)
        
        # Step 0: Check if matrix is symmetric
        if not self.check_symmetric():
            print('Error')
            print('\nMatrix is not Symmetric')
            return None, None
        
        # Step 1: Initialize L matrix
        L = [[0.0 for _ in range(n)] for _ in range(n)]
        
        # Step 2: Cholesky decomposition A = L * L^T
        for i in range(n):
            for j in range(i + 1):  # Only lower triangular part
                if i == j:  # Diagonal elements
                    sum_sq = sum(L[i][k] ** 2 for k in range(j))
                    value = self.data[i][i] - sum_sq
                    if value <= 0:
                        raise ValueError(f"Matrix not positive definite at ({i},{i}). Value: {value}")
                    L[i][j] = math.sqrt(value)
                else:  # Below diagonal elements
                    sum_prod = sum(L[i][k] * L[j][k] for k in range(j))
                    L[i][j] = (self.data[i][j] - sum_prod) / L[j][j]
        
        # Step 3: Forward Substitution - Solve Ly = b
        y = [0.0] * n
        for i in range(n):
            sum_val = sum(L[i][j] * y[j] for j in range(i))
            y[i] = (b[i] - sum_val) / L[i][i]
        
        # Step 4: Backward Substitution - Solve L^T * x = y
        x = [0.0] * n
        for i in range(n-1, -1, -1):
            sum_val = sum(L[j][i] * x[j] for j in range(i+1, n))
            x[i] = (y[i] - sum_val) / L[i][i]
        
        return x, L
    
    # ==================== Iterative Methods ====================

    # ==================== Jacobi ====================
    
    def jacobi(self, b, x0=None, tol=1e-6, max_iter=1000):
        """
        Solve Ax = b using Jacobi method with a given maximum iter as max_iter.
        Returns: (solution_vector, iterations)
        """
        # Make a copy to avoid modifying original matrix
        A = copy.deepcopy(self.data)
        b_copy = b[:]
        n = len(b)
        
        # Step 0: Check if any diagonal element of A is zero. If yes then swap the rows.
        for i in range(n):
            if A[i][i] == 0:
                # Find a row with non-zero element in column i
                for k in range(i+1, n):
                    if A[k][i] != 0:
                        # Swap rows in matrix A
                        A[i], A[k] = A[k], A[i]
                        # Swap corresponding elements in vector b
                        b_copy[i], b_copy[k] = b_copy[k], b_copy[i]
                        break
        
        # Step 1: Initialize guess vector x
        if x0 is None:
            x0 = [0.0] * n  # start with zeros

        x = x0[:]  # make a copy

        for k in range(max_iter):
            x_new = [0.0] * n  # to store new values

            # Step 2: Compute each x[i] using previous iteration values
            for i in range(n):
                # Calculate sum of a_ij * x_j for j != i
                sum_terms = 0.0
                for j in range(n):
                    if j != i:
                        sum_terms += A[i][j] * x[j]
                # Update x_new[i] using Jacobi formula
                x_new[i] = (b_copy[i] - sum_terms) / A[i][i]

            # Step 3: Compute maximum difference for convergence check
            diff = max(abs(x_new[i] - x[i]) for i in range(n))

            if diff < tol:
                print(f"Jacobi converged in {k+1} iterations")
                return x_new, k + 1

            # Step 4: Prepare for next iteration
            x = x_new

        print("Max iteration done")
        return x, max_iter
    
    # ==================== Gauss Seidal ====================

    def gauss_seidel(self, b, x0=None, max_iter=50, tol=1e-6):
        """
        Gauss-Seidel Method for solving Ax = b
        Returns: solution_vector
        """
        # Make a copy to avoid modifying original matrix
        A = copy.deepcopy(self.data)
        b_copy = b[:]
        n = len(b)

        # Check if any diagonal element of A is zero. If yes then swap the rows.
        for i in range(n):
            if A[i][i] == 0:
                # Find a row with non-zero element in column i
                for k in range(i+1, n):
                    if A[k][i] != 0:
                        # Swap rows in matrix A
                        A[i], A[k] = A[k], A[i]
                        # Swap corresponding elements in vector b
                        b_copy[i], b_copy[k] = b_copy[k], b_copy[i]
                        break

        # Step 0: Initialize guess vector x
        if x0 is None:
            x0 = [0.0] * n  # start with zeros

        x = x0[:]  # make a copy
        
        for iteration in range(max_iter):
            x_old = x[:]  # X(K)
            
            # Step 1: Update variables ONE BY ONE
            for i in range(n):
                # Step 2: Calculate sum using UPDATED values (j < i) and OLD values (j > i)
                sum_ax = 0.0
                for j in range(n):
                    if i != j:
                        sum_ax += A[i][j] * x[j]  # Uses NEW x[j] if j < i, OLD x[j] if j > i
                
                # Step 3: Apply Gauss-Seidel formula and update X(K+1)
                x[i] = (b_copy[i] - sum_ax) / A[i][i]
                    
            # Step 4: Check convergence
            converged = True
            for i in range(n):
                if abs(x[i] - x_old[i]) > tol:
                    converged = False
                    break
            
            if converged:
                print(f"Converged in {iteration + 1} iterations")
                return x
        
        return x
    
    # ==================== Matrix Operations ====================
    
    def mat_mul(self, B):
        """
        Matrix multiplication: A * B
        Input: B is another matrix object
        Returns: matrix object containing result of A * B
        """
        # Check if multiplication is possible
        if self.n != B.m:
            print(f"Error: Cannot multiply {self.m}x{self.n} matrix with {B.m}x{B.n} matrix")
            return None
        
        # Initialize result matrix with zeros
        result = [[0.0 for _ in range(B.n)] for _ in range(self.m)]
        
        # Perform matrix multiplication
        for i in range(self.m):
            for j in range(B.n):
                for k in range(self.n):
                    result[i][j] += self.data[i][k] * B.data[k][j]
        
        return matrix(result)
    
    def mat_vec_mul(self, b):
        """
        Matrix-vector multiplication: A * b
        Input: b is a vector (list)
        Returns: resulting vector (list)
        """
        # Check if multiplication is possible
        if self.n != len(b):
            print(f"Error: Cannot multiply {self.m}x{self.n} matrix with vector of length {len(b)}")
            return None
        
        # Initialize result vector
        result = [0.0 for _ in range(self.m)]
        
        # Perform matrix-vector multiplication
        for i in range(self.m):
            for j in range(self.n):
                result[i] += self.data[i][j] * b[j]
        
        return result
    
    def inverse_gauss_jordan(self):
        """
        Calculate inverse of matrix using Gauss-Jordan elimination
        Method: [A | I] -> [I | A^(-1)]
        Returns: inverse matrix as a list of lists
        """
        if self.m != self.n:
            print("Error: Matrix must be square to find inverse")
            return None
        
        # Use gauss_jorden method with None to get inverse
        inverse = self.gauss_jorden(b=None)
        
        return inverse
    
    def inverse_LU(self):
        """
        Calculate inverse of matrix using LU decomposition
        Method: Solve A*X = I column by column using LU decomposition
        Returns: inverse matrix as a list of lists
        """
        if self.m != self.n:
            print("Error: Matrix must be square to find inverse")
            return None
        
        n = self.m
        
        # Create identity matrix
        identity = [[0.0] * n for _ in range(n)]
        for i in range(n):
            identity[i][i] = 1.0
        
        # Get LU decomposition once
        LU_combined = self.LU_decomposition()
        
        # Extract L and U matrices
        L = []
        U = []
        for i in range(n):
            L_row = []
            U_row = []
            for j in range(n):
                if i == j:
                    L_row.append(1.0)  # Diagonal of L is 1 (Doolittle)
                    U_row.append(LU_combined[i][j])
                elif i > j:
                    L_row.append(LU_combined[i][j])  # Lower triangular
                    U_row.append(0.0)
                else:
                    L_row.append(0.0)
                    U_row.append(LU_combined[i][j])  # Upper triangular
            L.append(L_row)
            U.append(U_row)
        
        # Initialize inverse matrix
        inverse = [[0.0] * n for _ in range(n)]
        
        # Solve for each column of the inverse
        for col in range(n):
            # Get the col-th column of identity matrix
            b = [identity[i][col] for i in range(n)]
            
            # Forward Substitution - Solve Ly = b
            y = []
            for i in range(n):
                sum_val = 0
                for j in range(i):
                    sum_val += L[i][j] * y[j]
                y.append(b[i] - sum_val)
            
            # Backward Substitution - Solve Ux = y
            x = [0.0] * n
            for i in range(n-1, -1, -1):
                sum_val = 0
                for j in range(i+1, n):
                    sum_val += U[i][j] * x[j]
                x[i] = (y[i] - sum_val) / U[i][i]
            
            # Store this column in the inverse matrix
            for i in range(n):
                inverse[i][col] = x[i]
        
        return inverse
    
    def determinant(self):
        """
        Calculate determinant using LU decomposition
        det(A) = det(L) * det(U) = product of diagonal elements of U
        (since diagonal of L is all 1s in Doolittle method)
        Returns: determinant value
        """
        if self.m != self.n:
            print("Error: Matrix must be square to find determinant")
            return None
        
        n = self.m
        
        # Get LU decomposition
        LU_combined = self.LU_decomposition()
        
        # Calculate determinant as product of diagonal elements of U
        det = 1.0
        for i in range(n):
            det *= LU_combined[i][i]
        
        return det


# ==================== ASSIGNMENT 3: LU Decomposition ====================

def LU_decomposition(A):
    """
    This function returns LU decomposition of matrix A using Doolittle method.
    """
    n = len(A)
    
    # Doolittle LU Decomposition Algorithm
    for i in range(n):
        # Calculate U matrix elements (upper triangular)
        for k in range(i, n):
            sum_val = 0
            for j in range(i):
                sum_val += A[i][j] * A[j][k]
            A[i][k] = A[i][k] - sum_val
        
        # Calculate L matrix elements (lower triangular)
        for k in range(i+1, n):
            sum_val = 0
            for j in range(i):
                sum_val += A[k][j] * A[j][i]
            A[k][i] = (A[k][i] - sum_val) / A[i][i]
    
    return A


def solve_linear_equations(A, b):
    """
    This function solves system of linear equations using LU decomposition
    with forward-backward substitution.
    """
    n = len(A)
    
    # Step 1: Get LU decomposition
    LU_combined = LU_decomposition(A)
    
    # Step 2: Extract L and U matrices from combined LU
    L = []
    U = []
    for i in range(n):
        L_row = []
        U_row = []
        for j in range(n):
            if i == j:
                L_row.append(1.0)  # Diagonal of L is 1 (Doolittle)
                U_row.append(LU_combined[i][j])
            elif i > j:
                L_row.append(LU_combined[i][j])  # Lower triangular
                U_row.append(0.0)
            else:
                L_row.append(0.0)
                U_row.append(LU_combined[i][j])  # Upper triangular
        L.append(L_row)
        U.append(U_row)
    
    # Step 3: Forward Substitution - Solve Ly = b
    y = []
    for i in range(n):
        sum_val = 0
        for j in range(i):
            sum_val += L[i][j] * y[j]
        y.append(b[i] - sum_val)
    
    # Step 4: Backward Substitution - Solve Ux = y
    x = [0.0] * n
    for i in range(n-1, -1, -1):
        sum_val = 0
        for j in range(i+1, n):
            sum_val += U[i][j] * x[j]
        x[i] = (y[i] - sum_val) / U[i][i]
    
    return x


# ==================== ASSIGNMENT 4 & 5: Cholesky, Jacobi, Gauss-Seidel ====================

def check_sym(A):
    """Check if matrix A is symmetric"""
    n = len(A)
    for i in range(n):
        for j in range(n):
            if A[i][j] != A[j][i]:
                return False
    return True


def cholesky_solve(A, b):
    """
    This function solves system of linear equations using Cholesky decomposition
    with forward-backward substitution.
    """
    n = len(A)
    
    # Step 0: Check if A is symmetric
    if check_sym(A) == False:
        print('Error')
        print('\nMatrix is not Symmetric')
        return None
    
    # Step 1: Initialize L matrix
    L = [[0.0 for _ in range(n)] for _ in range(n)]
    
    # Step 2: Cholesky decomposition A = L * L^T
    for i in range(n):
        for j in range(i + 1):  # Only lower triangular part
            if i == j:  # Diagonal elements
                sum_sq = sum(L[i][k] ** 2 for k in range(j))
                value = A[i][i] - sum_sq
                if value <= 0:
                    raise ValueError(f"Matrix not positive definite at ({i},{i}). Value: {value}")
                L[i][j] = math.sqrt(value)
            else:  # Below diagonal elements
                sum_prod = sum(L[i][k] * L[j][k] for k in range(j))
                L[i][j] = (A[i][j] - sum_prod) / L[j][j]
    
    # Step 3: Forward Substitution - Solve Ly = b
    y = [0.0] * n
    for i in range(n):
        sum_val = sum(L[i][j] * y[j] for j in range(i))
        y[i] = (b[i] - sum_val) / L[i][i]
    
    # Step 4: Backward Substitution - Solve L^T * x = y
    x = [0.0] * n
    for i in range(n-1, -1, -1):
        sum_val = sum(L[j][i] * x[j] for j in range(i+1, n))
        x[i] = (y[i] - sum_val) / L[i][i]
    
    return x, L


def jacobi(A, b, x0=None, tol=1e-6, max_iter=1000):
    """
    Solve Ax = b using Jacobi method with a given maximum iter as max_iter.
    """
    # Number of equations
    n = len(b)
    
    # Step 0: Check if any diagonal element of A is zero. If yes then swap the rows.
    for i in range(n):
        if A[i][i] == 0:
            # Find a row with non-zero element in column i
            for k in range(i+1, n):
                if A[k][i] != 0:
                    # Swap rows in matrix A
                    A[i], A[k] = A[k], A[i]
                    # Swap corresponding elements in vector b
                    b[i], b[k] = b[k], b[i]
                    break
    
    # Step 1: Initialize guess vector x
    if x0 is None:
        x0 = [0.0] * n  # start with zeros

    x = x0[:]  # make a copy

    for k in range(max_iter):
        x_new = [0.0] * n  # to store new values

        # Step 2: Compute each x[i] using previous iteration values
        for i in range(n):
            # Calculate sum of a_ij * x_j for j != i
            sum_terms = 0.0
            for j in range(n):
                if j != i:
                    sum_terms += A[i][j] * x[j]
            # Update x_new[i] using Jacobi formula
            x_new[i] = (b[i] - sum_terms) / A[i][i]

        # Step 3: Compute maximum difference for convergence check
        diff = max(abs(x_new[i] - x[i]) for i in range(n))

        if diff < tol:
            print(f"Jacobi converged in {k+1} iterations")
            return x_new, k + 1

        # Step 4: Prepare for next iteration
        x = x_new

    print("Max iteration done")
    return x


def gauss_seidel(A, b, x0=None, max_iter=50, tol=1e-6):
    """
    Gauss-Seidel Method for solving Ax = b
    """
    n = len(b)

    # Check if any diagonal element of A is zero. If yes then swap the rows.
    for i in range(n):
        if A[i][i] == 0:
            # Find a row with non-zero element in column i
            for k in range(i+1, n):
                if A[k][i] != 0:
                    # Swap rows in matrix A
                    A[i], A[k] = A[k], A[i]
                    # Swap corresponding elements in vector b
                    b[i], b[k] = b[k], b[i]
                    break

    # Step 0: Initialize guess vector x
    if x0 is None:
        x0 = [0.0] * n  # start with zeros

    x = x0[:]  # make a copy
    
    for iteration in range(max_iter):
        x_old = x[:]  # X(K)
        
        # Step 1: Update variables ONE BY ONE
        for i in range(n):
            # Step 2: Calculate sum using UPDATED values (j < i) and OLD values (j > i)
            sum_ax = 0.0
            for j in range(n):
                if i != j:
                    sum_ax += A[i][j] * x[j]  # Uses NEW x[j] if j < i, OLD x[j] if j > i
            
            # Step 3: Apply Gauss-Seidel formula and update X(K+1)
            x[i] = (b[i] - sum_ax) / A[i][i]
                
        # Step 4: Check convergence
        converged = True
        for i in range(n):
            if abs(x[i] - x_old[i]) > tol:
                converged = False
                break
        
        if converged:
            print(f"Converged in {iteration + 1} iterations")
            return x
    
    return x


# ==================== ASSIGNMENT 6 & 7: Root Finding Methods ====================

def Bisection(a=1.5, b=3.0, t=0, iter=0, e=10**-6, d=10**-6):
    """This function utilises the bisection method 
    to find the roots of a monotonic function."""

    if f(a, t) * f(b, t) <= 0:
        if abs(b-a) < e:
            if abs(f(a, t)) < d:
                return a, iter
            if abs(f(b, t)) < d:
                return b, iter
    
        else:
            iter_new = iter + 1
            c = (a + b)/2
            if f(c, t) * f(a, t) < 0:
                return Bisection(a=a, b=c, t=t, iter=iter_new)
            if f(c, t) * f(b, t) < 0:
                return Bisection(a=c, b=b, t=t, iter=iter_new)
    else:
        return 'No root found in the given interval', iter


def regular_falsi(a, b, t=0, c=0, iter=0, e=10**-6, d=10**-6):
    """This function utilises the Regular Falsi method 
    to find roots of a monotonic function."""

    if f(a, t) * f(b, t) <= 0:
        if abs(b-a) < e:
            if abs(f(a, t)) < d:
                return a, iter
            if abs(f(b, t)) < d:
                return b, iter
        else:
            iter_new = iter + 1
            c = a
            c_new = (b - (((b - a)*f(b, t))/(f(b, t)-f(a, t))))
            if f(a, t) * f(c_new, t) <= 0:
                if abs(c_new - c) < e:
                    return c, iter
                else:
                    return regular_falsi(a=a, b=c_new, c=c_new, t=t, iter=iter_new)
            
            if f(b, t) * f(c, t) <= 0:
                if abs(c_new - c) < e:
                    return c, iter
                else:
                    return regular_falsi(a=c_new, b=b, c=c_new, t=t, iter=iter_new)
            
    else:
        return 'No root found in the given interval', iter


def bracketing(a, b, t=0, beta=0.2):
    """This function finds the bracket containing the root of the function."""

    if a > b:
        a, b = b, a

    if f(a, t) * f(b, t) <= 0:
        l = [a, b]
        return l
    
    else:
        if abs(f(a, t)) < abs(f(b, t)):
            a_new = a - beta*(b-a)
            return bracketing(a=a_new, b=b, t=0, beta=0.5)
        if abs(f(a, t)) >= abs(f(b, t)):
            b_new = b + beta*(b-a)
            return bracketing(a=a, b=b_new, t=0, beta=0.5)


def fixed_point(x, t=2, e=10**-6, max_iter=100):
    """This Function returns roots using Fixed Point Method"""
    
    for i in range(max_iter):
        # Iteration step: x_{n+1} = g(x_n)
        x_new = f(x, t)
        
        # Checking convergence
        if abs(x_new - x) < e:
            return x_new, i + 1  # Return i+1 for correct iteration count
            
        x = x_new

    return x


def newton_raphson(x=2, t=3, k=32, y0=2.5, x0=0, h=0.1, e=10**-6, d=10**-6, max_iter=1000):
    """Newton-Raphson method for equation"""
    for i in range(max_iter):
        G = f(x=x, t=t)
        dG = f(x=x, t=k)

        x_new=x + (G/dG)

        if abs(f(x=x_new,t=t)-f(x=x,t=t)) < e:
            return x_new
        
    if abs(x - x_new)<d:
        return x_new
    else: 
        return x_new
# ==================== ASSIGNMENT 8: Multivariable Root Finding ====================

def fixed_point_multi(X, g, e=10**-6, max_iter=100):
    """This function finds the roots of multivariable function 
    using Fixed point method"""
    
    for i in range(max_iter):
        Y = [0.0 for _ in range(len(X))]
        
        for j in range(len(X)):
            Y[j] = g(X, j)  # Iteration Step
        
        # Calculating relative error
        norm_diff = sum((Y[k] - X[k])**2 for k in range(len(X)))**0.5
        norm_Y = sum(Y[k]**2 for k in range(len(X)))**0.5
        
        if norm_diff / norm_Y < e:
            return Y, i + 1
        
        X = Y[:]  # Copy Y to X for next iteration
    
    return X, max_iter


def Newton_Raphson_multi(X, J_func, f_func, e=10**-6, max_iter=100):
    """This function finds the roots of multivariable function 
    using Newton-Raphson method"""
    
    for i in range(max_iter):
        # Jacobian Matrix
        J_matrix = J_func(X)
        
        # Inverse of Jacobian through Gauss Jorden method
        J = matrix(J_matrix)
        J_inv = J.gauss_jorden()
        
        # Calculate function values
        F = [f_func(X, k) for k in range(len(X))]
        
        # Newton-Raphson update: X_new = X - J^(-1) * F(X)
        Y = [0.0 for _ in range(len(X))]
        for j in range(len(X)):
            Total = 0
            for k in range(len(X)):
                Total = Total + J_inv[j][k] * F[k]
            Y[j] = X[j] - Total
        
        # Calculate relative error
        norm_diff = sum((Y[j] - X[j])**2 for j in range(len(X)))**0.5
        norm_Y = sum(Y[j]**2 for j in range(len(X)))**0.5
        
        if norm_diff / norm_Y < e:
            return Y, i + 1
        
        X = Y[:]
    
    return X


# ==================== ASSIGNMENT 9: Polynomial Root Finding ====================

class poly():
    """This Class defines a polynomial and 
    does various operations on it."""
    
    def __init__(self, L):
        """If P(x) = sum(i=n to 0)( (a_i) * (x^i)), 
        then L is list of coefficients i.e L= [a_i]
        for all i from n to 0 (in the descending order of power) 
        where n is the degree of the polynomial"""

        if isinstance(L, list):
            D = {}
            for i in range(len(L)):
                power = len(L) - 1 - i 
                D[power] = L[i]
        else:
            # Already a dictionary
            D = L.copy()

        self.coff = D  # For dictionary format: {power: coefficient}
        self.degree = max(D.keys()) if D else 0

    def comp(self, a):
        """ This function computes P(a). """
        N = self.coff
        val = 0

        for power, coeff in N.items():
            val = val + coeff * (a ** power)

        return val

    def poly_der(self, m):
        """ This Function returns the mth derivative 
        of the polynomial"""

        C = self.coff.copy()
        R = {}
        
        for power, coeff in C.items():
            if power < m:
                # This term disappears after m derivatives
                continue
            else:
                # Apply derivative m times
                new_coeff = coeff
                for j in range(m):
                    new_coeff = new_coeff * (power - j)
                R[power - m] = new_coeff
                
        return poly(R)

    def SDM(self, a):
        """ This function does synthetic division of P(x)
        by x-a where input 'a' is a root of P(x). """

        R = {}
        max_power = max(self.coff.keys())
        
        # Bring down first coefficient 
        R[max_power - 1] = self.coff[max_power]
        
        # Synthetic division loop 
        for i in range(max_power - 1, -1, -1):
            coeff = self.coff.get(i, 0)  # Get coefficient or 0
            if i > 0:  # No Negative powers
                R[i - 1] = coeff + a * R[i]
        
        return poly(R)

    def lag(self, a, e=10**-6, max_iter=100):
        """ This Function finds one root of polynomial through
        Laguerre's Method. The input 'a' is the initial guess 
        and e is the error tolerance. """

        for i in range(max_iter):
            if abs(self.comp(a)) < e:
                return a
                
            n = self.degree
            P_1 = self.poly_der(1)
            P_2 = self.poly_der(2)

            G = P_1.comp(a) / self.comp(a)
            H = G**2 - (P_2.comp(a) / self.comp(a))

            d_1 = (G + ((n-1) * (n*H - G**2))**0.5)
            d_2 = (G - ((n-1) * (n*H - G**2))**0.5)

            if abs(d_1) > abs(d_2):
                d = d_1
            else:
                d = d_2
            
            b = n / d
            a_new = a - b

            if abs(a_new - a) < e:
                return a_new
            
            a = a_new
        
        return a
            
    def root(self, a):
        """ This Function finds roots of polynomial through
        Laguerre's Method and do deflation through synthetic 
        division method. The input 'a' is the initial guess 
        and e is the error tolerance. """

        R = []
        n = self.degree
        P = poly(self.coff)

        for i in range(n):
            r = P.lag(a)
            R.append(r)
            P = P.SDM(r)
        
        return R


# ==================== ASSIGNMENT 10: Numerical integration Midpoint and Trapizoidal Method ====================
def midpoint_int(t,a,b,N):
    '''This does Numerical intergartion of f(x) from a to b
    using Midpoint method. 

    Formula used: 
    M_n= summation w(x_n)*f(x_n) from n=1 to n=N
    where w(x_n)=h for all n given h=(b-a)/2.+'''
    
    # Finding h
    h=(b-a)/N

    # Intialising the result
    r=0

    for i in range(1,N):
        # Finding x_n
        x=(2*a+i*h)/2

        # Summation step
        r=r+h*(f(t,x))
    
    return r

def trap_int(t=4,a=0,b=2,N=None):
    '''This does Numerical intergartion of f(x) from a to b
    using Trapezoidal method. 

    Formula used: 
    T_n= summation w(x_n)*f)x_n from n=1 to n=N
    where if n=0 or N, then  w(x_n) = h/2 
          else w(x_n)=h/2 given h= (b-a)/2'''
    
    # Finding h
    h=(b-a)/N

    # Intialising the result
    r=0

    for i in range(N+1):
        # Finding x_n
        x=a+i*h

        # Finding Weight function (w(x_n))
        if i==0 or i==N:
            w=h/2
        else:
            w=h
        
        # Summation step of w(x_n).f(x_n)
        r = r + (w*f(t=t,x=x))
    
    return r

# ==================== ASSIGNMENT 11: Numerical integration Simpson and Monte Carlo Method====================

def simpson_int(t,a,b,N):
    
    '''This does Numerical intergartion of f(x) from a to b
    using Simpson method. 

    Formula used: 
    S_n= summation w(x_n)*f(x_n) from n=0 to n=N
    where w(x_0)=w(x_N)=h/3, w(x_i)=2h/3 for all even i and w(x_j)=4h/3 for odd j given h=(b-a)/N.'''

    #Find h
    h=(b-a)/N

    #Intialising the result
    r=0

    # Summation loop for summation of w(x_n)*f(x_n) from n=0 to N
    for i in range(0,N+1):

        # Finding x_i
        x=a+i*h

        # Finding weight
        if i==0 or i==N:
            w=h/3
        elif i%2==0:
            w=2*h/3
        else:
            w=4*h/3
        
        # Summation step
        r=r+w*f(t,x)

    return r

'''
def pRNG(s=12345, n=100):

    L = []
    x = s
    a = 1664525
    c = 1013904223
    m = 2**32
    
    for i in range(n):
        x = (a * x + c) % m
        L.append(x / m)
    
    return L

'''

def monte_carlo_int(t, a, b, v, max_iter=30):

    """This Function does Monte Carlo integartion. 
    It gives the answer accurate upto 3 decimal places 
    as discussed in the class."""

    R = []
    M = []
    N = 100
    iter = 0
    seed = 12345
    
    sum_f = 0
    sum_f2 = 0
    total_samples = 0

    while iter < max_iter:
        iter = iter + 1
        
        seed = (seed * 1103515245 + 12345) % (2**31)
        
        L = pRNG(s=seed, n=N)
        
        for i in range(N):
            x = a + (b-a) * L[i]
            fx = f(t, x)
            
            sum_f = sum_f + fx
            sum_f2 = sum_f2 + fx**2
        
        total_samples = total_samples + N
        
        r = (b-a) * sum_f / total_samples
        k = sum_f2 / total_samples
        o = k - (sum_f / total_samples)**2
        
        R.append(r)
        M.append(total_samples)
        
        if len(R) >= 3 and abs(r - v) < 0.0001:
                break
        
        N = N + 500

    plt.figure(figsize=(8,6))
    plt.plot(M, R, marker='o', markersize=8, color='blue', label='F_N')
    plt.axhline(y=v, color='r', linestyle='--', linewidth=2, label=f'Target={v:.4f}')
    plt.xlabel("N")
    plt.ylabel("F_N")
    plt.title("Plot of F_n vs N for Monte Carlo Integration")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"Assgn_11_Monte_Carlo.png", dpi=300, bbox_inches='tight')
    plt.close()

    return R[-1]


# ==================== ASSIGNMENT 12: Numerical integration Gaussian Quadrature Method====================


def gauss_quad_int(k=1,a=-1,b=1,n=13):
    '''This does Numerical intergartion of f(x) from a to b
    using Gauss Quadrature. '''
    x,w=np.polynomial.legendre.leggauss(n)

    s=0
    t=[]
    for i in range(n):
        t=((b-a)/2)*x[i] +(b+a)/2
        s=s+(w[i])*(f(t,k))

    r= ((b-a)/2)*s
    return r

def find_opt_n(k,a=-1,b=1,val=2,e=10**-9,max_iter=1000,t=1):
    """ This function find N for Gauss Quadrature and Simpson"""

    if t==1: # the value of t lets choose wheather to find N for Simpson or not.

        # Finding n for for Simpson method
        m=2
        for j in range(max_iter):
            r=simpson_int(t=k,a=a,b=b,N=m)
            if abs(r-val) > e:
                m=m+20
                continue
            elif abs(r-val) < e:
                break
        
        # Finding n for for Gauss Quadrature
        n=2
        for i in range(max_iter):
            r = gauss_quad_int(k=k,a=a,b=b,n=n)
            if abs(r-val) > e:
                n=n+1
                continue
            elif abs(r-val) < e:
                break
        return n
    
    else:
        # Finding n for for Gauss Quadrature
        n=2
        for i in range(max_iter):
            r = gauss_quad_int(k=k,a=a,b=b,n=n)
            if abs(r-val) > e:
                n=n+1
                continue
            elif abs(r-val) < e:
                break
        return n,i


# ==================== ASSIGNMENT 13: Solving ODE though Forward Euler, Back ward Euler and Predector-Corrector====================

def forward_euler(t=1, y0=0, a=0, b=2, h=0.1):
    """Forward Euler method to solve ODE"""

    L = [] 
    L.append(y0)
    N = int((b - a) / h)
    x = a
    for i in range(0, N):
        x = x + h
        y_new = L[i] + h * (f(y=L[i], x=x, t=t))
        L.append(y_new)
    
    return L

      

def backward_euler(t=1, y0=0, a=0, b=2, h=0.1):
    """Backward Euler method to solve ODE"""
    L = [y0]
    N = int((b - a) / h)
    x = a
    for i in range(N):
        x_new = x + h
        
        # Initial guess
        m=L[i] + h*(f(y=L[i],x=x,t=t))
        # Newton-Raphson step
        y_new = newton_raphson(x=m, x0=x_new, y0=L[i], t=t, h=h)

        y_new= L[i] + h*(f(t=t,y=y_new,x=x_new))
        
        L.append(y_new)
        x = x_new
    return L


def predictor_corrector(t=1, a=0, b=2, y0=0, h=0.1):
    """Predictor-Corrector method to solve ODE"""
    L = []
    L.append(y0)
    N = int((b - a) / h)
    x = a
    for i in range(0, N):
        x_new = x + h
        # Predictor step
        k_1 = h * (f(y=L[i], x=x, t=t))
        y_p = L[i] + k_1
        # Corrector step
        k_2 = h * (f(y=y_p, x=x_new, t=t))
        r = L[i] + (k_1 + k_2) / 2
        L.append(r)
        x = x + h
    
    return L

# ==================== ASSIGNMENT 14: Solving ODE through RK4 and Simple Harmonic Oscillator (SHO)====================


def rk4(t=1,a=0,b=2,h=0.1,y0=1):
    N=int((b-a)/h)
    x=a
    Y=[]
    X=[]
    Y.append(y0)
    X.append(a)
    for i in range(0,N):
        y_n=Y[i]
        x=X[i]
        k_1= h*f(y=y_n,x=x,t=t)
        k_2=h*f(y=y_n + k_1/2, x= x + h/2, t=t)
        k_3= h*f(y=y_n + k_2/2,x=x+h/2,t=t)
        k_4=h*f(y=y_n + k_3,x=x+h,t=t)
        y_new= y_n + (k_1 + 2*k_2 +2*k_3 + k_4)/6
        Y.append(y_new)
        X.append(x+h)
    
    return X,Y

def SHO(a=0,b=40,h=0.01,x0=1,v0=0):
    u=0.15
    k=1.0
    m=1.0
    w=1.0
    w2=w**2
    N=int((b-a)/h)
    V=[]
    X=[]
    T=[]
    E=[]
    V.append(v0)
    X.append(x0)
    T.append(a)
    dt=h
    e=(1/2)*m*(v0**2) + (1/2)*k*(x0**2)
    E.append(e)
    
    for i in range(0,N):
        v=V[i]
        x=X[i]
        t=T[i]

        k_1x=dt*v
        k_1v=dt*(-u*(v) - (w2)* (x))

        k_2x=dt*(v + (k_1v)/2)
        k_2v=dt*(-u*(v +(k_1v)/2) - (w2)* (x + (k_1x)/2))

        k_3x=dt*(v + (k_2v)/2)
        k_3v=dt*(-u*(v +(k_2v)/2) - (w2)* (x + (k_2x)/2))

        k_4x=dt*(v+k_3v)
        k_4v=dt*((-u*(v +k_3v) - (w2)* (x + k_3x)))

        x= x+ (k_1x + 2*k_2x + 2*k_3x + k_4x)/6
        v= v+ (k_1v + 2*k_2v + 2*k_3v + k_4v)/6
        t=t+dt

        X.append(x)
        V.append(v)
        T.append(t)
        e=(1/2)*m*(v**2) + (1/2)*k*(x**2)
        E.append(e)

    return X,V,E,T
 

# ==================== ASSIGNMENT 15: Solving coupled ODES, Boundary value problems and Heat Equation====================

# Solving Boundary value problem using Shooting With RK4
def f(T):
    """
    Returns d²T/dx² = -α(Ta - T)
    where α = 0.01, Ta = 20
    """
    alpha = 0.01
    Ta = 20
    return -alpha * (Ta - T)


def RK4_2(k=5,a=0, b=10, t_i=0, t_f=None, h=0.01, v0=0):
    """
    RK4 integrator for the coupled ODEs:
    dy/dt = v
    dv/dt = f(x=y,t=k)
    """
    N = int((t_f - t_i) / h)
    V = []
    Y = []
    T = []
    V.append(v0)
    T.append(t_i)
    Y.append(a)
    dt = h
    
    for i in range(0, N):
        v = V[i]
        t = T[i]
        y = Y[i]

        k_1Y = dt * v
        k_1v = dt * f(x=t,t=k,v=v)

        k_2Y = dt * (v + k_1v / 2)
        k_2v = dt * f(t + k_1Y / 2,t=k,v=v)

        k_3Y = dt * (v + k_2v / 2)
        k_3v = dt * f(t + k_2Y / 2,t=k,v=v)

        k_4Y = dt * (v + k_3v)
        k_4v = dt * f(t + k_3Y,t=k,v=v)

        y_new = y + (k_1Y + 2*k_2Y + 2*k_3Y + k_4Y) / 6
        v_new = v + (k_1v + 2*k_2v + 2*k_3v + k_4v) / 6
        t_new = t + dt

        T.append(t_new)
        V.append(v_new)
        Y.append(y_new)

    return Y, V, T


def RK4_shooting(a=0, b=10, T_a=40, T_b=200, h=0.01, max_iter=100, tol=0.01):
    """
    Shooting method to solve boundary value problem.
    Uses Lagrange interpolation to refine the initial slope guess.
    """
    
    # Initial guesses for the slope at x=a
    zeta_l = -1.5  
    zeta_h = -0.5  
    
    # Get solutions with initial guesses
    X, V_l, T_l = RK4_2(a, b, T_a, T_b, h, zeta_l)
    X, V_h, T_h = RK4_2(a, b, T_a, T_b, h, zeta_h)
    
    # Ensure the solution is with in the bracket
    if (T_l[-1] - T_b) * (T_h[-1] - T_b) > 0:
        if T_l[-1] < T_b and T_h[-1] < T_b:
            zeta_h = 0.5
            X, V_h, T_h = RK4_2(a, b, T_a, T_b, h, zeta_h)
        elif T_l[-1] > T_b and T_h[-1] > T_b:
            zeta_l = -3.0
            X, V_l, T_l = RK4_2(a, b, T_a, T_b, h, zeta_l)
    
    # Iterate using Lagrange interpolation
    for iteration in range(max_iter):
        # Check convergence
        if abs(T_h[-1] - T_b) < tol:
            return X, V_h, T_h
        
        if abs(T_l[-1] - T_b) < tol:
            return X, V_l, T_l
        
        # Lagrange interpolation to get new guess
        zeta_new = zeta_l + (zeta_h - zeta_l) / (T_h[-1] - T_l[-1]) * (T_b - T_l[-1])
        
        # Get solution with new guess
        X, V_new, T_new = RK4_2(a, b, T_a, T_b, h, zeta_new)
        
        # Update brackets
        if T_new[-1] < T_b:
            zeta_l, T_l, V_l = zeta_new, T_new, V_new
        else:
            zeta_h, T_h, V_h = zeta_new, T_new, V_new
    
    # Return best solution
    if abs(T_h[-1] - T_b) < abs(T_l[-1] - T_b):
        return X, V_h, T_h
    else:
        return X, V_l, T_l


# Solving Heat Equations
def g(x):
    """
    Initial condition: heated to 300°C at center, 0°C elsewhere
    """
    if abs(x - 1.0) < 0.05:  # At center (x=1 for L=2)
        return 300
    else:
        return 0


def create_matrix_A(alpha=None, n=None):
    """
    Creates the evolution matrix for explicit heat equation scheme
    """
    A = []
    for i in range(n):
        row = [0] * n
        A.append(row)
    
    # Fill the tridiagonal matrix
    for i in range(n):
        A[i][i] = 1 - 2*alpha
        if i > 0:
            A[i][i-1] = alpha
        if i < n - 1:
            A[i][i+1] = alpha
    
    return A


def matmul(A=None, V=None):
    """
    Matrix-vector multiplication
    """
    C = []
    for i in range(len(V)):
        sum_val = 0
        for j in range(len(V)):
            sum_val += A[i][j] * V[j]
        C.append(sum_val)
    
    return C


def PDE_H(L=2, dx=None, dt=None, T=None):
    """
    Solves 1D heat equation using explicit scheme
    u_xx = u_t

    """
    N = int(L / dx) + 1  
    alpha = dt / (dx**2)
    
    # Create spatial grid
    X = []
    for i in range(N):
        X.append(i * dx)
    
    # Initial condition
    u0 = []
    for x in X:
        u0.append(g(x=x))
    
    U = []
    U.append(u0)
    
    A = create_matrix_A(alpha, N - 2)  
    
    # Time evolution
    for i in range(1, T):
        V_prev = U[i-1]
        
        V_interior = V_prev[1:-1]
        
        V_new_interior = matmul(A, V_interior)
        
        V_new = [0] + V_new_interior + [0]
        
        U.append(V_new)
    
     
    
    return U, X


# ============= ASSIGNMENT 16: Lagrange's Interpolation, Linear Square Fitting and Polynomial Square fitting====================

# Lagrange's Interpolation

def lagrange_interpolation(X=None,Y=None,x=None):
    N=len(X)

    p=0

    for i in range(0,N):
        c=Y[i]
        for k  in range(0,N):
            if i==k:
                m=1
            else:
                m=((x-X[k])/(X[i]-X[k]))
            
            c= m*c

        p=p+c
    
    return p



# Least Square Fitting

def least_square_fitting(X=None,Y=None,sigma=None):
    N=len(X)
    if sigma==None:
        sigma=[]
        for i in range(N):
            sigma.append(1)

    L_S=[]
    L_S_x=[]
    L_S_y=[]
    L_S_xx=[]
    L_S_yy=[]
    L_S_xy=[]
    

    for i in range(0,N):
        L_S.append(1/(sigma[i])**2)
        L_S_x.append(X[i]/sigma[i])
        L_S_y.append(Y[i]/sigma[i])
        L_S_xy.append((L_S_x[i])*(L_S_y[i]))
        L_S_xx.append((L_S_x[i])**2)
        L_S_yy.append((L_S_y[i])**2)

    S=sum(L_S)
    S_x=sum(L_S_x)
    S_y=sum(L_S_y)
    S_xx=sum(L_S_xx)
    S_yy=sum(L_S_yy)
    S_xy=sum(L_S_xy)
    delta=S*S_x - (S_x)**2

    a1=((S_xx)*(S_y) - (S_x)*(S_xy))/delta
    a2=((S_xy)*(S) - (S_x)*(S_y))/delta

    sigma_a1=math.sqrt(abs(S_xx/delta))
    sigma_a2=math.sqrt(abs(S/delta))

    r2=((S_xy)**2)/((S_xx)*(S_yy))

    return a1,sigma_a1,a2,sigma_a2,r2



# Polynomial Fitting

def polynomial_fitting(X, Y, k):
    """
    Least square polynomial fitting for data points (X, Y)
    Uses Gauss-Jordan elimination from matrix class
    
    Parameters:
    -----------
    X : list
        x-coordinates of data points
    Y : list
        y-coordinates of data points
    k : int
        Degree of polynomial (e.g., k=2 for quadratic)
    
    Returns:
    --------
    a : list
        Coefficients [a0, a1, a2, ..., ak] where 
        f(x) = a0 + a1*x + a2*x^2 + ... + ak*x^k
    """
    
    n = len(X)
    
    # Create the coefficient matrix (left side of equation)
    # Matrix dimensions: (k+1) x (k+1)
    coeff_matrix = []
    
    for i in range(k+1):
        row = []
        for j in range(k+1):
            # Each element is sum of x^(i+j)
            sum_val = 0
            for x in X:
                sum_val += x**(i+j)
            row.append(sum_val)
        coeff_matrix.append(row)
    
    # Create the constant vector (right side of equation)
    # Vector dimensions: (k+1) x 1
    const_vector = []
    
    for i in range(k+1):
        # Each element is sum of (x^i * y)
        sum_val = 0
        for x, y in zip(X, Y):
            sum_val += (x**i) * y
        const_vector.append(sum_val)
    
    # Solve the system using Gauss-Jordan elimination from matrix class
    A = matrix(coeff_matrix)
    a = A.gauss_jordan(const_vector)
    
    return a



# ================================= Functions ==================================

def f(x=None,t=None,c=None,v=None,l=2,N=None):
    "This function functions where 'x' is the input and t is the question no."
    if t==3:
        F=2.5
        r= F-x*((math.e)**x)
        return r
    if t==32:
        return -((math.e)**x) -x*((math.e)**x)
    
    if t==41:
        return x**2
    if t==42:
        return x**3
    if t==5:
        gama=0.02
        g=10
        return -gama*v-g


def g(x=None):
    return 20*abs(math.sin((math.pi)*x))


# ==================== Plotting Functions ======================================

def plot_single(data, label="Data", marker="o", linestyle="-", linewidth=2,
                title="Single Plot", xlabel="X-axis", ylabel="Y-axis",
                figsize=(8, 5), grid=True, save_path=None,
                xlim=None, ylim=None):
    """
    Function for plotting a single graph given data=[X, Y]

    Parameters:
    -----------
    data : list  -> [X, Y]
    label : str  -> Name of the line
    marker : str -> marker style ('o', 's', '^', etc.)
    linestyle : str -> line style ('-', '--', '-.', ':')
    linewidth : float -> width of line
    save_path : str or None -> location to save image (e.g. 'plot.png')

    # ========== Example Usage ==========
    data = [
        [0, 1, 2, 3, 4, 5],      # X values
        [0, 1, 4, 9, 16, 25]     # Y values
    ]

    plot_single(
        data,
        label="Quadratic",
        marker="o",
        linestyle="-",
        linewidth=2,
        title="Single Line Example",
        xlabel="Time (s)",
        ylabel="Value",
        save_path="single_plot.png"   # comment out to disable saving
    )
    
    """

    X, Y = data

    plt.figure(figsize=figsize)

    plt.plot(X, Y, marker=marker, linestyle=linestyle,
             linewidth=linewidth, label=label)

    # Basic plot formatting
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    if grid:
        plt.grid(True, linestyle="--", alpha=0.6)

    plt.legend(fontsize=12)

    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    # Save plot if required
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    plt.show()

''''
# ========== Example Usage ==========
data = [
    [0, 1, 2, 3, 4, 5],      # X values
    [0, 1, 4, 9, 16, 25]     # Y values
]

plot_single(
    data,
    label="Quadratic",
    marker="o",
    linestyle="-",
    linewidth=2,
    title="Single Line Example",
    xlabel="Time (s)",
    ylabel="Value",
    save_path="single_plot.png"   # comment out to disable saving
)
'''