import torch
import numpy as np
from scipy.linalg import eigh_tridiagonal

def tridiag_eig(A):
    """
    Computes the eigenvalue decomposition of a tridiagonal matrix using SciPy.

    Args:
    - A (torch.Tensor): N x N tridiagonal matrix in PyTorch.

    Returns:
    - eigvals (np.ndarray): Eigenvalues of A.
    - eigvecs (np.ndarray): Eigenvectors of A.
    """
    N = A.shape[0]
    
    # Extract diagonal and off-diagonal elements
    diagonal = A.diag()
    off_diagonal = A.diag(-1)

    # Convert to numpy arrays
    d = diagonal.numpy()
    e = off_diagonal.numpy()

    # Compute eigenvalue decomposition using scipy
    eigvals, eigvecs = eigh_tridiagonal(d, e)
    eigvals_tensor = torch.tensor(eigvals)
    eigvecs_tensor = torch.tensor(eigvecs)

    return eigvals_tensor, eigvecs_tensor

# Example usage:
#N = 5
# Create a tridiagonal matrix A
#main_diag = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
#off_diag = torch.tensor([0.1, 0.2, 0.3, 0.4])
#A = torch.diag(main_diag) + torch.diag(off_diag, diagonal=-1) + torch.diag(off_diag, diagonal=1)

# Compute eigenvalue decomposition
#eigenvalues, eigenvectors = tridiag_eig(A)

#print("Eigenvalues:")
#print(eigenvalues)
#print("Eigenvectors:")
#print(eigenvectors)

def lanczos_with_threshold(A, threshold=0.001, max_iter=100, tol=1e-6):
    """
    Perform the Lanczos algorithm to find the largest eigenvalues of matrix A until
    the largest eigenvalue falls below a given threshold.
    
    Parameters:
    A (torch.Tensor): The NxN symmetric matrix.
    threshold (float): The threshold below which eigenvalues are ignored.
    max_iter (int): The maximum number of iterations.
    tol (float): Tolerance for convergence.
    
    Returns:
    (torch.Tensor, torch.Tensor): The eigenvalues larger than the threshold and corresponding eigenvectors.
    """
    N = A.size(0)
    v = torch.randn(N, device=A.device)  # Start with a random vector
    v = v / torch.norm(v)  # Normalize the initial vector
    
    V = torch.zeros(N, max_iter+1, device=A.device)  # Krylov subspace vectors
    T = torch.zeros(max_iter+1, max_iter, device=A.device)  # Tridiagonal matrix elements

    beta = 0
    v_old = torch.zeros_like(v)
    #converged = False

    for j in range(max_iter):
        V[:, j] = v
        
        if j == 0:
            w = A @ v #O(N^2)
        else:
            w = A @ v - beta * v_old #O(N^2)
        
        alpha = torch.dot(v, w) #O(N)
        T[j, j] = alpha
        
        if j < max_iter-1 :
            w = w - alpha * v
            beta = torch.norm(w)
            T[j, j+1] = beta
            T[j+1, j] = beta
            
            if beta < tol:
                print('hi')
                #T_trimmed = T[:j+1, :j+1]
                break
            
            v_old = v
            v = w / beta

        # Solve eigenproblem for the tridiagonal matrix T up to j+1
        T_trimmed = T[:j+1, :j+1]
        eigenvalues, eigenvectors_T = tridiag_eig(T_trimmed) #calculating vectors is unneceesary
        #eigenvalues = torch.linalg.eigvalsh(T_trimmed) #O(J**3) should be only O(j**2) using tridiag of T_trimmed
        
        # Check if the largest eigenvalue is below the threshold
        if eigenvalues[-1] < threshold:
            print('iteration',j)
            #converged = True
            break

    # If not converged, use the last computed eigenvalues
    #if beta<tol:
    #    print('hi')

    #eigenvalues, eigenvectors_T = torch.linalg.eigh(T_trimmed)
    
    # Sort eigenvalues in descending order
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors_T = eigenvectors_T[:, sorted_indices]

    # Compute the corresponding eigenvectors of the original matrix
    eigenvectors = V[:, :j+1] @ sorted_eigenvectors_T

    # Filter out eigenvalues larger than the threshold
    filtered_indices = sorted_eigenvalues > threshold
    filtered_eigenvalues = sorted_eigenvalues[filtered_indices]
    filtered_eigenvectors = eigenvectors[:, filtered_indices]

    return filtered_eigenvalues, filtered_eigenvectors

# Example usage
N = 100
A = torch.randn(N, N)
A = A @ A.t()  # Make it symmetric

threshold = 100
max_iter = 100
tol = 1e-6

eigenvalues, eigenvectors = lanczos_with_threshold(A, threshold, max_iter, tol)

print("Eigenvalues larger than the threshold:")
print(eigenvalues)
print(len(eigenvalues))
