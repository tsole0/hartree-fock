# Math/science imports
import numpy as np
import numpy.typing as npt
from pyscf import gto

#Utility
from typing import Union
import tqdm  # Progress bar

class UHF:
    """
    Implements unrestricted Hartree-Fock
    """

    def __init__(self, mol: gto.M, tolerance: float=1e-10, iter: int=1000) -> None:
        """
        :param mol: Input molecule on which to run RHF
        :param tolerance: Tolerance at which a solution is considered "converged." Default 1e-8 (Hartrees).
        :param iter: Max number of iterations. Default 1000.
        """
        # User parameters
        self.mol = mol
        self.tol = tolerance
        self.iter = iter

        # Generate invariants
        self.h1 = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        self.h2 = mol.intor('int2e') # 2 e- integrals

        # Base calculations
        S = mol.intor('int1e_ovlp') #overlap matrix
        self.n_basis = mol.nao_nr()
        self.X = self.orthogonalize_matrix(S)
        self.n_alpha = int((mol.spin + mol.nelectron) / 2) # Number of spin alpha e-
        self.n_beta = int((mol.nelectron - mol.spin) / 2) # Number of spin beta e-

        self.energy_output = [] # For graphing energy evolution of system

    @property
    def P_alpha(self) -> np.ndarray:
        """
        Final e- density (P) matrix
        """
        if not hasattr(self, "_P_alpha") or self._P_alpha is None:
            raise AttributeError("Density matrix P has not been generated. Use .run() first")
        return self._P_alpha
    
    @property
    def P_beta(self) -> np.ndarray:
        """
        Final e- density (P) matrix
        """
        if not hasattr(self, "_P_beta") or self._P_beta is None:
            raise AttributeError("Density matrix P has not been generated. Use .run() first")
        return self._P_beta
    
    @P_beta.setter
    def P_beta(self, mat: np.ndarray) -> None:
        self._P_beta = mat
    
    @P_alpha.setter
    def P_alpha(self, mat: np.ndarray) -> None:
        self._P_alpha = mat

    def run(self) -> None:
        """
        Begin the SCF procedure.
        """
        # Clear Energy output for graphing
        self.energy_output = []

        # Get initial guess
        P_alpha_guess = self.generate_density(self.h1, self.n_alpha)
        P_beta_guess = self.generate_density(self.h1, self.n_beta)

        # Initialize exchange, coloumb, etc. matrices
        G_alpha = np.zeros((self.n_basis, self.n_basis))
        G_beta = np.zeros((self.n_basis, self.n_basis))
        J = np.zeros((self.n_basis, self.n_basis))
        K_alpha = np.zeros((self.n_basis, self.n_basis))
        K_beta = np.zeros((self.n_basis, self.n_basis))

        energy_guess = np.sum((P_alpha_guess + P_beta_guess) * self.h1) # Guess for energy of system, used for convergence test
        
        for i in tqdm.tqdm(range(self.iter)):
            # Generate coloumb and exchange terms:
            for mu in range(self.n_basis):
                for nu in range(self.n_basis):
                    # Generate exchange and coulomb terms seperately since we need to access them separately
                    J[mu, nu] = np.sum((P_alpha_guess + P_beta_guess) * self.h2[mu,nu,:,:])
                    K_alpha[mu, nu] = np.sum(P_alpha_guess * self.h2[mu,:,:,nu])
                    K_beta[mu, nu] = np.sum(P_beta_guess * self.h2[mu,:,:,nu])
        
            # Create spin-respective G martrix
            G_alpha = J - K_alpha
            G_beta = J - K_beta

            # Combine to yield Fock matrix F
            F_alpha = self.h1 + G_alpha
            F_beta = self.h1 + G_beta
            P_alpha = self.generate_density(F_alpha, self.n_alpha)
            P_beta = self.generate_density(F_beta, self.n_beta)

            # Dampen the new guess with the linear combination of old one to kill oscillations
            damping = 0.3
            P_alpha = P_alpha_guess * damping + P_alpha * (1 - damping)
            P_beta = P_beta_guess * damping + P_beta * (1 - damping)

            energy = np.sum((P_alpha + P_beta) * (self.h1) + 0.5 * ((P_alpha * G_alpha) + (P_beta * G_beta)) )
            self.energy_output.append(energy)

            # Check for convergence based on energy
            if np.allclose(energy, energy_guess, atol=self.tol):
                print(f"Converged in {i} iterations.")
                self.P_alpha = P_alpha
                self.P_beta = P_beta
                return
            energy_guess = energy
            P_alpha_guess = P_alpha.copy()
            P_beta_guess = P_beta.copy()
        
        print(f"SCF did not converge within {self.tol} after {self.iter} iterations. Reporting last value...")
        self.P_alpha = P_alpha_guess
        self.P_beta = P_beta_guess

    def generate_density(self, P_guess: np.ndarray, n_orbit: int) -> np.ndarray:
        """
        Generate the density matrix P from a transformed MO coefficient matrix

        :param P_guess: Guess of P matrix that will be iterated (Normally Fock matrix)
        :param n_orbit: number of orbits to be solved for.
        """
        P_guess_prime = self.transform(P_guess) # Translate into orthogonal space
        _, C_prime = np.linalg.eigh(P_guess_prime) # Calculate eigenvectors
        C_prime_occ = C_prime[:, :n_orbit] # Find filled orbitals (eigenvecs)
        C = self.X @ C_prime_occ #Translate back into AO space
        P = C @ C.T # Calculate density matrix

        return P
    
    def transform(self, matrix: np.ndarray) -> np.ndarray:
        """
        Transform input matrix by the transformation matrix X.
        """
        return self.X.T @ matrix @ self.X
    
    @staticmethod
    def orthogonalize_matrix(mat: np.ndarray) -> np.ndarray:
        """
        Orthogonalize matrix using symmetrical orthogonalization:
        X = U * s^(-1/2) * U^T
        """
        eigvals, eigvecs = np.linalg.eigh(mat)
        s_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
        X = eigvecs @ s_inv_sqrt @ eigvecs.T
        
        # Verify orthogonality:
        if not np.allclose(X.T @ mat @ X, np.eye(mat.shape[0]), atol=1e-10):
            # Should be very close to identity matrix. If not, something went wrong.
            raise ValueError('Orthogonalization failed.')
        
        return X