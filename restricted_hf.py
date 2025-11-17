# Math/science imports
import numpy as np
import numpy.typing as npt
from pyscf import gto

from typing import Union

class RHF:
    """
    Implements restricted Hartree-Fock
    """

    def __init__(self, mol: gto.M, conv: float=1e-10, iter: int=1000) -> None:
        """
        :param mol: Input molecule on which to run RHF
        :param conv: Tolerance at which a solution is considered "converged." Default 1e-8 (Hartrees).
        :param iter: Max number of iterations. Default 1000.
        """
        self.conv = conv
        self.iter = iter

        # Generate invariants
        self.h1 = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        self.h2 = mol.intor('int2e') # 2 e- integrals

        S = mol.intor('int1e_ovlp')
        self.n_elect = mol.nelectron
        self.n_occ = self.n_elect // 2
        self.n_basis = mol.nao_nr()
        self.X = self.orthogonalize_matrix(S)

        self.energy_output = [] # For graphing energy evolution of system

    @property
    def P(self) -> np.ndarray:
        """
        Final e- density (P) matrix
        """
        if not hasattr(self, "_P") or self._P is None:
            raise AttributeError("Density matrix P has not been generated. Use .run() first")
        return self._P
    
    @P.setter
    def P(self, mat: np.ndarray) -> None:
        self._P = mat

    def run(self) -> None:
        """
        Begin the SCF procedure.
        """
        # Clear Energy output for graphing
        self.energy_output = []

        # Get initial guess
        P_guess = self.inital_guess()
        G = np.zeros((self.n_basis, self.n_basis))
        energy_guess = np.sum(P_guess * self.h1) # Guess for energy of system, used for convergence test
        
        for i in range(self.iter):
            # Generate coloumb and exchange terms:
            for mu in range(self.n_basis):
                for nu in range(self.n_basis):
                    G[mu, nu] = np.sum(P_guess * (self.h2[mu,nu,:,:] - 0.5 * self.h2[mu,:,:,nu]))

            # Combine to yield Fock matrix F
            F = self.h1 + G
            P = self.generate_density(F)
            energy = np.sum(P * (self.h1 + 0.5 * G))
            self.energy_output.append(energy)

            # Check for convergence based on energy
            if np.allclose(energy, energy_guess, atol=self.conv):
                print(f"Converged in {i} iterations.")
                self.P = P
                return
            energy_guess = energy
            P_guess = P.copy()
        
        print(f"SCF did not converge within {self.conv} after {self.iter} iterations. Reporting last value...")
        self.P = P_guess

    def inital_guess(self) -> np.ndarray:
        """
        Generate an intial guess of the P charge density matrix based on the core-shell Hamiltonian (h1)
        Should be accurate enough guess for simple systems for the SCF to converge.
        """
        return self.generate_density(self.h1)

    def generate_density(self, P_guess: np.ndarray) -> np.ndarray:
        """
        Generate the density matrix P from a transformed MO coefficient matrix

        :param P_guess: Guess of P matrix that will be iterated (Normally Fock matrix)
        """
        P_guess_prime = self.transform(P_guess) # Translate into orthogonal space
        _, C_prime = np.linalg.eigh(P_guess_prime) # Calculate eigenvectors
        C_prime_occ = C_prime[:, :self.n_occ] # Find filled orbitals (eigenvecs)
        C = self.X @ C_prime_occ #Translate back into AO space
        P = 2 * (C @ C.T) # Calculate density matrix

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